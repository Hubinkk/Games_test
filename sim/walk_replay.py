"""
学会的走路（sim/rl/walk_policy.npz，第 2 轮）在完整身体上走一段"巡演"，逐帧录下来给 Walk.html 回放。

用法（~/venv-sys，mujoco 3.17，不需要显卡）：
    python walk_replay.py            # 写 ../walk_frames.js，并打印每一段的指令、实际速度和"去那里"的结果

控制循环照 sim/rl/eval_walk.py 的 run()：观测 49 个数（重力方向、骨盆角速度、指令、腿上 12 个关节角和角速度、上一步输出、节拍），
目标角 = 站姿 + 0.5 × 策略输出，每 20 ms 一次；自动配平关掉。地面摩擦用整体模型默认的 1.0。
外面套的几层（策略本身不变，都照 sim/rl/ 里的检验脚本）：
  - 朝向控制（eval_walk.py 的 --hold）：按指令走的段落里，把"该朝哪边"按指令的转速累加，偏了就在转向指令上补
    （每偏 1 弧度补 1.5 rad/s，限 ±0.8）。每个按指令走的段落开始时，"该朝哪边"重设为它当时的朝向。
  - "去那里"（eval_goto.py 的 goto_cmd）：目标点按它当时的位置和朝向摆（比如"左前方 0.8 m"），
    指令由 goto_cmd 算；到了以后再站 1.5 秒进入下一段，最长 15 秒。位置和朝向用的是仿真里的真值。
  - 上身被别处指挥（eval_walk.py 的 --upper）：腰、脖子、手臂每 2 秒换一个随机姿势（范围 walk_common.UPPER，
    三成的时候回到自然下垂，腰俯仰 ×2 因为姿势里写的是腰两段合起来的角），策略看不到。其余段落上身固定在站姿。
起始姿势腿上每个关节随机偏 ±0.03 弧度、节拍起点随机、被推的方向随机、上身的姿势都由固定的种子 SEED=0 决定
（第一次就用 0，没有挑种子）。中途摔了也照样录下去并记下时刻。

帧格式沿用 sim/full_demo.py 的 Recorder（g = 每个几何体的位置 mm + 四元数），帧率 30；整个文件超过 6 MB 就降到 20。另外每帧记：
    seg   当前是第几段
    cmd   给策略的指令（含朝向修正、"去那里"算出来的）[前进 cm/s, 侧移 cm/s, 转向 rad/s]
    act   实际速度（1 秒滑动平均，按它面对的方向量）[前进 cm/s, 侧移 cm/s, 转向 rad/s]
    push  正在被推的那一次的序号（-1 = 没在推）
    p     骨盆位置 [x, y] mm；yaw 朝向（度，累计）
    tilt  躯干倾斜（度）
"""
import json
import math
import os
import sys
from collections import deque

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "rl"))
import getup_hands as G          # noqa: E402
import walk_common as W          # noqa: E402
from eval_goto import goto_cmd   # noqa: E402
from policy_np import Policy     # noqa: E402

SEED = 0
HOLD_K = 1.5
MAX_MB = 6.0
PUSH_S = 0.2
GOTO_AFTER, GOTO_MAX = 1.5, 15.0
# 每一段：name；dur（秒，"去那里"是最长时间）；cmd（前进 m/s, 侧移 m/s, 转向 rad/s）或 goal（相对它当时的 [前方 m, 左方 m]）；
# upper = 上身被随意摆；push = (这一段第几秒开始推, 推力 N)
TOUR = [
    dict(name="站着", dur=2.0, cmd=(0, 0, 0)),
    dict(name="往前走 15 cm/s", dur=8.0, cmd=(0.15, 0, 0)),
    dict(name="停", dur=2.0, cmd=(0, 0, 0)),
    dict(name="原地左转 0.6 rad/s", dur=5.0, cmd=(0, 0, 0.6)),
    dict(name="快走 25 cm/s", dur=6.0, cmd=(0.25, 0, 0), push=(3.0, 5.0)),
    dict(name="停", dur=2.0, cmd=(0, 0, 0)),
    dict(name="往左侧移 5 cm/s", dur=4.0, cmd=(0, 0.05, 0)),
    dict(name="后退 8 cm/s", dur=5.0, cmd=(-0.08, 0, 0)),
    dict(name="站着被推 8 N", dur=4.0, cmd=(0, 0, 0), push=(1.5, 8.0)),
    dict(name="去那里：左前方 0.8 m", dur=GOTO_MAX, goal=(0.8 * math.cos(math.radians(45)), 0.8 * math.sin(math.radians(45)))),
    dict(name="去那里：正后方 0.6 m", dur=GOTO_MAX, goal=(-0.6, 0.0)),
    dict(name="去那里：右边 0.7 m", dur=GOTO_MAX, goal=(0.0, -0.7)),
    dict(name="上身在动，走 15 cm/s", dur=8.0, cmd=(0.15, 0, 0), upper=True),
    dict(name="站着，上身和手臂在动", dur=4.0, cmd=(0, 0, 0), upper=True),
    dict(name="停", dur=2.0, cmd=(0, 0, 0)),
]


def main():
    pol = Policy(os.path.join(HERE, "rl", "walk_policy.npz"))
    rng = np.random.default_rng(SEED)
    sim = G.get_sim()
    m, d = sim.m, sim.d
    jid = [m.joint(n).id for n in W.LEGS]
    qa = np.array([m.jnt_qposadr[j] for j in jid])
    va = np.array([m.jnt_dofadr[j] for j in jid])
    lo, hi = m.jnt_range[jid, 0], m.jnt_range[jid, 1]
    pel, chest = m.body("pelvis").id, m.body("chest").id
    dp = W.default_pose()
    q0 = np.array([dp.get(n, 0.0) for n in W.LEGS])
    sim.reset()
    d.qpos[qa] = np.clip(d.qpos[qa] + rng.uniform(-0.03, 0.03, 12), lo, hi)
    mujoco.mj_forward(m, d)
    if getattr(sim, "trim", None) is not None:
        sim.trim.enabled = False
    n_sub = int(round(W.DT / m.opt.timestep))
    ts = m.opt.timestep
    ph0 = rng.uniform()
    cscale = np.array(W.CMD_SCALE)
    yaw_of = lambda: math.atan2(d.xmat[pel].reshape(3, 3)[1, 0], d.xmat[pel].reshape(3, 3)[0, 0])   # noqa: E731

    win = deque(maxlen=int(1.0 / W.DT) + 1)        # 1 秒滑动窗口：(x, y, 朝向)
    state = dict(seg=0, cmd=[0, 0, 0], act=[0.0, 0.0, 0.0], push=-1)
    frames = {30: [], 20: []}
    nstep = [0]
    yaw_prev, yaw_acc = yaw_of(), 0.0

    def snap():
        g = []
        for i in range(m.ngeom):
            q = np.zeros(4)
            mujoco.mju_mat2Quat(q, d.geom_xmat[i])
            g.append([round(float(x) * 1000, 1) for x in d.geom_xpos[i]] + [round(float(x), 3) for x in q])
        return dict(g=g, seg=state["seg"], cmd=state["cmd"], act=state["act"], push=state["push"],
                    p=[round(float(d.qpos[0]) * 1000, 1), round(float(d.qpos[1]) * 1000, 1)],
                    yaw=round(math.degrees(yaw_acc), 1), tilt=round(sim.tilt_deg(), 1))

    def rec():
        nstep[0] += 1
        tnow = nstep[0] * ts
        cache = None
        for fps, fl in frames.items():
            if tnow + 1e-9 >= len(fl) / fps:
                cache = cache or snap()
                fl.append(dict(cache, t=round(len(fl) / fps, 3)))
    sim.recorder = rec

    last = np.zeros(12)
    pushes, goals, segs = [], [], []
    fell_at, fell_reason = None, None
    t = 0                      # 策略步数
    for si, sg in enumerate(TOUR):
        t0 = t * W.DT
        state["seg"] = si
        yaw_ref = yaw_of()
        upose, rows = {}, []
        goal, arrived, t_arrive = None, False, None
        if "goal" in sg:
            yw = yaw_of()
            fx, fy = sg["goal"]
            goal = d.qpos[0:2] + np.array([fx * math.cos(yw) - fy * math.sin(yw), fx * math.sin(yw) + fy * math.cos(yw)])
            gi = len(goals)
            goals.append(dict(name=sg["name"], seg=si, x=round(float(goal[0]) * 1000, 1), y=round(float(goal[1]) * 1000, 1), t0=round(t0, 2)))
        if "push" in sg:
            ang = float(rng.uniform(0, 2 * math.pi))
            pt0 = t0 + sg["push"][0]
            pushes.append(dict(t=round(pt0, 2), dur=PUSH_S, N=sg["push"][1], dir_deg=round(math.degrees(ang), 1), seg=si, rel_deg=None))
        n_steps = int(round(sg["dur"] / W.DT))
        for k in range(n_steps):
            tt = t * W.DT
            if goal is not None:
                cmd, arrived = goto_cmd(goal[0] - d.qpos[0], goal[1] - d.qpos[1], yaw_of(), arrived)
                if arrived and t_arrive is None:
                    t_arrive = tt
                if t_arrive is not None and tt >= t_arrive + GOTO_AFTER:
                    break
            else:
                cmd = np.array(sg["cmd"], dtype=float)
                if cmd.any():
                    yaw_ref += cmd[2] * W.DT
                    err = (yaw_ref - yaw_of() + math.pi) % (2 * math.pi) - math.pi
                    cmd[2] = float(np.clip(cmd[2] + HOLD_K * err, -0.8, 0.8))
            if sg.get("upper") and k % W.UPPER_EVERY == 0:
                upose = {n: dp.get(n, 0.0) for n in W.UPPER} if rng.uniform() < W.UPPER_REST else {n: float(rng.uniform(a, b)) for n, (a, b) in W.UPPER.items()}
                upose["waist_pitch"] *= 2.0
            R = d.xmat[pel].reshape(3, 3)
            ph = ph0 + t * W.DT * W.GAIT_HZ
            a2 = 2 * math.pi * np.array([ph % 1.0, (ph + 0.5) % 1.0])
            obs = np.concatenate([-R[2, :], d.qvel[3:6] * 0.25, cmd * cscale, d.qpos[qa] - q0, d.qvel[va] * 0.05, last, np.cos(a2), np.sin(a2)])
            a = pol(obs)
            pose = dict(dp)
            pose.update(upose)
            pose.update({n: float(v) for n, v in zip(W.LEGS, np.clip(q0 + W.ACT_SCALE * a, lo, hi))})
            pushing = "push" in sg and pushes[-1]["t"] - 1e-9 <= tt < pushes[-1]["t"] + PUSH_S - 1e-9
            if pushing and pushes[-1]["rel_deg"] is None:
                pushes[-1]["rel_deg"] = round((math.degrees(ang - yaw_of()) + 180) % 360 - 180, 1)   # 0 = 从背后往前推，90 = 往它左边推
            state.update(cmd=[round(float(cmd[0]) * 100, 1), round(float(cmd[1]) * 100, 1), round(float(cmd[2]), 2)],
                         push=len(pushes) - 1 if pushing else -1)
            pn = sg["push"][1] if pushing else 0.0
            for _ in range(n_sub):
                sim.set_targets(pose)
                d.xfrc_applied[chest, 0:2] = [pn * math.cos(ang), pn * math.sin(ang)] if pushing else [0.0, 0.0]
                sim.step()
            last = a
            t += 1
            y = yaw_of()
            yaw_acc += (y - yaw_prev + math.pi) % (2 * math.pi) - math.pi
            yaw_prev = y
            win.append((float(d.qpos[0]), float(d.qpos[1]), yaw_acc))
            if len(win) > 1:
                (x0, y0, w0), (x1, y1, w1) = win[0], win[-1]
                dur = (len(win) - 1) * W.DT
                h = 0.5 * (w0 + w1)
                dx, dy = x1 - x0, y1 - y0
                state["act"] = [round((math.cos(h) * dx + math.sin(h) * dy) / dur * 100, 1),
                                round((-math.sin(h) * dx + math.cos(h) * dy) / dur * 100, 1), round((w1 - w0) / dur, 2)]
            if t * W.DT > t0 + 1.5:
                rows.append(state["act"])
            up = d.xmat[pel].reshape(3, 3)[2, 2]
            if fell_at is None and (d.qpos[2] < W.FALL_Z or up < math.cos(math.radians(W.FALL_DEG)) or not np.all(np.isfinite(d.qpos))):
                fell_at = round(t * W.DT, 2)
                fell_reason = f"骨盆高 {d.qpos[2]*100:.1f} cm，躯干倾斜 {math.degrees(math.acos(max(-1, min(1, up)))):.0f}°（{sg['name']}）"
                print(f"!! {fell_at} 秒摔了：{fell_reason}", flush=True)
            if not np.all(np.isfinite(d.qpos)):
                break
        t1 = t * W.DT
        xs = np.array(rows) if rows else np.zeros((1, 3))
        mean = [round(float(v), 2) for v in xs.mean(axis=0)]
        row = dict(name=sg["name"], t0=round(t0, 2), t1=round(t1, 2), act_mean=mean, upper=bool(sg.get("upper")),
                   cmd=None if goal is not None else [round(sg["cmd"][0] * 100, 1), round(sg["cmd"][1] * 100, 1), sg["cmd"][2]])
        if goal is not None:
            err = float(np.linalg.norm(d.qpos[0:2] - goal))
            goals[-1].update(t_arrive=None if t_arrive is None else round(t_arrive, 2), took=None if t_arrive is None else round(t_arrive - t0, 1),
                             err_end_cm=round(err * 100, 1), reached=t_arrive is not None and err < 0.05)
            row["goal"] = gi
            print(f"{t0:5.1f}–{t1:5.1f} s  {sg['name']:<16} {'到了' if goals[-1]['reached'] else '没到'}：用 {goals[-1]['took']} 秒，最后离目标 {err*100:.1f} cm")
        else:
            print(f"{t0:5.1f}–{t1:5.1f} s  {sg['name']:<16} 指令 {row['cmd']}  实际（去掉开头 1.5 秒的平均）{mean}")
        segs.append(row)
    sim.recorder = None
    d.xfrc_applied[chest, 0:2] = 0
    if getattr(sim, "trim", None) is not None:
        sim.trim.enabled = True

    heading = math.degrees(yaw_acc)
    dist = float(np.hypot(d.qpos[0], d.qpos[1]))
    print(f"一共 {t*W.DT:.1f} 秒；终点离起点 {dist*100:.0f} cm，朝向累计转了 {heading:.0f}°；摔：{fell_at}")
    for p in pushes:
        print(f"推：{p['t']} 秒 {p['N']} N，世界方向 {p['dir_deg']}°，相对它 {p['rel_deg']}°")

    geoms = [{"name": m.geom(i).name, "type": int(m.geom_type[i]), "size": [round(float(x) * 1000, 3) for x in m.geom_size[i]],
              "rgba": [round(float(x), 3) for x in m.geom_rgba[i]]} for i in range(m.ngeom)]
    path = os.path.join(HERE, "..", "walk_frames.js")
    for fps in (30, 20):
        out = dict(fps=fps, policy="sim/rl/walk_policy.npz（第 2 轮，walk_policy_w2_81M）", seed=SEED, friction=1.0, geoms=geoms,
                   frames=frames[fps], segs=segs, pushes=pushes, goals=goals, fell_at=fell_at, fell_reason=fell_reason,
                   end=dict(dist_cm=round(dist * 100, 1), heading_deg=round(heading, 1), total_s=round(t * W.DT, 2)))
        s = "window.WALK = " + json.dumps(out, ensure_ascii=False, separators=(",", ":")) + ";\n"
        if len(s.encode("utf-8")) <= MAX_MB * 1e6 or fps == 20:
            break
        print(f"帧率 {fps} 时 {len(s.encode('utf-8'))/1e6:.1f} MB，超过 {MAX_MB} MB，降到 20")
    with open(path, "w", encoding="utf-8") as f:
        f.write(s)
    print(f"已导出 walk_frames.js（{os.path.getsize(path)/1e6:.1f} MB，{len(out['frames'])} 帧，{fps} 帧/秒）")


if __name__ == "__main__":
    main()
