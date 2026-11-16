"""
学会的走路（sim/rl/walk_policy.npz）在完整身体上走一段约 37 秒的"巡演"，逐帧录下来给 Walk.html 回放。

用法（~/venv-sys，mujoco 3.17，不需要显卡）：
    python walk_replay.py            # 写 ../walk_frames.js，并打印每一段的指令和实际速度

控制循环照 sim/rl/eval_walk.py 的 run()：观测 49 个数（重力方向、骨盆角速度、指令、腿上 12 个关节角和角速度、上一步输出、节拍），
目标角 = 站姿 + 0.5 × 策略输出，每 20 ms 一次；自动配平关掉；手臂、腰、脖子固定在站姿。
地面摩擦用整体模型默认的 1.0。起始姿势腿上每个关节随机偏 ±0.03 弧度、节拍起点随机、被推的方向随机，
都由固定的种子 SEED=0 决定（第一次就用 0，没有挑种子）。中途摔了也照样录下去并记下时刻。

帧格式沿用 sim/full_demo.py 的 Recorder（每 1/30 秒一帧，g = 每个几何体的位置 mm + 四元数），另外每帧记：
    seg   当前是第几段
    cmd   当前指令 [前进 cm/s, 侧移 cm/s, 转向 rad/s]
    act   实际速度（1 秒滑动平均，按它面对的方向量）[前进 cm/s, 侧移 cm/s, 转向 rad/s]
    push  是否正在被推
    p     骨盆位置 [x, y] mm（镜头跟随、画走过的路线）
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
from policy_np import Policy     # noqa: E402

FPS = 30
SEED = 0
PUSH_N, PUSH_S = 5.0, 0.2
# (秒, (前进 m/s, 侧移 m/s, 转向 rad/s), 名字, 这一段第几秒开始推（None = 不推）)
TOUR = [
    (2.0, (0.0, 0.0, 0.0), "站着", None),
    (8.0, (0.15, 0.0, 0.0), "往前走 15 cm/s", None),
    (2.0, (0.0, 0.0, 0.0), "停", None),
    (5.0, (0.0, 0.0, 0.6), "原地左转 0.6 rad/s", None),
    (6.0, (0.25, 0.0, 0.0), "快走 25 cm/s", 3.0),
    (2.0, (0.0, 0.0, 0.0), "停", None),
    (4.0, (0.0, 0.05, 0.0), "往左侧移 5 cm/s", None),
    (5.0, (-0.08, 0.0, 0.0), "后退 8 cm/s", None),
    (3.0, (0.0, 0.0, 0.0), "停", None),
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
    every = int(round(1 / FPS / m.opt.timestep))
    ph0 = rng.uniform()
    ang = float(rng.uniform(0, 2 * math.pi))       # 推的方向（世界坐标，0 = +x）
    cscale = np.array(W.CMD_SCALE)
    yaw_of = lambda: math.atan2(d.xmat[pel].reshape(3, 3)[1, 0], d.xmat[pel].reshape(3, 3)[0, 0])   # noqa: E731

    ends = np.cumsum([s for s, *_ in TOUR])
    starts = ends - np.array([s for s, *_ in TOUR])
    total = float(ends[-1])
    seg_push = [i for i, x in enumerate(TOUR) if x[3] is not None][0]
    t_push0 = float(starts[seg_push] + TOUR[seg_push][3])
    T = int(round(total / W.DT))
    win = deque(maxlen=int(1.0 / W.DT) + 1)        # 1 秒滑动窗口：(x, y, 朝向)
    state = dict(seg=0, cmd=[0, 0, 0], act=[0.0, 0.0, 0.0], push=False)
    frames, k = [], [0]
    fell_at, fell_reason = None, None
    yaw_prev, yaw_acc = yaw_of(), 0.0

    def rec():
        k[0] += 1
        if k[0] % every:
            return
        g = []
        for i in range(m.ngeom):
            q = np.zeros(4)
            mujoco.mju_mat2Quat(q, d.geom_xmat[i])
            g.append([round(float(x) * 1000, 1) for x in d.geom_xpos[i]] + [round(float(x), 3) for x in q])
        frames.append(dict(t=round(len(frames) / FPS, 3), g=g, seg=state["seg"], cmd=state["cmd"], act=state["act"],
                           push=state["push"], p=[round(float(d.qpos[0]) * 1000, 1), round(float(d.qpos[1]) * 1000, 1)],
                           tilt=round(sim.tilt_deg(), 1)))
    sim.recorder = rec
    last = np.zeros(12)
    push_rel = [None]
    segrows = [dict(name=x[2], cmd=[round(x[1][0] * 100, 1), round(x[1][1] * 100, 1), x[1][2]], xs=[]) for x in TOUR]
    for t in range(T):
        tt = (t + 0.5) * W.DT
        si = int(np.searchsorted(ends, tt))
        cmd = np.array(TOUR[si][1], dtype=float)
        R = d.xmat[pel].reshape(3, 3)
        ph = ph0 + t * W.DT * W.GAIT_HZ
        a2 = 2 * math.pi * np.array([ph % 1.0, (ph + 0.5) % 1.0])
        obs = np.concatenate([-R[2, :], d.qvel[3:6] * 0.25, cmd * cscale, d.qpos[qa] - q0, d.qvel[va] * 0.05, last, np.cos(a2), np.sin(a2)])
        a = pol(obs)
        want = np.clip(q0 + W.ACT_SCALE * a, lo, hi)
        pose = dict(dp)
        pose.update({n: float(v) for n, v in zip(W.LEGS, want)})
        pushing = t_push0 <= t * W.DT < t_push0 + PUSH_S - 1e-9
        if pushing and push_rel[0] is None:
            push_rel[0] = (math.degrees(ang - yaw_of()) + 180) % 360 - 180    # 相对它面对的方向：0 = 从背后往前推，90 = 往它左边推
        state.update(seg=si, cmd=[round(cmd[0] * 100, 1), round(cmd[1] * 100, 1), round(float(cmd[2]), 2)], push=bool(pushing))
        for _ in range(n_sub):
            sim.set_targets(pose)
            d.xfrc_applied[chest, 0:2] = [PUSH_N * math.cos(ang), PUSH_N * math.sin(ang)] if pushing else [0.0, 0.0]
            sim.step()
        last = a
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
        if tt > starts[si] + 1.5:
            segrows[si]["xs"].append(state["act"])
        up = d.xmat[pel].reshape(3, 3)[2, 2]
        if fell_at is None and (d.qpos[2] < W.FALL_Z or up < math.cos(math.radians(W.FALL_DEG)) or not np.all(np.isfinite(d.qpos))):
            fell_at = round((t + 1) * W.DT, 2)
            fell_reason = f"骨盆高 {d.qpos[2]*100:.1f} cm，躯干倾斜 {math.degrees(math.acos(max(-1, min(1, up)))):.0f}°"
            print(f"!! {fell_at} 秒摔了（{TOUR[si][2]}）：{fell_reason}", flush=True)
        if not np.all(np.isfinite(d.qpos)):
            break
    sim.recorder = None
    d.xfrc_applied[chest, 0:2] = 0
    if getattr(sim, "trim", None) is not None:
        sim.trim.enabled = True

    segs = []
    for (sec, cmd, name, _), s0, row in zip(TOUR, starts, segrows):
        xs = np.array(row["xs"]) if row["xs"] else np.zeros((1, 3))
        mean = [round(float(v), 2) for v in xs.mean(axis=0)]
        segs.append(dict(name=name, t0=float(s0), t1=float(s0 + sec), cmd=row["cmd"], act_mean=mean))
        print(f"{s0:5.1f}–{s0+sec:5.1f} s  {name:<14} 指令 {row['cmd']}  实际（去掉每段开头 1.5 秒的平均）{mean}")
    heading = math.degrees(yaw_acc)
    dist = float(np.hypot(d.qpos[0], d.qpos[1]))
    print(f"终点离起点 {dist*100:.0f} cm，朝向累计转了 {heading:.0f}°；摔：{fell_at}；推：{t_push0} 秒，方向 {math.degrees(ang):.0f}°")

    geoms = [{"name": m.geom(i).name, "type": int(m.geom_type[i]), "size": [round(float(x) * 1000, 3) for x in m.geom_size[i]],
              "rgba": [round(float(x), 3) for x in m.geom_rgba[i]]} for i in range(m.ngeom)]
    out = dict(fps=FPS, policy="sim/rl/walk_policy.npz", seed=SEED, friction=1.0, geoms=geoms, frames=frames, segs=segs,
               push=dict(t=t_push0, dur=PUSH_S, N=PUSH_N, dir_deg=round(math.degrees(ang), 1),
                         rel_deg=None if push_rel[0] is None else round(push_rel[0], 1)),
               fell_at=fell_at, fell_reason=fell_reason, end=dict(dist_cm=round(dist * 100, 1), heading_deg=round(heading, 1)))
    path = os.path.join(HERE, "..", "walk_frames.js")
    with open(path, "w", encoding="utf-8") as f:
        f.write("window.WALK = " + json.dumps(out, ensure_ascii=False, separators=(",", ":")) + ";\n")
    print(f"已导出 walk_frames.js（{os.path.getsize(path)/1e6:.1f} MB，{len(frames)} 帧）")


if __name__ == "__main__":
    main()
