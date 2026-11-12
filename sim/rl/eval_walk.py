"""
走路的检验：把策略放回完整身体（整体模型，mujoco 3.17，所有部位都会碰撞，膝盖用线拉，转速上限 10 rad/s），
按指令走几段，报告实际速度、摔没摔、扛得住多大的推。规矩和 eval_robust.py 一样：
每种地面摩擦各跑几次，所有接触面的摩擦一起改，起始姿势每次偏一点（腿上每个关节 ±0.03 弧度）。

用法（~/venv-sys，不需要显卡）：
    python eval_walk.py walk_policy_w1.npz [--trials=4] [--frics=0.6,1.0,1.5] [--procs=10]
    python eval_walk.py walk_policy_w1.npz --simple        # 同样的检验放在训练用的简化身体上跑（对比两个身体差多少）
    python eval_walk.py walk_policy_w1.npz --quick         # 只跑"走 10 秒再停"，每种地面 2 次
结果写 eval_walk_<策略名>[_simple].json。

检验项目：
    走停      指令 15 cm/s 走 10 秒，再指令 0 站 4 秒：实际速度、走偏多少、停下后还挪了多少
    走+推     指令 15 cm/s 走 10 秒，第 3–8 秒之间胸口被水平推一下（5 N，0.2 秒，方向随机）
    快走      指令 25 cm/s 走 8 秒
    原地转    指令 0.6 rad/s 转 6 秒
    侧移后退  指令侧移 5 cm/s 走 5 秒，再后退 8 cm/s 走 5 秒
    大推      同"走+推"，推力 8 N、12 N（只在摩擦 1.0 上）
"""
import json
import math
import os
import sys
from multiprocessing import Pool

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import walk_common as W

TESTS = {
    "走停": dict(seg=[(10.0, (0.15, 0, 0)), (4.0, (0, 0, 0))]),
    "走+推5N": dict(seg=[(10.0, (0.15, 0, 0))], push=5.0),
    "快走": dict(seg=[(8.0, (0.25, 0, 0))]),
    "原地转": dict(seg=[(6.0, (0, 0, 0.6))]),
    "侧移后退": dict(seg=[(5.0, (0, 0.05, 0)), (5.0, (-0.08, 0, 0))]),
    "走+推8N": dict(seg=[(10.0, (0.15, 0, 0))], push=8.0, only=1.0),
    "走+推12N": dict(seg=[(10.0, (0.15, 0, 0))], push=12.0, only=1.0),
}


class Simple:
    """训练用的简化身体（huaran_walk.xml），接口做成和完整身体的 FullSim 一样，只够这里用。"""

    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(os.path.join(HERE, "huaran_walk.xml"))
        self.d = mujoco.MjData(self.m)
        self.names = [self.m.actuator(i).name for i in range(self.m.nu)]
        self.trim = None

    def reset(self):
        mujoco.mj_resetData(self.m, self.d)
        dp = W.default_pose()
        for n, v in dp.items():
            self.d.qpos[self.m.jnt_qposadr[self.m.joint(n).id]] = v
        self.d.qpos[2] = W.stand_height() + 0.002
        self.tgt = np.array([dp.get(n, 0.0) for n in self.names])
        mujoco.mj_forward(self.m, self.d)

    def set_targets(self, pose):
        want = np.array([pose.get(n, 0.0) for n in self.names])
        want[self.names.index("waist_pitch")] *= 0.5
        r = W.RATE * self.m.opt.timestep
        self.tgt = self.tgt + np.clip(want - self.tgt, -r, r)
        self.d.ctrl[:] = self.tgt

    def step(self):
        mujoco.mj_step(self.m, self.d)


def run(job):
    path, simple, name, friction, seed = job
    from policy_np import Policy
    test = TESTS[name]
    pol = Policy(path)
    rng = np.random.default_rng(seed * 1000 + int(friction * 100))
    if simple:
        sim = Simple()
    else:
        import getup_hands as G
        sim = G.get_sim()
    m, d = sim.m, sim.d
    m.geom_friction[:, 0] = friction          # 所有接触面一起改（MuJoCo 取两个面里较大的摩擦）
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
        sim.trim.enabled = False               # 自动配平关掉：站稳由策略自己负责
    n_sub = int(round(W.DT / m.opt.timestep))
    last = np.zeros(12)
    ph0 = rng.uniform()
    total = sum(s for s, _ in test["seg"])
    T = int(round(total / W.DT))
    ends = np.cumsum([s for s, _ in test["seg"]])
    push_n = test.get("push")
    t_push = int(rng.uniform(3.0, 8.0) / W.DT) if push_n else -1
    ang = rng.uniform(0, 2 * math.pi)
    yaw_of = lambda: math.atan2(d.xmat[pel].reshape(3, 3)[1, 0], d.xmat[pel].reshape(3, 3)[0, 0])
    xs, yaws, fell_at = [d.qpos[0:2].copy()], [yaw_of()], None
    cscale = np.array(W.CMD_SCALE)
    peak = 0.0
    for t in range(T):
        si = int(np.searchsorted(ends, (t + 0.5) * W.DT))
        cmd = np.array(test["seg"][si][1], dtype=float)
        R = d.xmat[pel].reshape(3, 3)
        ph = ph0 + t * W.DT * W.GAIT_HZ
        a2 = 2 * math.pi * np.array([ph % 1.0, (ph + 0.5) % 1.0])
        obs = np.concatenate([-R[2, :], d.qvel[3:6] * 0.25, cmd * cscale, d.qpos[qa] - q0, d.qvel[va] * 0.05, last, np.cos(a2), np.sin(a2)])
        a = pol(obs)
        want = np.clip(q0 + W.ACT_SCALE * a, lo, hi)
        pose = dict(dp)
        pose.update({n: float(v) for n, v in zip(W.LEGS, want)})
        pushing = bool(push_n) and 0 <= t - t_push < 10
        for _ in range(n_sub):
            sim.set_targets(pose)
            d.xfrc_applied[chest, 0:2] = [push_n * math.cos(ang), push_n * math.sin(ang)] if pushing else [0.0, 0.0]
            sim.step()
        last = a
        xs.append(d.qpos[0:2].copy())
        yaws.append(yaw_of())
        up = d.xmat[pel].reshape(3, 3)[2, 2]
        if d.qpos[2] < W.FALL_Z or up < math.cos(math.radians(W.FALL_DEG)) or not np.all(np.isfinite(d.qpos)):
            fell_at = round((t + 1) * W.DT, 1)
            break
    d.xfrc_applied[chest, 0:2] = 0
    if getattr(sim, "trim", None) is not None:
        sim.trim.enabled = True
    xs, yaws = np.array(xs), np.unwrap(np.array(yaws))
    # 每一段：去掉开头 1.5 秒，量"朝着它面对的方向"的平均前进、侧移速度和转速
    segs, start = [], 0.0
    for (sec, cmd), end in zip(test["seg"], ends):
        i0, i1 = int((start + 1.5) / W.DT), min(int(end / W.DT), len(xs) - 1)
        row = dict(cmd=list(cmd), done=bool(i1 >= int(end / W.DT) - 1 and fell_at is None))
        if i1 > i0 + 10:
            dx = np.diff(xs[i0:i1 + 1], axis=0)
            y = yaws[i0:i1]
            fwd = float(np.sum(np.cos(y) * dx[:, 0] + np.sin(y) * dx[:, 1]))
            side = float(np.sum(-np.sin(y) * dx[:, 0] + np.cos(y) * dx[:, 1]))
            dur = (i1 - i0) * W.DT
            row.update(vx=round(fwd / dur, 4), vy=round(side / dur, 4), wz=round(float(yaws[i1] - yaws[i0]) / dur, 3))
            if not any(cmd):                   # 站住的那一段：最后 2 秒还挪了多少
                j0 = max(i1 - int(2.0 / W.DT), i0)
                row["drift_cm"] = round(float(np.linalg.norm(xs[i1] - xs[j0])) * 100, 2)
        segs.append(row)
        start = end
    return dict(test=name, friction=friction, seed=seed, ok=fell_at is None, fell_at=fell_at,
                push_at=round(t_push * W.DT, 1) if push_n else None, segs=segs,
                heading_deg=round(math.degrees(float(yaws[-1] - yaws[0])), 1))


def main():
    path = os.path.abspath(sys.argv[1])
    get = lambda k, d, f: next((f(a.split("=")[1]) for a in sys.argv if a.startswith(f"--{k}=")), d)
    simple, quick = "--simple" in sys.argv, "--quick" in sys.argv
    trials = get("trials", 2 if quick else 4, int)
    frics = get("frics", (0.6, 1.0, 1.5), lambda s: tuple(float(x) for x in s.split(",")))
    procs = get("procs", 10, int)
    names = ["走停"] if quick else list(TESTS)
    jobs = [(path, simple, n, f, s) for n in names for f in frics if TESTS[n].get("only", f) == f for s in range(trials)]
    with Pool(min(procs, len(jobs))) as pool:
        res = pool.map(run, jobs)
    print(f"策略 {os.path.basename(path)}，{'简化身体' if simple else '完整身体'}，每格 {trials} 次")
    for n in names:
        for f in frics:
            rs = [r for r in res if r["test"] == n and r["friction"] == f]
            if not rs:
                continue
            ok = sum(r["ok"] for r in rs)
            parts = []
            for i, (sec, cmd) in enumerate(TESTS[n]["seg"]):
                got = [r["segs"][i] for r in rs if "vx" in r["segs"][i]]
                if not got:
                    continue
                mean = lambda k: float(np.mean([g[k] for g in got]))
                s = f"指令({cmd[0]*100:.0f} cm/s, {cmd[1]*100:.0f} cm/s, {cmd[2]:.1f} rad/s) → 实际({mean('vx')*100:.1f}, {mean('vy')*100:.1f}, {mean('wz'):.2f})"
                if "drift_cm" in got[0]:
                    s += f"，最后 2 秒挪了 {mean('drift_cm'):.1f} cm"
                parts.append(s)
            fails = "；".join(f"第 {r['seed']} 次 {r['fell_at']} 秒摔" + (f"（{r['push_at']} 秒被推）" if r["push_at"] else "") for r in rs if not r["ok"])
            print(f"  {n}  摩擦 {f}：{ok}/{len(rs)} 没摔  " + "  ".join(parts) + (f"  摔：{fails}" if fails else ""))
    out = "eval_walk_" + os.path.basename(path).replace(".npz", "") + ("_simple" if simple else "") + ("_quick" if quick else "") + ".json"
    json.dump(dict(policy=os.path.basename(path), body="simple" if simple else "full", trials=trials, runs=res),
              open(os.path.join(HERE, out), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
