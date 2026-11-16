"""
"去那里"：给地面上一个目标点，花然自己转过去、走过去、停住。走路的策略不变（walk_policy.npz），
外面套一层很简单的控制，把"目标在哪"换成走路的指令（前进、侧移、转向的速度）。不用训练。
这是 VR 里"用手一指，它走过去"要用的动作。

用法（~/venv-sys，完整身体，不需要显卡）：
    python eval_goto.py walk_policy.npz [--trials=2] [--frics=0.6,1.0,1.5] [--dist=1.0] [--push=5] [--upper]
结果写 eval_goto_<策略名>[_push5][_upper].json。

规矩和 eval_walk.py 一样：完整身体，mujoco 3.17，所有接触面的摩擦一起改，起始姿势随机偏。
目标放在 8 个方向（正前、左前 45°、正左……正后……），每个方向每种地面跑 trials 次。
算"到了"：骨盆在地面上的投影离目标 5 cm 以内，而且最后 2 秒挪动不到 2 cm，没摔。

外面这层控制（goto_cmd）：
  - 目标的方位偏得多（> 35°）：先原地转，不往前走。
  - 方位差不多了：边走边修方向。前进的速度看距离（远了 20 cm/s，近了按距离减速，最慢 4 cm/s）。
  - 离目标 15 cm 以内：不再转身对准，改用前进 + 侧移直接挪过去（转身对一个很近的点会来回摆）。
  - 进了 3 cm 就停（指令全零）；被推出 6 cm 以外再重新走。
实物上要用到"自己在哪、朝哪"：朝向可以由 IMU 的陀螺积分，位置要靠别的（头显或相机看它），这里直接用了仿真里的真值。
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

V_MAX, V_MIN, W_MAX, K_YAW, VY_MAX = 0.20, 0.04, 0.8, 1.5, 0.05
TURN_FIRST, NEAR, ARRIVE, LEAVE = math.radians(35), 0.15, 0.03, 0.06
TIME_LIMIT = 30.0


def goto_cmd(dx, dy, yaw, arrived):
    """dx, dy：目标相对骨盆的位置（世界坐标，米）；yaw：现在朝哪；arrived：上一步是不是已经算到了。返回 (指令, 是否到了)。"""
    dist = math.hypot(dx, dy)
    if dist < ARRIVE or (arrived and dist < LEAVE):
        return np.zeros(3), True
    bearing = (math.atan2(dy, dx) - yaw + math.pi) % (2 * math.pi) - math.pi
    fx, fy = dist * math.cos(bearing), dist * math.sin(bearing)          # 目标在"它的前方、左方"各多远
    if dist < NEAR:
        return np.array([float(np.clip(1.5 * fx, -0.08, V_MAX)), float(np.clip(1.5 * fy, -VY_MAX, VY_MAX)), 0.0]), False
    wz = float(np.clip(K_YAW * bearing, -W_MAX, W_MAX))
    if abs(bearing) > TURN_FIRST:
        return np.array([0.0, 0.0, wz]), False
    return np.array([float(np.clip(1.0 * dist, V_MIN, V_MAX)) * math.cos(bearing), 0.0, wz]), False


def run(job):
    path, direction, friction, seed, opts = job
    from policy_np import Policy
    import getup_hands as G
    pol = Policy(path)
    rng = np.random.default_rng(seed * 1000 + int(friction * 100) + direction * 17)
    sim = G.get_sim()
    m, d = sim.m, sim.d
    m.geom_friction[:, 0] = friction
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
    yaw_of = lambda: math.atan2(d.xmat[pel].reshape(3, 3)[1, 0], d.xmat[pel].reshape(3, 3)[0, 0])
    a0 = yaw_of() + direction * math.pi / 4
    goal = d.qpos[0:2] + opts["dist"] * np.array([math.cos(a0), math.sin(a0)])
    last, ph0 = np.zeros(12), rng.uniform()
    push_n = opts["push"]
    t_push = int(rng.uniform(2.0, 6.0) / W.DT) if push_n else -1
    ang = rng.uniform(0, 2 * math.pi)
    upose, up_off = {}, int(rng.integers(0, W.UPPER_EVERY))
    cscale = np.array(W.CMD_SCALE)
    arrived, t_arrive, fell_at, xs = False, None, None, []
    T = int(TIME_LIMIT / W.DT)
    for t in range(T):
        cmd, arrived = goto_cmd(goal[0] - d.qpos[0], goal[1] - d.qpos[1], yaw_of(), arrived)
        if arrived and t_arrive is None:
            t_arrive = t * W.DT
        if t_arrive is not None and t * W.DT > t_arrive + 3.0:      # 到了以后再看 3 秒
            break
        if opts["upper"] and t % W.UPPER_EVERY == up_off:
            upose = {n: dp.get(n, 0.0) for n in W.UPPER} if rng.uniform() < W.UPPER_REST else {n: rng.uniform(a, b) for n, (a, b) in W.UPPER.items()}
            upose["waist_pitch"] *= 2.0
        R = d.xmat[pel].reshape(3, 3)
        ph = ph0 + t * W.DT * W.GAIT_HZ
        a2 = 2 * math.pi * np.array([ph % 1.0, (ph + 0.5) % 1.0])
        obs = np.concatenate([-R[2, :], d.qvel[3:6] * 0.25, cmd * cscale, d.qpos[qa] - q0, d.qvel[va] * 0.05, last, np.cos(a2), np.sin(a2)])
        a = pol(obs)
        pose = dict(dp)
        pose.update(upose)
        pose.update({n: float(v) for n, v in zip(W.LEGS, np.clip(q0 + W.ACT_SCALE * a, lo, hi))})
        pushing = bool(push_n) and 0 <= t - t_push < 10
        for _ in range(n_sub):
            sim.set_targets(pose)
            d.xfrc_applied[chest, 0:2] = [push_n * math.cos(ang), push_n * math.sin(ang)] if pushing else [0.0, 0.0]
            sim.step()
        last = a
        xs.append(d.qpos[0:2].copy())
        if d.qpos[2] < W.FALL_Z or d.xmat[pel].reshape(3, 3)[2, 2] < math.cos(math.radians(W.FALL_DEG)) or not np.all(np.isfinite(d.qpos)):
            fell_at = round((t + 1) * W.DT, 1)
            break
    d.xfrc_applied[chest, 0:2] = 0
    if getattr(sim, "trim", None) is not None:
        sim.trim.enabled = True
    xs = np.array(xs)
    err = float(np.linalg.norm(xs[-1] - goal))
    drift = float(np.linalg.norm(xs[-1] - xs[max(0, len(xs) - int(2.0 / W.DT))]))
    path_len = float(np.sum(np.linalg.norm(np.diff(xs, axis=0), axis=1)))
    ok = fell_at is None and err < 0.05 and drift < 0.02 and t_arrive is not None
    return dict(direction_deg=direction * 45, friction=friction, seed=seed, ok=bool(ok), fell_at=fell_at,
                t_arrive=None if t_arrive is None else round(t_arrive, 1), err_cm=round(err * 100, 1), drift_cm=round(drift * 100, 1),
                path_m=round(path_len, 2), push_at=round(t_push * W.DT, 1) if push_n else None)


def main():
    path = os.path.abspath(sys.argv[1])
    get = lambda k, dflt, f: next((f(a.split("=")[1]) for a in sys.argv if a.startswith(f"--{k}=")), dflt)
    trials = get("trials", 2, int)
    frics = get("frics", (0.6, 1.0, 1.5), lambda s: tuple(float(x) for x in s.split(",")))
    opts = dict(dist=get("dist", 1.0, float), push=get("push", 0.0, float), upper="--upper" in sys.argv)
    jobs = [(path, k, f, s, opts) for f in frics for k in range(8) for s in range(trials)]
    with Pool(min(get("procs", 10, int), len(jobs))) as pool:
        res = pool.map(run, jobs)
    print(f"策略 {os.path.basename(path)}，完整身体，目标在 {opts['dist']} m 外的 8 个方向，每个方向 {trials} 次"
          + (f"，途中被推一次 {opts['push']:.0f} N" if opts["push"] else "") + ("，上身随意摆" if opts["upper"] else ""))
    for f in frics:
        rs = [r for r in res if r["friction"] == f]
        got = [r for r in rs if r["t_arrive"] is not None]
        fails = "；".join(f"{r['direction_deg']}° 第 {r['seed']} 次" + (f" {r['fell_at']} 秒摔" if r["fell_at"] else f" 离目标 {r['err_cm']} cm、还在挪 {r['drift_cm']} cm")
                         for r in rs if not r["ok"])
        print(f"  摩擦 {f}：{sum(r['ok'] for r in rs)}/{len(rs)} 到了并停住，摔 {sum(r['fell_at'] is not None for r in rs)} 次；"
              f"用时平均 {np.mean([r['t_arrive'] for r in got]):.1f} 秒（最长 {np.max([r['t_arrive'] for r in got]):.1f}），"
              f"停下后离目标平均 {np.mean([r['err_cm'] for r in rs if r['fell_at'] is None]):.1f} cm，实际走了 {np.mean([r['path_m'] for r in got]):.2f} m"
              + (f"  没成的：{fails}" if fails else ""))
    out = "eval_goto_" + os.path.basename(path).replace(".npz", "") + (f"_push{opts['push']:.0f}" if opts["push"] else "") + ("_upper" if opts["upper"] else "") + ".json"
    json.dump(dict(policy=os.path.basename(path), opts=opts, trials=trials, runs=res), open(os.path.join(HERE, out), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
