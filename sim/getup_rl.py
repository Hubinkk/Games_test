"""
用强化学习学到的策略起身（给 day.py 的 --getup=rl 用）。

控制循环和 sim/rl/eval_full.py 的 run 一样：
  - 观测：骨盆的"下"方向、骨盆角速度×0.25、26 个关节角、关节角速度×0.1、上一步动作、参考动作这一步的角、相位（3 个数）
  - 目标角 = 参考动作 + 0.3 × 策略输出，限在关节范围内；每 20 ms 最多变 0.2 弧度
  - waist_pitch 乘 2（整体模型的接口里腰写的是两段加起来的角度）
  - 共 T+25 步（参考动作 894 步 × 20 ms，再多 0.5 秒）；其间自动配平（sim.trim）关掉，结束后打开
  - 不接手写的平衡控制
起始姿势和训练时一样：脸朝下趴着，头朝 +x（和 getup_hands.simulate 的 start="prone" 相同）。
策略只练过从这个姿势开始。

用法（单独跑一次，在默认地面上）：
    python getup_rl.py [--side=L|R]
"""
import json
import math
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RL = os.path.join(HERE, "rl")
sys.path.insert(0, RL)
from policy_np import Policy     # noqa: E402

ACT_SCALE, SLEW, EXTRA = 0.3, 0.2, 25
# 头偏向一边（2026-11-12 起由策略自己做）：参考动作 rl/ref_getup_turnL.json 里写了"起身的头 4.5 秒脖子转向左边并低头，之后回正"，
# 策略（getup_policy.npz = getup_policy_turn1_4M.npz）跟着它转，侧面的头发着地，脸不碰地。
# 转右 = 把转左整条左右镜像（rl/mirror_lr.py）：参考动作镜像，策略看到的观测镜像、给出的动作再镜像回来。
# SIDE："L"、"R"，或 None = 每次起身随机选一边。
SIDE = None
# 以前（2026-11-11）的做法是起身的头 4.5 秒强行把脖子设成下面这个、不听策略的。现在关掉（None）；
# 只在用旧策略 rl/run3_final.npz + 原来的 ref_getup.json 时才需要。
TURN_HEAD = None
TURN_UNTIL = 4.5
_POL, _REF = None, None


def load():
    global _POL, _REF
    if _POL is None:
        _POL = Policy(os.path.join(RL, "getup_policy.npz"))
        _REF = json.load(open(os.path.join(RL, "ref_getup_turnL.json"), encoding="utf-8"))
    return _POL, _REF


def duration():
    """仿真里一共跑多少秒。"""
    _, ref = load()
    return (len(ref["q"]) + EXTRA) * ref["dt"]


def run(sim, verbose=False, side=None):
    """side：头转向哪边（"L" / "R"）；不给时用 SIDE，SIDE 也是 None 就随机。"""
    pol, ref = load()
    side = side or SIDE or ("L" if np.random.random() < 0.5 else "R")
    m, d = sim.m, sim.d
    names = ref["names"]
    q_ref = np.array(ref["q"])
    if side == "R":
        from mirror_lr import Mirror
        P = Mirror(names)
        q_ref = P.q(q_ref)
        pol0 = pol
        pol = lambda o: P.q(pol0(P.obs(o)))      # noqa: E731
    T = len(q_ref)
    jid = [m.joint(n).id for n in names]
    qa = np.array([m.jnt_qposadr[j] for j in jid])
    va = np.array([m.jnt_dofadr[j] for j in jid])
    lo, hi = m.jnt_range[jid, 0], m.jnt_range[jid, 1]
    pel = m.body("pelvis").id
    sim.reset()
    d.qpos[0:3] = [0, 0, 0.06]
    d.qpos[3:7] = [math.cos(math.pi / 4), 0, math.sin(math.pi / 4), 0]
    mujoco.mj_forward(m, d)
    trim = getattr(sim, "trim", None)
    if trim is not None:
        trim.enabled = False
    n_sub = int(round(ref["dt"] / m.opt.timestep))
    last, target = np.zeros(len(names)), np.zeros(len(names))
    wi = names.index("waist_pitch")
    hist = []
    for t in range(T + EXTRA):
        r = q_ref[min(t, T - 1)]
        ph = t / T
        obs = np.concatenate([-d.xmat[pel].reshape(3, 3)[2, :], d.qvel[3:6] * 0.25, d.qpos[qa], d.qvel[va] * 0.1, last, r,
                              [ph, math.sin(2 * math.pi * ph), math.cos(2 * math.pi * ph)]])
        a = pol(obs)
        want = np.clip(r + ACT_SCALE * a, lo, hi)
        target = target + np.clip(want - target, -SLEW, SLEW)
        pose = {n: float(v) for n, v in zip(names, target)}
        pose["waist_pitch"] = 2.0 * float(target[wi])
        if TURN_HEAD and t * ref["dt"] < TURN_UNTIL:
            pose.update(TURN_HEAD)
        for _ in range(n_sub):
            sim.set_targets(pose)
            sim.step()
        last = a
        if t % 25 == 0:
            hist.append((round(t * ref["dt"], 2), round(float(d.qpos[2]) * 100, 1), round(sim.tilt_deg())))
            if verbose:
                print(hist[-1], flush=True)
    if trim is not None:
        trim.enabled = True
    z, tilt = float(d.qpos[2]), sim.tilt_deg()
    return dict(ok=bool(z > 0.20 and tilt < 20), z_cm=z * 100, tilt=tilt, hist=hist, side=side)


if __name__ == "__main__":
    import getup_hands as G
    r = run(G.get_sim(), verbose=True, side=next((a.split("=")[1] for a in sys.argv if a.startswith("--side=")), None))
    print(f"结束（头转向{'左' if r['side'] == 'L' else '右'}）：骨盆高 {r['z_cm']:.1f} cm，倾斜 {r['tilt']:.0f}°，{'站起来了' if r['ok'] else '没站起来'}")
