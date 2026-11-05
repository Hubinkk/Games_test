"""
55 cm 人偶 · 静步行测试（手写步态 + 重心反馈平衡）

用法：
    python walk.py            # 走若干步并打印结果
    python walk.py --view     # 同时打开 3D 窗口

步态：每一步分三段
    1. 重心移到支撑脚上方（平衡控制器负责左右方向）
    2. 抬起另一只脚、向前迈出、放下；同时支撑腿向后摆，身体前移
    3. 双脚着地，准备换脚
这是最保守的"静步行"：任何时刻重心都在脚的上方，所以走得很慢。
真正自然的"动步行"需要在 Isaac Lab 里用强化学习训练。
"""
import argparse
import math

import numpy as np

from balance import Balance
from figure_test import Sim, run_phase, stand_pose

L = 0.115            # 大腿、小腿长度
H0 = 2 * L * math.cos(0.15)  # 站姿时髋到踝的高度


def smooth(a):
    a = max(0.0, min(1.0, a))
    return 0.5 - 0.5 * math.cos(math.pi * a)


def leg_ik(x, z):
    """脚相对髋的位置（x 向前，z 向下为负）→ 髋俯仰、膝、踝俯仰（脚掌保持水平）。"""
    d = min(math.hypot(x, z), 2 * L - 1e-4)
    knee = math.pi - math.acos((2 * L * L - d * d) / (2 * L * L))
    alpha = math.atan2(x, -z)
    hip = -(alpha + knee / 2)
    return hip, knee, -(hip + knee)


def walk(sim, bal, steps=6, stride=0.03, lift=0.02, T=2.4):
    sim.reset()
    bal.prev = None
    state = {"sup": None}
    shift, swing = 0.35 * T, 0.45 * T  # 剩下 0.2T 双脚着地

    def foot_targets(t):
        k = int(t // T)
        u = t - k * T
        stance = "l" if k % 2 == 0 else "r"
        swing_leg = "r" if stance == "l" else "l"
        x = {"l": 0.0, "r": 0.0}
        z = {"l": -H0, "r": -H0}
        if k >= steps:
            return x, z, ("l", "r"), 0.0, stance
        # 第一步从并脚开始，之后每步从 ±stride/2 开始
        start = 0.0 if k == 0 else stride / 2
        if u < shift:
            w = smooth(u / (0.8 * shift))
            x[stance], x[swing_leg] = start, -start
            return x, z, (stance,), w, stance
        a = smooth((u - shift) / swing)
        x[stance] = start + (-stride / 2 - start) * a
        x[swing_leg] = -start + (stride / 2 + start) * a
        if u < shift + swing:
            z[swing_leg] = -H0 + lift * math.sin(math.pi * (u - shift) / swing)
            return x, z, (stance,), 1.0, stance
        return x, z, ("l", "r"), 1.0 - smooth((u - shift - swing) / (0.2 * T)), stance

    def support(stance_feet):
        return state["sup"]

    def fn(t):
        x, z, st, w, stance = foot_targets(t)
        both = Balance.support(bal, ("l", "r"))
        one = Balance.support(bal, (stance,))
        state["sup"] = (1 - w) * both + w * one
        p = stand_pose()
        for s in ("l", "r"):
            p[f"{s}_hip_pitch"], p[f"{s}_knee"], p[f"{s}_ankle_pitch"] = leg_ik(x[s], z[s])
        return bal.adjust(p, (stance,) if w > 0.5 else ("l", "r"))

    bal.support = support
    try:
        x0 = float(sim.d.qpos[0])
        pk, _, fell = run_phase(sim, (steps + 1) * T, fn)
    finally:
        del bal.support
    return fell, float(sim.d.qpos[0]) - x0, float(sim.d.qpos[1]), pk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--stride", type=float, default=0.03)
    ap.add_argument("--period", type=float, default=2.4)
    a = ap.parse_args()
    sim = Sim(view=a.view)
    bal = Balance(sim)
    fell, dx, dy, pk = walk(sim, bal, a.steps, a.stride, T=a.period)
    print(f"步数 {a.steps}，步长 {a.stride*100:.1f} cm，每步 {a.period:.1f} 秒")
    print(f"结果：{'摔倒（第 %.1f 秒）' % fell if fell is not None else '走完，没有摔倒'}；"
          f"前进 {dx*100:.1f} cm，横向偏移 {dy*100:.1f} cm")
    for n in ("hip_pitch", "knee", "ankle_pitch", "hip_roll", "ankle_roll"):
        v = max(pk[f"l_{n}"], pk[f"r_{n}"])
        print(f"  {n:<12} 峰值 {v:.2f} N·m（上限 {sim.limit['l_' + n]:.1f}）")


if __name__ == "__main__":
    main()
