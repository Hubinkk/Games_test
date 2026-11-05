"""
55 cm 人偶 · 压力测试（比 figure_test.py 更吃力的动作）

用法：
    python figure_stress.py

测试：
    1. 从深蹲快速站起：分别用 1.0 / 0.6 / 0.4 秒站起来，看膝关节峰值扭矩
    2. 推一下：有无平衡控制器两种情况，推力 2 / 4 / 6 / 8 N
    3. 单腿站立：先把重心移到左脚，再抬起右腿（小幅 / 大幅），用平衡控制器
"""
import math

from balance import Balance
from figure_test import Sim, run_phase, stand_pose

KNEE_LIMIT = 2.8


def smooth(a):
    a = max(0.0, min(1.0, a))
    return 0.5 - 0.5 * math.cos(math.pi * a)


def fast_rise(sim, T):
    sim.reset()
    run_phase(sim, 1.0, lambda t: stand_pose())
    run_phase(sim, 2.0, lambda t: stand_pose(0.3 + 1.7 * smooth(t / 2)))
    pk, _, fell = run_phase(sim, 2.5, lambda t: stand_pose(2.0 - 1.7 * smooth(t / T)))
    return max(pk["l_knee"], pk["r_knee"]), fell


def push(sim, force, bal=None):
    sim.reset()
    if bal:
        bal.prev = None
    fn = (lambda t: bal.adjust(stand_pose())) if bal else (lambda t: stand_pose())
    _, _, fell = run_phase(sim, 3.0, fn, push=(1.0, 0.1, force))
    return fell is None


def single_leg(sim, bal, lift, shiftT=2.5, liftT=3.0):
    sim.reset()
    bal.prev = None
    state = {}

    def support(stance):
        return state["sup"]

    def fn(t):
        p = stand_pose()
        w = smooth(t / shiftT)
        both = Balance.support(bal, ("l", "r"))
        left = Balance.support(bal, ("l",))
        state["sup"] = (1 - w) * both + w * left
        b = smooth((t - shiftT - 0.5) / liftT)
        p["r_hip_pitch"] += -0.5 * lift * b
        p["r_knee"] += 0.9 * lift * b
        p["r_ankle_pitch"] += -0.4 * lift * b
        return bal.adjust(p, ("l",) if w > 0.5 else ("l", "r"))

    bal.support = support
    try:
        pk, _, fell = run_phase(sim, shiftT + liftT + 3.0, fn)
    finally:
        del bal.support
    return fell, pk


def main():
    sim = Sim()
    bal = Balance(sim)

    print("【1】从深蹲（膝 2.0 rad）快速站起")
    for T in (1.0, 0.6, 0.4):
        peak, fell = fast_rise(sim, T)
        print(f"  {T:.1f} 秒站起：膝峰值 {peak:.2f} N·m（上限 {KNEE_LIMIT}，占 {peak/KNEE_LIMIT*100:.0f}%）"
              f"{'，摔倒' if fell is not None else ''}")

    print("\n【2】站立时在胸口水平推 0.1 秒")
    print(f"  {'推力':<6}{'只有关节 PD':<14}{'加平衡控制器'}")
    for f in (2.0, 4.0, 6.0, 8.0):
        a = "站住" if push(sim, f) else "摔倒"
        b = "站住" if push(sim, f, bal) else "摔倒"
        print(f"  {f:.0f} N{'':<3}{a:<14}{b}")

    print("\n【3】单腿站立（重心移到左脚，抬起右腿）")
    for lift, label in ((0.4, "小幅抬腿"), (0.7, "大幅抬腿")):
        fell, pk = single_leg(sim, bal, lift)
        print(f"  {label}：{'站住' if fell is None else '摔倒（第 %.1f 秒）' % fell}；"
              f"支撑腿 髋侧摆 {pk['l_hip_roll']:.2f} / 踝侧摆 {pk['l_ankle_roll']:.2f} / 膝 {pk['l_knee']:.2f} N·m")

    print("\n说明：平衡控制器是手写的重心反馈（balance.py），不是学出来的策略。")
    print("      脚宽已按对照实验改为 4.4 cm；2.8 cm 的小脚在大幅抬腿时会摔倒（见 README）。")


if __name__ == "__main__":
    main()
