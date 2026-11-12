"""
给"头偏向一边"那一轮做起点：从一份已有的策略（默认 run3_final.npz）改出 turn_init.npz，再用 train_getup.py --init= 接着练。

为什么要改：旧策略练的时候脖子几乎不动、参考动作里 neck_yaw 一直是 0，
观测归一化里这几项的标准差只有 1e-6（参考的 neck_yaw、neck_roll）或 0.1–0.2（实际的脖子角）。
换成转头的参考（neck_yaw = ±1.3）以后，这几个数除以标准差变成上百万，网络直接乱掉（2026-11-12 实测：0/48，numpy 溢出）。

怎么改（只动和脖子有关的 12 个输入，观测的 113 个数和顺序不变）：
  - 实际的脖子角（4 个）、脖子角速度（4 个）、参考动作里的脖子角（4 个）：网络第一层里这 12 个输入的权重清零，
    于是一开始策略完全不看它们，行为和旧策略一样（脖子由参考动作带着转，相当于原来的"覆盖"）；
  - 这 12 项的归一化改成合理的尺度（均值 0、标准差至少 0.5 / 0.25），以后练的时候网络可以慢慢学着用它们。

用法： python make_turn_init.py [run3_final.npz] [turn_init.npz]
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NU = 26
NECK = [2, 3, 4, 5]                      # neck_yaw、neck_pitch、neck_low、neck_roll 在 26 个电机里的位置
Q0, V0, A0, R0 = 6, 6 + NU, 6 + 2 * NU, 6 + 3 * NU      # 观测里：关节角、关节角速度、上一步动作、参考角 各自从第几个数开始


def main(src="run3_final.npz", dst="turn_init.npz"):
    z = np.load(os.path.join(HERE, src))
    out = {k: z[k].copy() for k in z.files}
    count = float(out["norm/count/lo"]) + float(out["norm/count/hi"]) * 2.0 ** 32
    dims = [Q0 + i for i in NECK] + [V0 + i for i in NECK] + [R0 + i for i in NECK]
    floor = {**{Q0 + i: 0.5 for i in NECK}, **{V0 + i: 0.25 for i in NECK}, **{R0 + i: 0.5 for i in NECK}}
    for j in dims:
        out["norm/mean"][j] = 0.0
        out["norm/std"][j] = max(float(out["norm/std"][j]), floor[j])
        out["norm/summed_variance"][j] = out["norm/std"][j] ** 2 * count
    out["policy/params/hidden_0/kernel"][dims, :] = 0.0
    np.savez(os.path.join(HERE, dst), **out)
    print(f"写好了 {dst}：{len(dims)} 个和脖子有关的输入，权重清零、归一化改成合理的尺度")


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    main(*a)
