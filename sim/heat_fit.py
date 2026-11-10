"""
55 cm 人偶 · 发热测试的读数 → 热模型的两个数（热阻、时间常数）

用法：
    python heat_fit.py 读数.csv 功率瓦 [室温]
    python heat_fit.py --demo            # 用一组假数据演示

读数.csv：每行"秒,温度"，例如
    0,24.8
    30,27.9
    60,30.5
加热阶段的读数就够（恒定电流、从室温开始）。脚本按 T(t) = 室温 + P·R·(1 − e^(−t/τ)) 去拟合：
    R（K/W）：每瓦发热最终升多少度      → day.py 里的 R_TH / R_SMALL
    τ（秒） ：升到最终温升的 63% 要多久  → day.py 里的 TAU_TH / TAU_SMALL
"""
import math
import sys

import numpy as np


def fit(t, T, P, T_amb=None):
    t, T = np.asarray(t, float), np.asarray(T, float)
    if T_amb is None:
        T_amb = float(T[0])
    best = None
    for tau in np.geomspace(20, 3000, 400):
        g = 1 - np.exp(-t / tau)
        dT = float(np.dot(g, T - T_amb) / max(np.dot(g, g), 1e-12))   # 最终温升（最小二乘）
        err = float(np.sqrt(np.mean((T_amb + dT * g - T) ** 2)))
        if best is None or err < best[0]:
            best = (err, tau, dT)
    err, tau, dT = best
    return dict(R=dT / P, tau=tau, dT_final=dT, rms=err, T_amb=T_amb)


def main():
    if "--demo" in sys.argv:
        rng = np.random.default_rng(1)
        t = np.arange(0, 601, 30)
        T = 25 + 3.0 * 14.0 * (1 - np.exp(-t / 160)) + rng.normal(0, 0.3, len(t))
        r = fit(t, T, 3.0)
        print("假数据：3 W、R = 14 K/W、τ = 160 s，读数误差 ±0.3 °C")
    else:
        path, P = sys.argv[1], float(sys.argv[2])
        T_amb = float(sys.argv[3]) if len(sys.argv) > 3 else None
        rows = [ln.replace("，", ",").split(",") for ln in open(path, encoding="utf-8") if ln.strip() and ln[0].isdigit()]
        t, T = [float(r[0]) for r in rows], [float(r[1]) for r in rows]
        r = fit(t, T, P, T_amb)
    print(f"热阻 R = {r['R']:.1f} K/W；时间常数 τ = {r['tau']:.0f} 秒；最终会升 {r['dT_final']:.1f} °C（到 {r['T_amb'] + r['dT_final']:.0f} °C）；"
          f"拟合误差 {r['rms']:.2f} °C")
    if r["rms"] > 1.0:
        print("拟合误差偏大：检查电流是否恒定、温度探头是否贴牢、测的时间是否够长（至少 2 个 τ）")


if __name__ == "__main__":
    main()
