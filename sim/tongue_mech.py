"""
触手舌头的力学估算，以及"听自己的 s 来修正"的闭环仿真（人偶尺寸）。

用法：
    python tongue_mech.py      # 输出 tongue_mech.json

一、力学（静态估算，人偶尺寸）
  - 舌体中轴：超弹性镍钛丝 Ø0.5 mm（弯到 2.4 rad 时应变约 3%，镍钛可承受 6–8%；弹簧钢只能 0.5%，会断）
  - 软体：Ecoflex 00-30 级硅胶（弹性模量约 70 kPa）
  - 线：超高分子量聚乙烯（Dyneema）Ø0.2 mm，从头里的电机到舌根约 5 cm
  - 弯矩 M = EI·κ，线拉力 T = M / 线距，电机扭矩 = T × 绕线半径
二、s 的闭环
  s 的缝在人偶尺寸约 0.47 mm，要保持在 ±0.14 mm 以内（成人 1 ± 0.3 mm）。
  舌体弯曲线的位移每变 1 µm，缝变约 10 µm（tongue.py 的灵敏度），所以线要稳定在约 ±14 µm。
  误差来源（都折算成线的位移）：
    - 线的伸长：拉力随前一个元音的姿势变化 ±4 N → 约 ±65 µm（可以按拉力预测并补偿一部分）
    - 线在套管里的摩擦：换向时有死区 ±15 µm
    - 硅胶蠕变：每次发音后慢慢回弹，时间常数约 2 s，幅度约 10 µm
    - 温度与老化：一段对话里缓慢漂移约 60 µm
  听觉：从 s 的噪声响度和频谱估计缝宽，每次估计误差约 ±0.05 mm；说话过程中约 30 ms 能修正一次。
  触觉：舌尖的电容 / 压力传感器直接测到齿龈的距离，误差约 ±0.02 mm，每 10 ms 一次（人发 s 时也靠舌尖的触觉）。
  比较四种控制：不修正 / 按拉力补偿线的伸长 / 每说完一次听一下再调（像人逐次纠音）/ 说的同时边听边调。
"""
import json
import math
import os

import numpy as np

import tongue as T

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = T.FIG

# ---------------- 一、力学 ----------------
E_NITI, D_NITI = 60e9, 0.5e-3
E_SIL = 70e3
SEG1 = T.L1 * FIG / 100            # m
SEG2 = T.L2 * FIG / 100
H_AVG, W = 0.8e-2, 1.0e-2          # 软体平均厚度、宽度（m）
R_CABLE = T.CABLE_R * FIG / 100    # 线距（m）
R_SPOOL = 3e-3
CABLE_D, CABLE_E, CABLE_L = 0.2e-3, 100e9, 0.05


def mechanics():
    i_niti = math.pi * D_NITI ** 4 / 64
    i_sil = math.pi / 4 * (W / 2) * (H_AVG / 2) ** 3
    ei = E_NITI * i_niti + E_SIL * i_sil
    out = {"EI_Nm2": ei, "EI_backbone_share": E_NITI * i_niti / ei}
    for name, seg, phi in (("舌体", SEG1, 2.4), ("舌尖", SEG2, 1.8)):
        k = phi / seg
        m = ei * k
        t = m / R_CABLE
        out[name] = dict(length_mm=seg * 1e3, curvature=k, strain_pct=k * D_NITI / 2 * 100,
                         moment_Nmm=m * 1e3, tension_N=t, torque_Ncm=t * R_SPOOL * 100)
    k_cable = CABLE_E * math.pi * CABLE_D ** 2 / 4 / CABLE_L
    out["cable_k_N_per_mm"] = k_cable / 1e3
    out["stretch_um_per_N"] = 1e6 / k_cable
    # t/n 的密封：皮肤压入 0.1 mm（成人 0.2 mm × 8/17），皮肤层约 2 mm → 应变 5%
    out["seal_contact_kPa"] = E_SIL * 0.1 / 2.0 / 1e3
    out["seal_needed_kPa"] = 0.8           # 说话时口腔内的气压约 0.5–1 kPa
    # 气腔：体积约 0.5 cm³，70 ms 内改变 30%
    out["chamber_flow_ml_per_s"] = 0.5 * 0.3 / 0.07
    # 速度和功率：元音之间线最多走 3.1 mm，70 ms
    v = 3.1e-3 / 0.07
    out["cable_speed_mm_s"] = v * 1e3
    out["spool_rpm"] = v / R_SPOOL / (2 * math.pi) * 60
    out["power_W"] = out["舌体"]["tension_N"] * v
    return out


# ---------------- 二、s 的闭环 ----------------
GAIN = 10.0            # 缝宽 / 线位移
GAP0, TOL = 0.47, 0.14  # mm


def simulate(strategy, n=300, seed=7):
    rng = np.random.default_rng(seed)
    stretch_k = 1.0 / (CABLE_E * math.pi * CABLE_D ** 2 / 4 / CABLE_L) * 1e3   # mm/N
    drift, creep, last_dir = 0.0, 0.0, 1
    u = 0.0              # 指令（线位移，mm），0 = 出厂标定的位置
    learned = 0.0        # 逐次学习的修正量
    gaps, final_gaps, ok = [], [], []
    for k in range(n):
        drift += rng.normal(0, 0.06 / math.sqrt(n)) * 1.0          # 一段对话里约 60 µm 的随机漂移
        dt_tension = rng.uniform(-4, 4)                              # 前一个元音造成的拉力差（N）
        stretch = dt_tension * stretch_k                             # mm
        direction = rng.choice([-1, 1])                              # 从哪个方向接近 s 的姿势
        backlash = 0.015 * direction
        creep = creep * math.exp(-0.6 / 2.0) + 0.010 * rng.uniform(-1, 1)
        err_line = stretch + backlash + creep + drift                # 线实际少走 / 多走的量（mm）
        cmd = learned
        if strategy in ("model", "both_loops", "trial", "touch"):
            if strategy != "trial":
                cmd -= 0.8 * stretch                                 # 按拉力预测伸长，补偿 80%
        # 说 s 的约 100 ms 内：边听边调每 30 ms 修正一次（3 次）；舌尖触觉每 10 ms 一次（9 次）
        seg_gaps = []
        inner = 0.0
        if strategy == "touch":
            steps, sigma, gain = 10, 0.02, 0.7
        elif strategy == "both_loops":
            steps, sigma, gain = 4, 0.05, 0.6
        else:
            steps, sigma, gain = 4, 0.0, 0.0
        for j in range(steps):
            g = GAP0 + GAIN * (cmd + inner - err_line)
            seg_gaps.append(g)
            if gain and j < steps - 1:
                est = g + rng.normal(0, sigma)
                inner += gain * (GAP0 - est) / GAIN
        g_mean = float(np.mean(seg_gaps))
        g_last = seg_gaps[-1]
        # 说完以后：逐次学习（听这一次的 s，修正下一次）
        if strategy in ("trial", "both_loops", "touch"):
            est = g_mean + rng.normal(0, 0.05)
            learned += 0.4 * (GAP0 - est) / GAIN
        gaps.append(g_mean)
        final_gaps.append(g_last)
        ok.append(abs(np.mean(seg_gaps[len(seg_gaps) // 3:]) - GAP0) <= TOL)   # 前三分之一是起音，看后面的稳定段
    return dict(gaps=[round(x, 3) for x in gaps], success=float(np.mean(ok)),
                rms_err_mm=float(np.sqrt(np.mean((np.array(gaps) - GAP0) ** 2))))


if __name__ == "__main__":
    mech = mechanics()
    print("力学（人偶尺寸）")
    for k in ("舌体", "舌尖"):
        m = mech[k]
        print(f"  {k}：长 {m['length_mm']:.0f} mm，最大弯曲时 镍钛应变 {m['strain_pct']:.1f}%，"
              f"线拉力 {m['tension_N']:.1f} N，电机扭矩 {m['torque_Ncm']:.1f} N·cm")
    print(f"  中轴（镍钛）占弯曲刚度的 {mech['EI_backbone_share']*100:.0f}%；线每受 1 N 伸长 {mech['stretch_um_per_N']:.0f} µm")
    print(f"  t/n 密封：接触压力约 {mech['seal_contact_kPa']:.1f} kPa，口腔气压约 {mech['seal_needed_kPa']} kPa")
    print(f"  气腔流量 {mech['chamber_flow_ml_per_s']:.1f} ml/s；线速度 {mech['cable_speed_mm_s']:.0f} mm/s，"
          f"绕线轮 {mech['spool_rpm']:.0f} rpm，功率 {mech['power_W']:.2f} W")
    names = {"open": "不修正（出厂标定）", "model": "按拉力补偿线的伸长", "trial": "每次说完听一下再调",
             "both_loops": "补偿 + 说的同时边听边调 + 逐次学习",
             "touch": "再加舌尖触觉（直接测缝宽，每 10 ms）"}
    res = {}
    print("\ns 的闭环（300 次 s，缝宽 0.47 ± 0.14 mm 算成功）")
    for s, name in names.items():
        r = simulate(s)
        res[s] = dict(r, name=name)
        print(f"  {name}：成功 {r['success']*100:.0f}%，缝宽均方根误差 {r['rms_err_mm']*1000:.0f} µm")
    json.dump(dict(mech=mech, loop=res, gap0=GAP0, tol=TOL), open(os.path.join(HERE, "tongue_mech.json"), "w"),
              ensure_ascii=False, indent=1)
