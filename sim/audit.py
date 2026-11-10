"""
55 cm 人偶 · 整体核算：后来加的东西（眼睛、4 关节脖子、舌头……）有没有算进重量、空间、电和热里？

用法：
    python audit.py        # 打印核算表，跑 4 种重量分布的仿真，写 audit_result.json

两部分：
  A. 账本（估算）：头的重量、胸腔的空间、功耗和续航、头能散掉多少热
  B. 仿真：把头和背包的重量换成 4 种情况，重跑站立深蹲、单腿站、走路、前后被推、趴着抬头
     ① 现在的模型      头 0.30 kg，背包 0.20 kg
     ② 头变重          头 0.38 kg（再加实体舌头，或头发做得厚）
     ③ 大脑搬到背包    头 0.25 kg，背包 0.29 kg（计算单元 60 g + 散热片 30 g）
     ④ 搬到背包 + 舌头  头 0.32 kg，背包 0.29 kg
所有数字的来源：零件重量和功耗是估算；仿真部分是仿真结果（沿用整体模型的扭矩上限）。
"""
import json
import math
import os
import tempfile

import mujoco
import numpy as np

import figure_stress
import figure_test
import full_build as fb
import neck
import walk
from balance import Balance
from full_demo import FullSim, scene_single_leg, scene_squat, scene_walk

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------- A. 账本 ----------------
HEAD_PARTS = [  # (名字, 克, 说明)
    ("头壳和脸（直径 9.2 cm，壁厚 1.5 mm）", 50, "按面积算"),
    ("头发（薄壳或假发）", 60, "二次元头发体积大，40–80 g"),
    ("眼睛单元（2 屏、2 摄像头、3 个微电机、云台架、眼睑）", 85, "见 wiki/眼睛.md，85 cm³"),
    ("下巴和嘴的机构", 15, ""),
    ("扬声器、麦克风", 10, ""),
    ("计算单元（含散热）", 60, "体积预算里的数"),
    ("上颈关节的轴承和线轮、导线", 30, ""),
]
TONGUE = ("实体舌头（4 电机 + 气泵），还没决定做不做", 70, "")

CHEST_PARTS = [  # (名字, cm³)
    ("肩俯仰电机 ×2", 36), ("手部电机 12 个 + 6 个锁", 48), ("手部驱动板", 8),
    ("脊髓控制器、IMU、电源板、风扇", 20), ("脖子电机 4 个 + 1 个空位", 20),
]
CHEST_AVAIL = 147

POWER = {  # 瓦（估算）
    "屏幕眼睛 ×2": 0.6, "摄像头 ×2": 0.6, "麦克风、扬声器（平均）": 0.4, "控制器、传感器": 1.5,
}
COMPUTE = {"低功耗（听、看、发呆）": 3.0, "本地跑视觉和小语言模型": 10.0}
MOTORS = {"坐着（下半身不出力）": 1.0, "站着说话、做表情": 4.0, "走路、练习": 10.0}
BATTERY_WH, USABLE = 30.0, 0.85


def head_heat():
    """头不装风扇时能散掉多少热。人会摸它的头：长时间接触的表面不该超过约 43 °C。"""
    area = 4 * math.pi * 0.046 ** 2            # 整个头的表面 m²
    h = 10.0                                   # 自然对流 + 辐射，W/(m²·K)，估算
    dT = 43.0 - 25.0
    return dict(area_cm2=round(area * 1e4), all_W=round(h * area * dT, 1), half_W=round(h * area * 0.5 * dT, 1))


def ledger():
    head = sum(g for _, g, _ in HEAD_PARTS)
    chest = sum(v for _, v in CHEST_PARTS)
    base = sum(POWER.values())
    modes = {}
    for cn, cw in COMPUTE.items():
        for mn, mw in MOTORS.items():
            w = base + cw + mw
            modes[f"{cn} · {mn}"] = dict(W=round(w, 1), hours=round(BATTERY_WH * USABLE / w, 1))
    return dict(head_g=head, head_with_tongue_g=head + TONGUE[1], head_model_g=300,
                chest_need=chest, chest_avail=CHEST_AVAIL, power_base_W=round(base, 1), modes=modes, heat=head_heat())


# ---------------- B. 仿真 ----------------
VARIANTS = [
    ("now", "① 现在的模型", 0.30, 0.20),
    ("heavy", "② 头变重（加舌头或厚头发）", 0.38, 0.20),
    ("split", "③ 大脑搬到背包", 0.25, 0.29),
    ("split_tongue", "④ 搬到背包 + 舌头", 0.32, 0.29),
]


def variant_xml(head, pack):
    s = open(fb.OUT, encoding="utf-8").read()
    a = '<geom name="head" type="sphere" pos="0 0 0.044" size="0.046" mass="0.30"'
    b = '<geom name="backpack" type="box" pos="-0.041 0 0.032" size="0.012 0.030 0.026" mass="0.20"'
    assert s.count(a) == 1 and s.count(b) == 1
    s = s.replace(a, a.replace('mass="0.30"', f'mass="{head}"'))
    # 背包变重时也变厚一点（多装一块计算板和散热片）
    thick = 0.012 * pack / 0.20
    s = s.replace(b, f'<geom name="backpack" type="box" pos="{-0.029 - thick:.4f} 0 0.032" size="{thick:.4f} 0.030 0.026" mass="{pack}"')
    f = tempfile.NamedTemporaryFile("w", suffix=".xml", dir=HERE, delete=False, encoding="utf-8")
    f.write(s)
    f.close()
    return f.name


def push_limit(sim, bal, sign):
    """能扛住的最大推力（0.1 秒，推胸口）。sign=+1 从背后往前推，−1 从前面往后推。"""
    best = 0
    for F in (2, 3, 4, 5, 6, 7, 8):
        if figure_stress.push(sim, sign * F, bal):
            best = F
        else:
            break
    return best


def prone_neck(head_mass):
    """趴着抬头保持时，脖子最热的关节的稳定温度（用新 4 关节脖子，头的重量换掉）。"""
    orig = neck.HEAD_GEOMS
    neck.HEAD_GEOMS = orig.replace("{hm}", str(head_mass))
    try:
        r = neck.prone_test()["new"]
    finally:
        neck.HEAD_GEOMS = orig
    return r["T_hold"], max(r["peak_tau"])


def run_variant(head, pack):
    path = variant_xml(head, pack)
    try:
        figure_test.MODEL = path
        sim = FullSim()
        total = float(sum(sim.m.body_mass))
        sim.reset()
        mujoco.mj_forward(sim.m, sim.d)
        com = sim.d.subtree_com[0].copy()
        feet = np.mean([sim.d.xpos[sim.m.body(n).id] for n in ("l_foot", "r_foot")], axis=0)
        rec = type("R", (), {"label": ""})()
        sq = scene_squat(sim, rec)
        sl = scene_single_leg(sim, rec)
        wk = scene_walk(sim, rec)
        bal = Balance(sim)
        pf, pb = push_limit(sim, bal, +1), push_limit(sim, bal, -1)
    finally:
        os.unlink(path)
    # 新脖子里，电机搬走后头轻了 12 g
    T_neck, tau_neck = prone_neck(round(head - 0.012, 3))
    return dict(total_kg=round(total, 2), com_z_cm=round(float(com[2]) * 100, 1),
                com_x_mm=round(float(com[0] - feet[0]) * 1000, 1),
                squat_fell=bool(sq["fell"]), knee_peak_Nm=round(float(sq["knee_peak_Nm"]), 2),
                single_fell=bool(sl["fell"]), walk_fell=bool(wk["fell"]), walk_cm=round(float(wk["dx_cm"]), 1),
                push_fwd_N=pf, push_back_N=pb, neck_prone_C=T_neck, neck_prone_Nm=round(tau_neck, 3))


def main():
    L = ledger()
    print("A. 账本（估算）")
    print(f"  头：零件合计 {L['head_g']} g（模型里是 {L['head_model_g']} g）；再加实体舌头 {L['head_with_tongue_g']} g")
    print(f"  胸腔：需要 {L['chest_need']} cm³ / 可用 {L['chest_avail']} cm³（{100*L['chest_need']/L['chest_avail']:.0f}%）")
    print(f"  功耗：固定部分 {L['power_base_W']} W；30 Wh 电池按 {int(USABLE*100)}% 可用")
    for k, v in L["modes"].items():
        print(f"    {k}: {v['W']} W → {v['hours']} 小时")
    h = L["heat"]
    print(f"  头不装风扇、表面不超过 43 °C 时能散掉：整个头 {h['all_W']} W；头发盖住一半时 {h['half_W']} W")
    fb.build()
    print("\nB. 仿真")
    res = {}
    for key, title, head, pack in VARIANTS:
        r = run_variant(head, pack)
        res[key] = dict(title=title, head=head, pack=pack, **r)
        print(f"  {title}: 总重 {r['total_kg']} kg，重心高 {r['com_z_cm']} cm、在脚中心前 {r['com_x_mm']} mm；"
              f"深蹲{'摔' if r['squat_fell'] else '过'}（膝 {r['knee_peak_Nm']} N·m），单腿{'摔' if r['single_fell'] else '过'}，"
              f"走路{'摔' if r['walk_fell'] else '过'}（{r['walk_cm']} cm）；抗推 往前 {r['push_fwd_N']} N / 往后 {r['push_back_N']} N；"
              f"趴着抬头脖子 {r['neck_prone_C']} °C（{r['neck_prone_Nm']} N·m）")
    with open(os.path.join(HERE, "audit_result.json"), "w", encoding="utf-8") as f:
        json.dump(dict(ledger=L, sim=res), f, ensure_ascii=False, indent=1)
    print("\n已写 audit_result.json")


if __name__ == "__main__":
    main()
