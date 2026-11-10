"""
55 cm 人偶 · 自动配平（L1）：它用自己的身体数据算出重心，自己把重心摆回脚的上方

用法：
    python trim.py         # 用几种"头和背包重量"测试：不配平 vs 自动配平，写 trim_result.json

做法（不用强化学习，不用传感器，只用它自己的身体数据 = 每一段的重量和长度）：
  1. 它有两份身体数据：设计动作时的身体（参考），和现在的身体（加了电池、换了零件之后）
  2. 对"马上要摆的姿势"，在脑子里分别算两个身体的重心：现在的重心比参考偏了多少
  3. 算出腰往前弯 1 弧度，重心会前移多少；腰加一个修正量，把偏掉的那部分补回来（每 20 ms 重算一次）
  （第一版让每个姿势都回到同一个重心位置，结果腰修正到 ±13°，连原来能做的深蹲都摔了：
    深蹲、迈步本来就有各自的重心位置，要补的只是身体变化带来的偏移。）
以后身体数据不准时（穿了衣服、拿着东西），再用脚底压力把这份数据校准，方法不变。

测试的重量分布：
  now      头 0.30 / 背包 0.20   现在的模型
  split    头 0.25 / 背包 0.29   上次试的"大脑搬进背包"（不配平会摔）
  plan     头 0.32 / 背包 0.31   新方案：大脑留在头里 + 冷板；背后模块 = 60 Wh 电池 + 散热器、泵、风扇
  plan90   头 0.32 / 背包 0.43   同上，电池 90 Wh
"""
import json
import os
import tempfile

import mujoco
import numpy as np

import figure_stress
import figure_test
import full_build as fb
from autotrim import Trim
from balance import Balance
from full_demo import FullSim, scene_single_leg, scene_squat, scene_walk

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET_X = None      # 设计位置：重心在脚掌中心前后多少（从现在的模型量出来）
EVERY = 0.02         # 每 20 ms 重算一次
MAX_TRIM = 0.25      # 腰最多修正这么多弧度


def variant_xml(head, pack, pack_half_thick):
    s = open(fb.REF, encoding="utf-8").read()
    a = '<geom name="head" type="sphere" pos="0 0 0.044" size="0.046" mass="0.30"'
    b = '<geom name="backpack" type="box" pos="-0.041 0 0.032" size="0.012 0.030 0.026" mass="0.20"'
    assert s.count(a) == 1 and s.count(b) == 1
    s = s.replace(a, a.replace('mass="0.30"', f'mass="{head}"'))
    s = s.replace(b, f'<geom name="backpack" type="box" pos="{-0.029 - pack_half_thick:.4f} 0 0.032" '
                     f'size="{pack_half_thick:.4f} 0.030 0.026" mass="{pack}"')
    f = tempfile.NamedTemporaryFile("w", suffix=".xml", dir=HERE, delete=False, encoding="utf-8")
    f.write(s)
    f.close()
    return f.name


def push_limit(sim, bal, sign):
    best = 0
    for F in (2, 3, 4, 5, 6, 7, 8):
        if figure_stress.push(sim, sign * F, bal):
            best = F
        else:
            break
    return best


def run(head, pack, thick, use_trim):
    path = variant_xml(head, pack, thick)
    try:
        figure_test.MODEL = path
        sim = FullSim()
        total = float(sum(sim.m.body_mass))
        tr = Trim(sim, fb.REF) if use_trim else None
        rec = type("R", (), {"label": ""})()
        sq = scene_squat(sim, rec)
        sl = scene_single_leg(sim, rec)
        wk = scene_walk(sim, rec)
        bal = Balance(sim)
        pf, pb = push_limit(sim, bal, +1), push_limit(sim, bal, -1)
        lean = (float(np.degrees(np.min(tr.history))), float(np.degrees(np.max(tr.history)))) if tr else (0.0, 0.0)
    finally:
        os.unlink(path)
    return dict(total_kg=round(total, 2), squat_fell=bool(sq["fell"]), knee_peak_Nm=round(float(sq["knee_peak_Nm"]), 2),
                single_fell=bool(sl["fell"]), walk_fell=bool(wk["fell"]), walk_cm=round(float(wk["dx_cm"]), 1),
                push_fwd_N=pf, push_back_N=pb, lean_deg=[round(lean[0], 1), round(lean[1], 1)])


VARIANTS = [
    ("now", "现在的模型", 0.30, 0.20, 0.012),
    ("split", "大脑搬进背包（上次不配平会摔）", 0.25, 0.29, 0.0174),
    ("plan", "新方案：头里大脑 + 60 Wh + 背后散热", 0.32, 0.31, 0.022),
    ("plan90", "新方案 + 90 Wh", 0.32, 0.43, 0.028),
]


def main():
    global TARGET_X
    fb.build()
    out = {"variants": {}}
    for key, title, head, pack, thick in VARIANTS:
        res = {}
        for use in (False, True):
            r = run(head, pack, thick, use)
            res["trim" if use else "raw"] = r
            print(f"{title}（{'自动配平' if use else '不配平'}）: {r['total_kg']} kg；"
                  f"深蹲{'摔' if r['squat_fell'] else '过'}（膝 {r['knee_peak_Nm']}），单腿{'摔' if r['single_fell'] else '过'}，"
                  f"走路{'摔' if r['walk_fell'] else '过'}（{r['walk_cm']} cm）；抗推 前 {r['push_fwd_N']} / 后 {r['push_back_N']} N；"
                  f"腰的修正 {r['lean_deg'][0]}°～{r['lean_deg'][1]}°", flush=True)
        out["variants"][key] = dict(title=title, head=head, pack=pack, **res)
    with open(os.path.join(HERE, "trim_result.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n已写 trim_result.json")


if __name__ == "__main__":
    main()
