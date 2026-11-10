"""
55 cm 人偶 · 加一层类橡胶皮肤以后还站不站得住（2026-11-10，育苗人问：有没有为将来的人工皮肤留出支持）

用法：
    python skin_study.py        # 写 skin_study_result.json

只回答"重量"这一项：按每一段身体的表面积，加上 1 mm / 2 mm 厚的硅胶类皮肤（密度按 1.1 g/cm³），
重跑深蹲、单腿站、走路，以及"蹲着手撑地 → 站起"。
皮肤的另外三个影响这里算不了：关节处被拉伸时的阻力、保温、厚度占掉的活动间隙。
"""
import json
import math
import os

import mujoco
import numpy as np

import figure_test
import full_build as fb
import getup_hands as G
from balance import Balance
from full_demo import FullSim, scene_single_leg, scene_squat, scene_walk

HERE = os.path.dirname(os.path.abspath(__file__))
RHO = 1100.0


def areas(m):
    """每个身体段的表面积（m²），按碰撞几何体估算；手指太细，按手掌一块算。"""
    a = np.zeros(m.nbody)
    for g in range(m.ngeom):
        if not m.geom_contype[g] or m.geom_bodyid[g] == 0:
            continue
        t, s = m.geom_type[g], m.geom_size[g]
        if t == 2:
            A = 4 * math.pi * s[0] ** 2
        elif t == 3:
            A = 2 * math.pi * s[0] * 2 * s[1] + 4 * math.pi * s[0] ** 2
        elif t == 6:
            A = 8 * (s[0] * s[1] + s[1] * s[2] + s[0] * s[2])
        else:
            continue
        a[m.geom_bodyid[g]] += A
    return a


def run(thick_mm):
    figure_test.MODEL = fb.OUT
    G._SIM = None
    sim = G.get_sim()
    m = sim.m
    A = areas(m)
    dm = A * thick_mm / 1000.0 * RHO
    m.body_mass[:] += dm
    mujoco.mj_setConst(m, sim.d)
    rec = type("R", (), {"label": ""})()
    sq = scene_squat(sim, rec)
    sl = scene_single_leg(sim, rec)
    wk = scene_walk(sim, rec)
    up = G.squat_to_stand(verbose=False)
    return dict(skin_g=round(float(dm.sum()) * 1000), area_m2=round(float(A.sum()), 3), total_kg=round(float(m.body_mass.sum()), 2),
                squat_fell=bool(sq["fell"]), knee_peak_Nm=round(float(sq["knee_peak_Nm"]), 2), single_fell=bool(sl["fell"]),
                walk_fell=bool(wk["fell"]), walk_cm=round(float(wk["dx_cm"]), 1), stand_up=bool(up["ok"]),
                stand_up_peak={k: round(v, 2) for k, v in up["peak"].items()})


def main():
    out = {}
    for t in (0.0, 1.0, 2.0):
        r = run(t)
        out[f"{t:g} mm"] = r
        print(f"皮肤 {t:g} mm：{r['skin_g']} g（表面积 {r['area_m2']} m²），总重 {r['total_kg']} kg；深蹲{'摔' if r['squat_fell'] else '过'}（膝 {r['knee_peak_Nm']} N·m），"
              f"单腿{'摔' if r['single_fell'] else '过'}，走路{'摔' if r['walk_fell'] else '过'}（{r['walk_cm']} cm），"
              f"从蹲姿站起{'成功' if r['stand_up'] else '失败'}；站起时最吃力的关节（占上限）：{r['stand_up_peak']}", flush=True)
    json.dump(out, open(os.path.join(HERE, "skin_study_result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
