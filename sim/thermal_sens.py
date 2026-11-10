"""
55 cm 人偶 · 发热参数有多重要：同一天，散热好一倍 / 和估算一样 / 差 1.5 倍 / 差 2 倍

用法：
    python thermal_sens.py --getup=kneel3      # 写 thermal_sens_result.json；不写日记、不动记忆

数字花然的"热"来自估算的参数（day.py）：小电机 15 K/W、大电机 4 K/W、持续扭矩 = 峰值的 48%。
这里把"同样的发热，散热差 f 倍"（等于温升乘 f）套进同一天里，看它的一天会不会变成另一个样子。
f = 1.5～2 大致对应：穿着衣服、电机挤在胸腔里，或者持续扭矩其实只有峰值的 35%。
"""
import collections
import json
import os
import sys

import agility
import day
import figure_test
import full_build as fb
import full_demo as FD
import getup
from balance import Balance

HERE = os.path.dirname(os.path.abspath(__file__))


def one_day(f):
    fb.build()
    figure_test.MODEL = fb.OUT
    sim = FD.FullSim()
    getup._SIM = sim
    agility.add_speed_limit(sim)
    bal = Balance(sim)
    body = day.Body(sim)
    for n in body.joints:           # 发热不变（k 不动），散热差 f 倍
        body.R[n] *= f
    day.BODY = body
    tag = day.GETUP_TAG
    gx = json.load(open(os.path.join(HERE, f"getup_best_{tag}.json" if tag else "getup_best.json"), encoding="utf-8"))["x"]
    getup.METHOD = tag
    res = {}
    for key, title, clock, mins, simT, fn, desc in day.scenes(sim, bal, gx):
        body.scene = dict(key=key, clock=clock)
        body.scale = mins * 60.0 / simT
        body.scene_start[key] = dict(body.T)
        body.t_scene = 0.0
        body.effort_t = {n: 0.0 for n in body.joints}
        if key == "walk":
            sim.reset()
        out = fn()
        if key == "walk":
            res["walk_cm"] = round(float(sim.d.qpos[0]) * 100, 1)
        elif key in ("push1", "push2"):
            res[key] = bool(out)
    kinds = collections.Counter(e["kind"] for e in body.events)
    top = sorted(body.peak_T.items(), key=lambda kv: -kv[1])[:4]
    return dict(f=f, events=dict(kinds), hottest=[(day.NAMES.get(n, n), round(t, 1)) for n, t in top],
                battery_left=round(body.battery(), 2), **res)


def main():
    out = []
    for f in (0.5, 1.0, 1.5, 2.0):
        r = one_day(f)
        out.append(r)
        print(json.dumps(r, ensure_ascii=False), flush=True)
    json.dump(out, open(os.path.join(HERE, "thermal_sens_result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
