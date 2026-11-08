"""tongue.py 的补充：最完整的结构（两段 + 舌根升降 + 气腔）的舌形图、t/n 的升降行程、s 的精度要求。"""
import json
import os

import numpy as np

import tongue as T

T._calibrate()
KEYS = T.CONFIGS["两段+升降+气腔"]
START = {  # tongue.py 中“两段+升降”的结果，作为起点
    "a": dict(lift=0.444, slide=0.446, phi1=-1.465, phi2=1.196), "i": dict(lift=0.557, slide=0.135, phi1=-0.141, phi2=1.312),
    "u": dict(lift=0.795, slide=0.785, phi1=-1.386, phi2=1.442), "e": dict(lift=0.634, slide=0.637, phi1=-1.425, phi2=1.574),
    "o": dict(lift=-0.07, slide=0.019, phi1=-1.478, phi2=1.443),
}
res = {}
for v, tgt in T.V.TARGETS.items():
    p, c = T.search(T.vowel_cost(tgt), KEYS, dict(T.REST, **T.FIXED[v]), 5, starts=[START[v]], iters=8)
    f = T.formants_fast(p)
    res[v] = dict(p=p, f=f, err=[abs(f[0] / tgt[0] - 1), abs(f[1] / tgt[1] - 1)])
    print(f"{v}: {f[0]:.0f}/{f[1]:.0f}  误差 {res[v]['err'][0]*100:.0f}% / {res[v]['err'][1]*100:.0f}%", flush=True)
cons = {}
for name, region, gap in (("t/n", T.ALV, 0.0), ("k", T.VEL, 0.0), ("s", T.ALV, 0.1)):
    for lift_max in (1.2, 1.6):
        T.RANGES["lift"] = (-0.6, lift_max)
        p, c = T.search(T.closure_cost(region, gap), KEYS, dict(T.REST, **T.FIXED["cons"]), 5, iters=8)
        _, _, d = T.area_fn(p)
        m = d[len(T.PHAR_Y):len(T.PHAR_Y) + len(T.RAY_DEG)][region].min()
        print(f"{name}（舌根升降上限 {lift_max} cm）: {'可以' if c < 0.03 else '不行'}  最小距离 {m*10:+.1f} mm", flush=True)
        if c < 0.03 or lift_max == 1.6:
            cons[name] = dict(p=p, ok=bool(c < 0.03), lift_max=lift_max, gap=float(m))
            break
T.RANGES["lift"] = (-0.6, 1.2)
# s 的窄缝对每个驱动的灵敏度 → 人偶尺寸下需要的分辨率
p = cons["s"]["p"]
def gap(pp):
    _, _, d = T.area_fn(pp)
    return d[len(T.PHAR_Y):len(T.PHAR_Y) + len(T.RAY_DEG)][T.ALV].min()
prec = {}
for k in ("slide", "lift", "phi1", "phi2", "bulge"):
    e = 1e-3
    dg = (gap(dict(p, **{k: p[k] + e})) - gap(dict(p, **{k: p[k] - e}))) / (2 * e)
    tol = 0.03 / max(abs(dg), 1e-9)                  # 缝宽 1 ± 0.3 mm（成人）
    if k.startswith("phi"):
        prec[k] = T.CABLE_R * T.FIG * tol * 1e4      # µm 线位移
    elif k == "bulge":
        prec[k] = tol * 100                          # 厚度的百分比
    else:
        prec[k] = tol * T.FIG * 1e4                  # µm 滑台位移
    print(f"s 精度 {k}: {prec[k]:.0f} {'%' if k == 'bulge' else 'µm'}")
shapes = [(v, res[v]["p"], max(res[v]["err"]) < 0.2) for v in "aiueo"] + [(n, cons[n]["p"], cons[n]["ok"]) for n in ("t/n", "k", "s")]
T.svg_shapes(shapes, os.path.join(T.HERE, "tongue_shapes.svg"))
json.dump(dict(vowels={v: dict(f=[None if np.isnan(x) else float(x) for x in r["f"]], err=r["err"]) for v, r in res.items()},
               cons={n: dict(ok=c["ok"], lift_max=c["lift_max"], gap_mm=c["gap"] * 10) for n, c in cons.items()}, prec=prec),
          open(os.path.join(T.HERE, "tongue_result.json"), "w"), ensure_ascii=False, indent=1)
