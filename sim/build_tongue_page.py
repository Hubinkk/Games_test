"""把触手舌头的仿真结果填进 Tongue.html（页面里 /*DATA_START*/ … /*DATA_END*/ 之间）。

用法：
    python tongue_like.py && python tongue_mech.py && python build_tongue_page.py
"""
import json
import os
import re

import numpy as np

import tongue as T
import voice as V

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(HERE, "..", "Tongue.html")


def marks(p, ref):
    """收紧点（触手）与人的收紧点位置，画在中线上。"""
    a, seg, _ = T.area_fn(p)
    _, mids = T.midsagittal(p)
    pos = np.cumsum(seg) - seg / 2
    i = 2 + int(np.argmin(a[2:-3]))
    c = mids[i - 2]
    pm = pos[2:]
    j = int(np.clip(np.searchsorted(pm, ref), 1, len(pm) - 2))
    w = (ref - pm[j - 1]) / (pm[j] - pm[j - 1])
    h = mids[j - 1] + (mids[j] - mids[j - 1]) * np.clip(w, 0, 1)
    tdir = mids[j + 1] - mids[j - 1]
    n = np.array([-tdir[1], tdir[0]]) / (np.linalg.norm(tdir) + 1e-9)
    return dict(c=[round(float(x), 3) for x in c], h=[round(float(x), 3) for x in h], hn=[round(float(x), 3) for x in n])


def explain(name, v, r, ref):
    like = abs(r["dloc"]) <= 1.5
    heard = r["heard"] == v
    if name == "sound":
        if like:
            return "只求声音时，这个音的形状碰巧也接近人。"
        return f"声音对了，但它在离声门 {r['loc']:.1f} cm 处收窄（人在 {ref} cm），用了人不会用的形状。同一个声音可以由好几种形状发出。"
    s = "声音和形状都接近人。" if heard and like else ("形状接近人，但声音被听成了别的元音。" if like else "没能同时做到。")
    if heard and r["err"] > 0.15:
        s += f" 共振峰偏离目标 {r['err']*100:.0f}%，听得出是 {v}，但不够准。"
    return s


def main():
    T._calibrate()
    like = json.load(open(os.path.join(HERE, "tongue_like.json"), encoding="utf-8"))
    mech = json.load(open(os.path.join(HERE, "tongue_mech.json"), encoding="utf-8"))
    audio = json.load(open(os.path.join(HERE, "tongue_audio.json"), encoding="utf-8"))
    data = dict(
        sets=like["sets"], ref_loc=like["ref_loc"], tol=like["tol"],
        targets={v: list(t) for v, t in V.TARGETS.items()},
        wall=T.WALL.tolist(),
        geo=dict(L1=T.L1, L2=T.L2, thick=0.25 + 2.35, baseY=3.0, tmj=T.TMJ.tolist()),
        marks={k: {v: marks(like["sets"][k][v]["p"], like["ref_loc"][v]) for v in "aiueo"} for k in ("sound", "both")},
        explain={k: {v: explain(k, v, like["sets"][k][v], like["ref_loc"][v]) for v in "aiueo"} for k in ("sound", "both")},
        mech=mech["mech"], loop=mech["loop"], gap0=mech["gap0"], tol_gap=mech["tol"],
        audio=audio,
    )
    html = open(PAGE, encoding="utf-8").read()
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = re.sub(r"/\*DATA_START\*/.*?/\*DATA_END\*/", lambda m: "/*DATA_START*/" + blob + "/*DATA_END*/", html, flags=re.S)
    open(PAGE, "w", encoding="utf-8").write(html)
    print(f"已写入 {os.path.relpath(PAGE)}：{len(html)/1024:.0f} KB")


if __name__ == "__main__":
    main()
