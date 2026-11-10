"""
髋和腰的结构设计：算几个要放进文档的数，并导出侧面图用的形状（flex_fig.json，给 HipWaist.html 画图）

用法： python flex_fig.py
"""
import json
import math
import os

import mujoco
import numpy as np

import flex_study as F

HERE = os.path.dirname(os.path.abspath(__file__))
NEW = dict(two_seg=True, hip_dx=0.015, cut=0.02, back=0.02)
SHOW = ("pelvis", "abdomen", "chest", "neck", "neck_up", "head", "l_thigh", "l_shin", "l_foot", "l_upper_arm", "l_forearm", "L_palm")


def shapes(B):
    """左侧身体各部分在侧面（前后 x、上下 z）的投影，单位 cm；地面在 z = 0。"""
    m, d = B.m, B.d
    sole = d.geom_xpos[B.foot][2] - m.geom_size[B.foot][2]
    fx = d.geom_xpos[B.foot][0]
    out = []
    for g in range(m.ngeom):
        b = m.body(m.geom_bodyid[g]).name
        if b not in SHOW or not m.geom_contype[g]:
            continue
        R = d.geom_xmat[g].reshape(3, 3)
        c = d.geom_xpos[g]
        t, sz = int(m.geom_type[g]), m.geom_size[g]
        P = lambda p: [round(float(p[0] - fx) * 100, 2), round(float(p[2] - sole) * 100, 2)]
        if t == 2:
            out.append(dict(k="c", b=b, c=P(c), r=round(float(sz[0]) * 100, 2)))
        elif t == 3:
            a, e = c - R[:, 2] * sz[1], c + R[:, 2] * sz[1]
            out.append(dict(k="l", b=b, a=P(a), e=P(e), r=round(float(sz[0]) * 100, 2)))
        elif t == 6:
            pts = [P(c + R @ (np.array([sx, 0, sz_]) * sz)) for sx, sz_ in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
            out.append(dict(k="p", b=b, pts=pts, name=m.geom(g).name))
    mujoco.mj_comPos(m, d)
    return dict(shapes=out, com=round(float(d.subtree_com[0][0] - fx) * 100, 2),
                hip=[round(float(d.xpos[m.body("l_thigh").id][0] - fx) * 100, 2), round(float(d.xpos[m.body("l_thigh").id][2] - sole) * 100, 2)])


def best_touch(B, hp_lim, wp_lim):
    best = None
    for hp in np.arange(-hp_lim, -1.39, 0.2):
        for kn in (2.5, 2.3, 2.1):
            for an in (-1.0, -0.8, -0.6):
                for wp in np.arange(wp_lim, -0.01, -0.2):
                    for sp in np.arange(-2.0, 0.41, 0.2):
                        for el in (0.0, -0.4, -0.8, -1.2):
                            B.flat(hp, kn, an, wp, 0.0, sp, el)
                            r = B.measure()
                            if abs(r["palm_z"] - 0.010) < 0.010 and r["hip_z"] > 0.045 and r["palm_x"] > -0.02:
                                if best is None or abs(r["com"]) < best[0]:
                                    best = (abs(r["com"]), (hp, kn, an, wp, 0.0, sp, el), r)
    return best


def main():
    out = {}
    # 1. 现在的身体：站着；手掌着地时重心最靠近脚的姿势
    B0 = F.Body(F.build())
    B0.flat(-0.025, 0.05, -0.025, 0.0, 0.0, 0.0, -0.2)
    out["now_stand"] = shapes(B0)
    b = best_touch(B0, 2.2, 0.6)
    B0.flat(*b[1])
    out["now_squat"] = dict(shapes(B0), gap_mm=round(b[2]["gap"] * 1000, 1))
    # 2. 新设计
    B1 = F.Body(F.build(**NEW))
    B1.flat(-0.025, 0.05, -0.025, 0.0, 0.0, 0.0, -0.2)
    out["new_stand"] = shapes(B1)
    res = json.load(open(os.path.join(HERE, "flex_study_result.json"), encoding="utf-8"))["full_fwd15"]["best"]
    B1.flat(res["hp"], res["kn"], res["an"], res["wp"], res["roll"], res["sp"], res["el"])
    out["new_squat"] = dict(shapes(B1), gap_mm=res["gap_mm"], pose=res)
    # 3. 站着时重心相对脚掌中心的位置（髋轴前移、骨盆后移以后会往后跑多少）
    out["stand_com_cm"] = dict(now=out["now_stand"]["com"], new=out["new_stand"]["com"])
    # 4. 腰弯到 57° 要多大的力：上身弯下去时，腰电机要顶住的重力矩
    m = F.build(two_seg=True)
    d = mujoco.MjData(m)
    A = {m.joint(j).name: (m.jnt_qposadr[j], m.jnt_dofadr[j]) for j in range(m.njnt)}
    tq = {}
    for name, w1, w2 in (("一个关节弯 57°", 1.0, 0.0), ("两段各弯 29°", 0.5, 0.5), ("现在：一个关节弯 34°", 0.6, 0.0)):
        d.qpos[:] = m.qpos0
        d.qpos[A["waist_pitch"][0]], d.qpos[A["waist_pitch2"][0]] = w1, w2
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        tq[name] = dict(lower=round(abs(float(d.qfrc_bias[A["waist_pitch"][1]])), 2), upper=round(abs(float(d.qfrc_bias[A["waist_pitch2"][1]])), 2))
    out["waist_torque_Nm"] = tq
    json.dump(out, open(os.path.join(HERE, "flex_fig.json"), "w", encoding="utf-8"), ensure_ascii=False)
    table = json.load(open(os.path.join(HERE, "flex_study_result.json"), encoding="utf-8"))
    with open(os.path.join(HERE, "..", "hipwaist_fig.js"), "w", encoding="utf-8") as f:
        f.write("window.FLEX=" + json.dumps(dict(fig=out, table=table), ensure_ascii=False, separators=(",", ":")) + ";\n")
    print("站着时重心离脚掌中心（cm，负 = 偏后）：", out["stand_com_cm"])
    print("腰的重力矩（N·m，上限 2.0）：", tq)
    print("现在的身体，手掌着地时：重心", out["now_squat"]["com"], "cm；新设计：", out["new_squat"]["com"], "cm，间隙", out["new_squat"]["gap_mm"], "mm", res)


if __name__ == "__main__":
    main()
