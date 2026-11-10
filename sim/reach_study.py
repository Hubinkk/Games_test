"""
55 cm 人偶 · 手够不够得着地：蹲着、手掌着地时，重心能不能在脚上？

用法：
    python reach_study.py        # 写 reach_study_result.json

"手不离地的起身"需要一个过渡姿势：蹲着，手掌撑地，然后把重心交给脚、松手站起来。
这里不做动力学，只用身体数据（每段的长度和重量）把蹲姿扫一遍，看这个姿势存不存在。
脚掌长 6 cm（中心前后各 3 cm）：重心离脚掌中心 3 cm 以内才算"在脚上"。
比较四种身体：现在的；手臂加长 3 cm；髋和腰能弯得更深；两样都改。
"""
import json
import math
import os

import mujoco
import numpy as np

import full_build as fb

HERE = os.path.dirname(os.path.abspath(__file__))


def model(arm_extra=0.0, deep=False):
    s = open(fb.OUT, encoding="utf-8").read()
    if arm_extra:
        a = '<geom type="capsule" fromto="0 0 0 0 0 -0.075" size="0.009" mass="0.072"/>'
        assert s.count(a) == 2
        s = s.replace(a, f'<geom type="capsule" fromto="0 0 0 0 0 {-0.075 - arm_extra:.3f}" size="0.009" mass="0.072"/>')
        for k in "LR":
            b = f'<body name="{k}_hand" pos="0 0 -0.081"'
            assert s.count(b) == 1
            s = s.replace(b, f'<body name="{k}_hand" pos="0 0 {-0.081 - arm_extra:.3f}"')
    m = mujoco.MjModel.from_xml_string(s.replace('meshdir="', 'meshdir="' + HERE + "/"))
    lim = dict(hp=-2.2, wp=0.6)
    if deep:
        lim = dict(hp=-2.6, wp=1.0)
        for j in ("l_hip_pitch", "r_hip_pitch"):
            m.jnt_range[m.joint(j).id][0] = -2.6
        m.jnt_range[m.joint("waist_pitch").id][1] = 1.0
    return m, lim


def scan(m, lim):
    d = mujoco.MjData(m)
    adr = {m.joint(j).name: m.jnt_qposadr[j] for j in range(m.njnt)}
    fg = m.geom("l_foot").id
    out = []
    for hp in np.arange(lim["hp"], -1.39, 0.2):
        for kn in (2.5, 2.3, 2.1, 1.9):
            for an in (-1.0, -0.8, -0.6):
                for wp in np.arange(lim["wp"], -0.01, -0.2):
                    for sp in np.arange(-2.2, 0.61, 0.2):
                        for el in (0.0, -0.4, -0.8, -1.2):
                            d.qpos[:] = m.qpos0
                            th = -(hp + kn + an)                    # 脚掌放平时骨盆的前倾角
                            d.qpos[3:7] = [math.cos(th / 2), 0, math.sin(th / 2), 0]
                            for s_ in "lr":
                                d.qpos[adr[s_ + "_hip_pitch"]], d.qpos[adr[s_ + "_knee"]], d.qpos[adr[s_ + "_ankle_pitch"]] = hp, kn, an
                                d.qpos[adr[s_ + "_shoulder_pitch"]], d.qpos[adr[s_ + "_elbow"]] = sp, el
                            d.qpos[adr["waist_pitch"]] = wp
                            mujoco.mj_kinematics(m, d)
                            mujoco.mj_comPos(m, d)
                            foot = d.geom_xpos[fg]
                            sole = foot[2] - m.geom_size[fg][2]
                            palm = d.xpos[m.body("L_palm").id]
                            hip = d.xpos[m.body("l_thigh").id]
                            if abs(palm[2] - sole - 0.010) < 0.010 and hip[2] - sole > 0.045 and palm[0] > foot[0] - 0.02:
                                out.append(dict(com_cm=round(float(d.subtree_com[0][0] - foot[0]) * 100, 1),
                                                palm_cm=round(float(palm[0] - foot[0]) * 100, 1), tilt=round(math.degrees(th)),
                                                pose=dict(hp=round(float(hp), 1), kn=kn, an=an, wp=round(float(wp), 1), sp=round(float(sp), 1), el=el)))
    return out


def main():
    res = {}
    for key, title, arm, deep in (("now", "现在的身体", 0.0, False), ("arm", "手臂加长 3 cm", 0.03, False),
                                  ("deep", "髋能弯到 149°、腰能弯到 57°", 0.0, True), ("both", "两样都改", 0.03, True)):
        m, lim = model(arm, deep)
        poses = scan(m, lim)
        best = min(poses, key=lambda p: abs(p["com_cm"])) if poses else None
        ok = sum(1 for p in poses if abs(p["com_cm"]) <= 3.0)
        res[key] = dict(title=title, n_touch=len(poses), n_balanced=ok, best=best)
        print(f"{title}：手掌能着地的蹲姿 {len(poses)} 个，其中重心在脚上的 {ok} 个；"
              + (f"重心离脚掌中心最近 {best['com_cm']} cm（手在脚前 {best['palm_cm']} cm，上身前倾 {best['tilt']}°）" if best else "一个也没有"))
    json.dump(res, open(os.path.join(HERE, "reach_study_result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
