"""
55 cm 人偶 · 髋和腰要弯到多少，结构上怎么让开（2026-11-10，育苗人决定先做这部分的机械设计）

用法：
    python flex_study.py        # 写 flex_study_result.json 和 flex_study_poses.json（给网页画图）

问题：现在的身体找不到"蹲着、手掌着地、重心在脚上"的姿势（reach_study.py）。
那次扫描没有检查身体自己会不会撞到自己。这里补上：大腿、小腿不能插进骨盆、肚子、胸口（允许压进软皮 3 mm）。

每种设计，扫一遍蹲姿（髋、膝、踝、腰、膝盖外开的角度、手臂），数出：
  手掌着地（离地 2 cm 以内）、脚掌放平、重心在脚掌内（离中心 ≤ 2.5 cm）、不撞自己 的姿势有多少个。
设计的旋钮：
  髋的弯曲上限（现在 126°）、腰的弯曲上限（现在 34°）、腰分成两段、髋轴往前挪、深蹲时膝盖朝外打开。
只算几何和重量，不是动力学。
"""
import json
import math
import os

import mujoco
import numpy as np

import full_build as fb

HERE = os.path.dirname(os.path.abspath(__file__))
SOFT = 0.003          # 软皮可以压进去 3 mm
COM_OK = 0.025        # 重心离脚掌中心 2.5 cm 以内（脚掌前后各 3 cm，留 5 mm）


def build(two_seg=False, hip_dx=0.0):
    s = open(fb.OUT, encoding="utf-8").read()
    if two_seg:           # 腰分两段：在胸口再加一个俯仰关节
        a = '<body name="chest" pos="0 0 0.05">'
        assert s.count(a) == 1
        s = s.replace(a, a + '\n          <joint name="waist_pitch2" axis="0 1 0" range="-0.2 0.8" class="big"/>')
    m = mujoco.MjModel.from_xml_string(s)
    if hip_dx:
        for side in "lr":
            m.body_pos[m.body(f"{side}_hip_yaw_link").id][0] += hip_dx
    return m


class Body:
    def __init__(self, m):
        self.m, self.d = m, mujoco.MjData(m)
        self.adr = {m.joint(j).name: m.jnt_qposadr[j] for j in range(m.njnt)}
        self.foot = m.geom("l_foot").id
        leg = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name in ("l_thigh", "l_shin") and m.geom_contype[g]]
        torso = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name in ("pelvis", "abdomen", "chest", "head")
                 and m.geom_contype[g] and m.geom(g).name != "backpack"]
        self.pairs = [(a, b) for a in leg for b in torso]
        self.two = "waist_pitch2" in self.adr

    def set(self, hp, kn, an, wp, roll, sp, el, th, aroll):
        d, A = self.d, self.adr
        d.qpos[:] = self.m.qpos0
        d.qpos[3:7] = [math.cos(th / 2), 0, math.sin(th / 2), 0]
        for s_, sg in (("l", 1), ("r", -1)):
            d.qpos[A[s_ + "_hip_pitch"]], d.qpos[A[s_ + "_knee"]], d.qpos[A[s_ + "_ankle_pitch"]] = hp, kn, an
            d.qpos[A[s_ + "_hip_roll"]] = sg * roll
            d.qpos[A[s_ + "_ankle_roll"]] = sg * aroll
            d.qpos[A[s_ + "_shoulder_pitch"]], d.qpos[A[s_ + "_elbow"]] = sp, el
            d.qpos[A[s_ + "_shoulder_roll"]] = sg * 0.25
        if self.two:
            d.qpos[A["waist_pitch"]] = d.qpos[A["waist_pitch2"]] = wp / 2
        else:
            d.qpos[A["waist_pitch"]] = wp
        mujoco.mj_kinematics(self.m, d)

    def flat(self, hp, kn, an, wp, roll, sp, el):
        """解出骨盆前倾角和踝的侧摆，让脚掌放平。"""
        th, ar = -(hp + kn + an), -roll

        def res(th, ar):
            self.set(hp, kn, an, wp, roll, sp, el, th, ar)
            n = self.d.geom_xmat[self.foot].reshape(3, 3)[:, 2]
            return np.array([n[0], n[1]])
        for _ in range(4):
            r0 = res(th, ar)
            if np.abs(r0).max() < 1e-4:
                break
            J = np.column_stack([(res(th + 1e-4, ar) - r0) / 1e-4, (res(th, ar + 1e-4) - r0) / 1e-4])
            try:
                step = np.linalg.solve(J, -r0)
            except np.linalg.LinAlgError:
                break
            th, ar = th + step[0], ar + step[1]
        self.set(hp, kn, an, wp, roll, sp, el, th, ar)
        return th, ar

    def measure(self):
        m, d = self.m, self.d
        mujoco.mj_comPos(m, d)
        foot = d.geom_xpos[self.foot]
        sole = foot[2] - m.geom_size[self.foot][2]
        palm = d.xpos[m.body("L_palm").id]
        ft = np.zeros(6)
        pen = min(mujoco.mj_geomDistance(m, d, a, b, 0.05, ft) for a, b in self.pairs)
        return dict(com=float(d.subtree_com[0][0] - foot[0]), palm_z=float(palm[2] - sole), palm_x=float(palm[0] - foot[0]),
                    hip_z=float(d.xpos[m.body("l_thigh").id][2] - sole), gap=float(pen))


def scan(m, hp_lim, wp_lim, rolls):
    B = Body(m)
    lo_ar, hi_ar = m.jnt_range[m.joint("l_ankle_roll").id]
    n_touch = n_bal = n_ok = 0
    best = None
    worst_gap = 0.0
    for hp in np.arange(-hp_lim, -1.39, 0.2):
        for kn in (2.5, 2.3, 2.1):
            for an in (-1.0, -0.8, -0.6):
                for wp in np.arange(wp_lim, -0.01, -0.2):
                    for roll in rolls:
                        for sp in np.arange(-2.0, 0.41, 0.2):
                            for el in (0.0, -0.4, -0.8, -1.2):
                                th, ar = B.flat(hp, kn, an, wp, roll, sp, el)
                                if not (lo_ar - 0.01 <= ar <= hi_ar + 0.01):
                                    continue
                                r = B.measure()
                                if not (abs(r["palm_z"] - 0.010) < 0.010 and r["hip_z"] > 0.045 and r["palm_x"] > -0.02):
                                    continue
                                n_touch += 1
                                if abs(r["com"]) > COM_OK:
                                    continue
                                n_bal += 1
                                worst_gap = min(worst_gap, r["gap"])
                                if r["gap"] < -SOFT:
                                    continue
                                n_ok += 1
                                score = abs(r["com"]) - min(r["gap"], 0.01)
                                if best is None or score < best[0]:
                                    best = (score, dict(hp=round(float(hp), 2), kn=kn, an=an, wp=round(float(wp), 2), roll=roll,
                                                        sp=round(float(sp), 2), el=el, th=round(float(th), 3), aroll=round(float(ar), 3),
                                                        com_cm=round(r["com"] * 100, 1), gap_mm=round(r["gap"] * 1000, 1),
                                                        palm_cm=round(r["palm_x"] * 100, 1), tilt_deg=round(math.degrees(th))))
    return dict(touch=n_touch, balanced=n_bal, ok=n_ok, best=best[1] if best else None,
                deepest_overlap_mm=round(-worst_gap * 1000, 1))


DESIGNS = [
    # key, 名字, 髋上限, 腰上限, 腰两段, 髋轴前移, 膝盖外开
    ("now", "现在的身体（髋 126°，腰 34°）", 2.2, 0.6, False, 0.0, (0.0,)),
    ("now_open", "现在的身体 + 膝盖外开", 2.2, 0.6, False, 0.0, (0.0, 0.3, 0.6)),
    ("hip", "只加大髋（149°）", 2.6, 0.6, False, 0.0, (0.0, 0.3, 0.6)),
    ("waist", "只加大腰（57°，一个关节）", 2.2, 1.0, False, 0.0, (0.0, 0.3, 0.6)),
    ("both_closed", "髋 149° + 腰 57°，膝盖并拢", 2.6, 1.0, False, 0.0, (0.0,)),
    ("both", "髋 149° + 腰 57° + 膝盖外开", 2.6, 1.0, False, 0.0, (0.0, 0.3, 0.6)),
    ("both_2seg", "同上，腰分两段", 2.6, 1.0, True, 0.0, (0.0, 0.3, 0.6)),
    ("both_2seg_fwd", "同上，髋轴再前移 1 cm", 2.6, 1.0, True, 0.01, (0.0, 0.3, 0.6)),
    ("mid_2seg", "髋 138° + 腰 46°（两段）+ 膝盖外开", 2.4, 0.8, True, 0.0, (0.0, 0.3, 0.6)),
]


def main():
    out, poses = {}, {}
    for key, title, hp, wp, two, dx, rolls in DESIGNS:
        m = build(two, dx)
        r = scan(m, hp, wp, rolls)
        out[key] = dict(title=title, **r)
        b = r["best"]
        print(f"{title}：手掌着地 {r['touch']}；重心也在脚上 {r['balanced']}；还不撞自己 {r['ok']}"
              + (f"；最好的：重心偏 {b['com_cm']} cm，间隙 {b['gap_mm']} mm，髋 {abs(b['hp'])}、腰 {b['wp']}、外开 {b['roll']}" if b else "")
              + f"；重心在脚上的姿势里最深的重叠 {r['deepest_overlap_mm']} mm", flush=True)
    json.dump(out, open(os.path.join(HERE, "flex_study_result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
