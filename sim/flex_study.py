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


def build(two_seg=False, hip_dx=0.0, cut=0.0, back=0.0, abd_r=0.03, arm=0.0, chest_x=0.028):
    """two_seg 腰分两段；hip_dx 髋轴前移；cut 骨盆前面削掉；back 骨盆后面（屁股）加厚；abd_r 腰的半径"""
    s = open(fb.OUT, encoding="utf-8").read()
    if two_seg:
        a = '<body name="chest" pos="0 0 0.05">'
        assert s.count(a) == 1
        s = s.replace(a, a + '\n          <joint name="waist_pitch2" axis="0 1 0" range="-0.2 0.8" class="big"/>')
    if arm:               # 前臂加长（只做对比用）
        a = '<geom type="capsule" fromto="0 0 0 0 0 -0.075" size="0.009" mass="0.072"/>'
        s = s.replace(a, f'<geom type="capsule" fromto="0 0 0 0 0 {-0.075 - arm:.3f}" size="0.009" mass="0.072"/>')
        for k in "LR":
            s = s.replace(f'<body name="{k}_hand" pos="0 0 -0.081"', f'<body name="{k}_hand" pos="0 0 {-0.081 - arm:.3f}"')
    m = mujoco.MjModel.from_xml_string(s)
    gc = m.geom("chest").id
    m.geom_size[gc][0] = chest_x                                   # 胸口前后的厚度（半厚）
    for side in "lr":
        m.body_pos[m.body(f"{side}_hip_yaw_link").id][0] += hip_dx
    g = m.geom("pelvis").id
    m.geom_size[g][0] += (back - cut) / 2
    m.geom_pos[g][0] -= (cut + back) / 2
    m.geom_sameframe[g] = 0                                       # 不然改了位置也不生效
    m.body_sameframe[m.body("pelvis").id] = 0
    m.body_ipos[m.body("pelvis").id][0] -= (cut + back) / 2      # 骨盆里的东西跟着往后挪
    for gi in range(m.ngeom):
        if m.body(m.geom_bodyid[gi]).name == "abdomen" and m.geom_contype[gi]:
            m.geom_size[gi][0] = abd_r
    return m


class Body:
    def __init__(self, m):
        self.m, self.d = m, mujoco.MjData(m)
        self.adr = {m.joint(j).name: m.jnt_qposadr[j] for j in range(m.njnt)}
        self.foot = m.geom("l_foot").id
        self.torso = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name in ("pelvis", "abdomen", "chest", "head")
                      and m.geom_contype[g] and m.geom(g).name != "backpack"]
        self.thigh, self.shin = m.body("l_thigh").id, m.body("l_shin").id
        # 沿大腿（半径 2 cm）和小腿（半径 1.3 cm）取点；大腿根 3 cm 以内是球形关节，不算
        self.pts = [(self.thigh, -s_, 0.020, "大腿") for s_ in (0.03, 0.05, 0.07, 0.09, 0.115)] + \
                   [(self.shin, -s_, 0.013, "小腿") for s_ in (0.03, 0.06, 0.09, 0.115)]
        self.two = "waist_pitch2" in self.adr
        self.where = ""

    def sdf(self, g, p):
        """点 p 到几何体 g 表面的距离（里面为负）"""
        m, d = self.m, self.d
        q = d.geom_xmat[g].reshape(3, 3).T @ (p - d.geom_xpos[g])
        t, sz = m.geom_type[g], m.geom_size[g]
        if t == 2:
            return float(np.linalg.norm(q) - sz[0])
        if t == 3:
            q2 = q.copy()
            q2[2] -= np.clip(q2[2], -sz[1], sz[1])
            return float(np.linalg.norm(q2) - sz[0])
        if t == 6:
            a = np.abs(q) - sz
            return float(np.linalg.norm(np.maximum(a, 0)) + min(a.max(), 0.0))
        return 1.0

    def set(self, hp, kn, an, wp, roll, sp, el, th, aroll):
        d, A = self.d, self.adr
        d.qpos[:] = self.m.qpos0
        d.qpos[3:7] = [math.cos(th / 2), 0, math.sin(th / 2), 0]
        for s_, sg in (("l", 1), ("r", -1)):
            d.qpos[A[s_ + "_hip_pitch"]], d.qpos[A[s_ + "_knee"]], d.qpos[A[s_ + "_ankle_pitch"]] = hp, kn, an
            d.qpos[A[s_ + "_hip_yaw"]] = sg * roll      # 膝盖朝外打开：大腿抬高以后，这个动作是髋的"转"（绕竖轴），不是侧摆
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
        pen = 1.0
        for b, z, r, nm in self.pts:
            p = d.xpos[b] + d.xmat[b].reshape(3, 3) @ np.array([0.0, 0.0, z])
            for g in self.torso:
                c = self.sdf(g, p) - r
                if c < pen:
                    pen, self.where = c, nm + "-" + {"pelvis": "骨盆", "abdomen": "肚子", "chest": "胸口", "head": "头"}[m.body(m.geom_bodyid[g]).name]
        return dict(com=float(d.subtree_com[0][0] - foot[0]), palm_z=float(palm[2] - sole), palm_x=float(palm[0] - foot[0]),
                    hip_z=float(d.xpos[m.body("l_thigh").id][2] - sole), gap=float(pen))


def scan(m, hp_lim, wp_lim, rolls):
    B = Body(m)
    lo_ar, hi_ar = m.jnt_range[m.joint("l_ankle_roll").id]
    n_touch = n_bal = n_ok = 0
    best = None
    worst_gap, best_gap, best_where = 0.0, -1.0, ""
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
                                if r["gap"] > best_gap:
                                    best_gap, best_where = r["gap"], B.where
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
                deepest_overlap_mm=round(-worst_gap * 1000, 1),
                least_overlap_mm=round(-best_gap * 1000, 1) if n_bal else None, least_where=best_where)


R3 = (0.0, 0.3, 0.6)
DESIGNS = [
    # key, 名字, 髋上限, 腰上限, 膝盖外开, build 的参数
    ("now", "现在的身体（髋 126°，腰 34°）", 2.2, 0.6, (0.0,), {}),
    ("now_open", "现在的身体 + 膝盖外开", 2.2, 0.6, R3, {}),
    ("hip", "只加大髋（149°）", 2.6, 0.6, R3, {}),
    ("waist", "只加大腰（57°）", 2.2, 1.0, R3, {}),
    ("both_closed", "髋 149° + 腰 57°，膝盖并拢", 2.6, 1.0, (0.0,), {}),
    ("both", "髋 149° + 腰 57° + 膝盖外开", 2.6, 1.0, R3, {}),
    ("two", "同上，腰分两段", 2.6, 1.0, R3, dict(two_seg=True)),
    ("two_cut", "腰两段 + 骨盆前面削 2 cm、后面加 2 cm", 2.6, 1.0, R3, dict(two_seg=True, cut=0.02, back=0.02)),
    ("two_fwd1", "腰两段 + 髋轴前移 1 cm", 2.6, 1.0, R3, dict(two_seg=True, hip_dx=0.01)),
    ("two_fwd1_cut", "腰两段 + 髋轴前移 1 cm + 骨盆前削后加 2 cm", 2.6, 1.0, R3, dict(two_seg=True, hip_dx=0.01, cut=0.02, back=0.02)),
    ("full", "同上 + 腰细一点（半径 3 → 2.5 cm）", 2.6, 1.0, R3, dict(two_seg=True, hip_dx=0.01, cut=0.02, back=0.02, abd_r=0.025)),
    ("full_fwd15", "全部改动，髋轴前移 1.5 cm", 2.6, 1.0, R3, dict(two_seg=True, hip_dx=0.015, cut=0.02, back=0.02)),
    ("full_chest", "全部改动 + 胸口下缘前面收 6 mm", 2.6, 1.0, R3, dict(two_seg=True, hip_dx=0.01, cut=0.02, back=0.02, chest_x=0.022)),
    ("full_arm", "全部改动 + 手臂加长 1.5 cm", 2.6, 1.0, R3, dict(two_seg=True, hip_dx=0.01, cut=0.02, back=0.02, arm=0.015)),
    ("full_138_57", "全部改动，范围 髋 138° 腰 57°", 2.4, 1.0, R3, dict(two_seg=True, hip_dx=0.01, cut=0.02, back=0.02, abd_r=0.025)),
    ("full_149_46", "全部改动，范围 髋 149° 腰 46°", 2.6, 0.8, R3, dict(two_seg=True, hip_dx=0.01, cut=0.02, back=0.02, abd_r=0.025)),
    ("full_138_46", "全部改动，范围 髋 138° 腰 46°", 2.4, 0.8, R3, dict(two_seg=True, hip_dx=0.01, cut=0.02, back=0.02, abd_r=0.025)),
]


def main():
    out, poses = {}, {}
    for key, title, hp, wp, rolls, kw in DESIGNS:
        m = build(**kw)
        r = scan(m, hp, wp, rolls)
        out[key] = dict(title=title, **r)
        b = r["best"]
        print(f"{title}：手掌着地 {r['touch']}；重心也在脚上 {r['balanced']}；还不撞自己 {r['ok']}"
              + (f"；最好：重心偏 {b['com_cm']} cm，间隙 {b['gap_mm']} mm，髋 {abs(b['hp'])} 腰 {b['wp']} 开 {b['roll']}" if b else "")
              + (f"；最少重叠 {r['least_overlap_mm']} mm（{r['least_where']}）" if r["balanced"] else ""), flush=True)
    json.dump(out, open(os.path.join(HERE, "flex_study_result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
