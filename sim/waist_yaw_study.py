"""
转腰（waist_yaw）值不值：在现在的整体模型上量（仿真，mujoco 3.17，CPU）。

核查结果先说：整体模型从第一版身体起就有转腰关节 waist_yaw（骨盆和腰俯仰之间，竖直轴，范围 ±0.8 弧度 ≈ ±46°，
"big" 一档，执行器扭矩上限 ±1.5 N·m，kp 8），所以这里没有另做新身体，直接在现有关节上比"锁住"和"能转"。

用法（~/venv-sys）：
    python waist_yaw_study.py            # 写 waist_yaw_study.json，打印结果

1. 手能够到的范围（只算运动学，不跑动力学）
   - 身体站着不动脚：腿和骨盆固定在站姿（整体模型站稳 2 秒后的姿势），只动上身
   - 随机取上身的关节角（腰俯仰两段同一个角、左右肩的俯仰/侧摆/旋转、肘，都在模型的关节范围里均匀取）：
     A = 转腰锁在 0；B = 转腰在 ±45° 里均匀取。每种 N_SAMPLE 组
   - 只留"重心还在两只脚撑起来的范围里"的姿势（重心投影在两只脚掌外框的凸包内，往里缩 5 mm）
   - 左右手掌中心的位置按 2 cm 一格记下来，数格子：格子数 × 8 cm³ = 够得到的体积
   - 按方向分：以骨盆中心为原点、骨盆朝向为前，水平方位角 |角| ≤ 45° 算前，45–112.5° 侧，112.5–157.5° 侧后，> 157.5° 后
   - 没有算手和身体之间会不会穿模（模型里手和身体不碰撞），所以有一些格子在身体里面；A、B 一样处理，比的是差别
2. 费不费力：完整身体站着（平衡控制 + 自动配平，和 day.py 平时站的一样），转腰 1.5 秒平滑转到目标再停 3 秒，
   量停住那 3 秒里转腰电机、两边脚踝侧摆的扭矩均方根（以及转的过程中转腰的最大扭矩）；
   温度按 day.py 的热模型（Body：大关节 4 K/W、300 秒，小关节 15 K/W、150 秒）算"一直这样停着"的稳态和停 10 分钟的温度。
   情形：转腰 0（对照）、+45°、−45°、+45° 同时左手前伸、+45° 同时右手前伸（肩俯仰到水平、肘伸直）、转腰 0 左手前伸（对照）；
   再加快转：0.5 秒、0.25 秒转到 +45°（看转得快时转腰电机和脚踝侧摆的最大扭矩）。
"""
import json
import math
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import getup_hands as G          # noqa: E402
from balance import Balance      # noqa: E402
from figure_test import stand_pose   # noqa: E402

N_SAMPLE = 200000
YAW = math.radians(45)
VOX = 0.02
SEED = 0
UPPER = ["waist_pitch", "l_shoulder_pitch", "l_shoulder_roll", "l_shoulder_yaw", "l_elbow",
         "r_shoulder_pitch", "r_shoulder_roll", "r_shoulder_yaw", "r_elbow"]


def smooth(a):
    a = max(0.0, min(1.0, a))
    return 0.5 - 0.5 * math.cos(math.pi * a)


def hull(pts):
    pts = sorted(map(tuple, pts))
    cross = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])   # noqa: E731
    lo, up = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    return np.array(lo[:-1] + up[:-1])


def inside(h, p, margin):
    """p 在凸包 h（逆时针）里面，并且离每条边至少 margin。"""
    for i in range(len(h)):
        a, b = h[i], h[(i + 1) % len(h)]
        e = b - a
        if (e[0] * (p[1] - a[1]) - e[1] * (p[0] - a[0])) / np.linalg.norm(e) < margin:
            return False
    return True


def settle_stand(sim, secs=2.0):
    sim.reset()
    bal = Balance(sim)
    for _ in range(int(secs / sim.m.opt.timestep)):
        sim.set_targets(bal.adjust(stand_pose()))
        sim.step()
    return bal


def reach(sim):
    m = sim.m
    settle_stand(sim)
    d = mujoco.MjData(m)
    d.qpos[:] = sim.d.qpos
    mujoco.mj_fwdPosition(m, d)
    pel = m.body("pelvis").id
    R0 = d.xmat[pel].reshape(3, 3).copy()
    c0 = d.xpos[pel].copy()
    # 两只脚掌的外框（8 个角）→ 支撑范围
    corners = []
    for g in ("l_foot", "r_foot"):
        gid = m.geom(g).id
        sz, Rg, pg = m.geom_size[gid], d.geom_xmat[gid].reshape(3, 3), d.geom_xpos[gid]
        for sx in (-1, 1):
            for sy in (-1, 1):
                corners.append((pg + Rg @ np.array([sx * sz[0], sy * sz[1], -sz[2]]))[:2])
    sup = hull(np.array(corners))
    adr = {n: m.jnt_qposadr[m.joint(n).id] for n in UPPER + ["waist_pitch2", "waist_yaw"]}
    rng_j = {n: m.jnt_range[m.joint(n).id] for n in UPPER}
    palms = [m.body("L_palm").id, m.body("R_palm").id]
    q_base = d.qpos.copy()
    out = {}
    for tag, yaw_rng in (("锁住", (0.0, 0.0)), ("能转±45°", (-YAW, YAW))):
        rng = np.random.default_rng(SEED)
        cells, kept = set(), 0
        far_back, far_side = 0.0, 0.0
        for _ in range(N_SAMPLE):
            d.qpos[:] = q_base
            for n in UPPER:
                d.qpos[adr[n]] = rng.uniform(*rng_j[n])
            d.qpos[adr["waist_pitch2"]] = d.qpos[adr["waist_pitch"]]
            d.qpos[adr["waist_yaw"]] = rng.uniform(*yaw_rng) if yaw_rng[1] else 0.0
            mujoco.mj_fwdPosition(m, d)
            if not inside(sup, d.subtree_com[pel][:2], 0.005):
                continue
            kept += 1
            for b in palms:
                p = R0.T @ (d.xpos[b] - c0)              # 骨盆坐标：x 前、y 左、z 上
                cells.add(tuple(np.floor(p / VOX).astype(int)))
                far_back = min(far_back, p[0])
                far_side = max(far_side, abs(p[1]))
        sec = {"前": 0, "侧": 0, "侧后": 0, "后": 0}
        for c in cells:
            x, y = (c[0] + 0.5) * VOX, (c[1] + 0.5) * VOX
            a = abs(math.degrees(math.atan2(y, x)))
            sec["前" if a <= 45 else "侧" if a <= 112.5 else "侧后" if a <= 157.5 else "后"] += 1
        out[tag] = dict(samples=N_SAMPLE, balanced=kept, cells=len(cells), volume_cm3=round(len(cells) * (VOX * 100) ** 3),
                        sectors_cm3={k: round(v * (VOX * 100) ** 3) for k, v in sec.items()},
                        farthest_back_cm=round(-far_back * 100, 1), farthest_side_cm=round(far_side * 100, 1),
                        cell_set=cells)
    a, b = out["锁住"]["cell_set"], out["能转±45°"]["cell_set"]
    out["只有能转时才够得到"] = round(len(b - a) * (VOX * 100) ** 3)
    out["只有锁住时才够得到"] = round(len(a - b) * (VOX * 100) ** 3)
    for v in (out["锁住"], out["能转±45°"]):
        v.pop("cell_set")
    return out


def arm_forward(sim):
    """肩俯仰取哪个角时手臂朝正前方水平：在模型里试出来（不猜符号）。"""
    m = sim.m
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    best = None
    for sp in np.linspace(-3.0, 1.0, 81):
        d.qpos[m.jnt_qposadr[m.joint("l_shoulder_pitch").id]] = sp
        mujoco.mj_fwdPosition(m, d)
        v = d.xpos[m.body("L_palm").id] - d.xpos[m.body("l_upper_arm").id]
        v = v / np.linalg.norm(v)
        if best is None or v[0] > best[1]:
            best = (float(sp), float(v[0]), float(v[2]))
    return best


def effort(sim, body_cls, yaw, arm=None, sp_fwd=None, T_RAMP=1.5):
    from day import T_AMB
    bal = settle_stand(sim)
    dt = sim.m.opt.timestep
    if getattr(sim, "trim", None) is not None:
        sim.trim.enabled = True
    names = ["waist_yaw", "l_ankle_roll", "r_ankle_roll", "l_ankle_pitch", "r_ankle_pitch", "waist_pitch",
             "l_shoulder_pitch", "r_shoulder_pitch"]
    acc, n, peak_yaw, fell = {k: 0.0 for k in names}, 0, 0.0, None
    peak_ar = 0.0
    T_HOLD = 3.0
    for k in range(int((T_RAMP + T_HOLD) / dt)):
        t = k * dt
        a = smooth(t / T_RAMP)
        pose = stand_pose()
        pose["waist_yaw"] = yaw * a
        if arm:
            pose[f"{arm}_shoulder_pitch"] = sp_fwd * a
            pose[f"{arm}_elbow"] = -0.2 * (1 - a)
            pose[f"{arm}_shoulder_roll"] = (0.15 if arm == "l" else -0.15) * (1 - a)
        sim.set_targets(bal.adjust(pose))
        sim.step()
        tq = sim.torques()
        peak_yaw = max(peak_yaw, abs(float(tq.get("waist_yaw", 0.0))))
        peak_ar = max(peak_ar, abs(float(tq.get("l_ankle_roll", 0.0))), abs(float(tq.get("r_ankle_roll", 0.0))))
        if t >= T_RAMP:
            for j in names:
                acc[j] += float(tq.get(j, 0.0)) ** 2
            n += 1
        if fell is None and sim.fallen():
            fell = round(t, 2)
    m, d = sim.m, sim.d
    yaw_got = math.degrees(float(d.qpos[m.jnt_qposadr[m.joint("waist_yaw").id]]))
    rows = {}
    for j in names:
        s = acc[j] / max(1, n)
        row = dict(rms_Nm=round(math.sqrt(s), 3))
        if j in body_cls.k:
            rise = body_cls.R[j] * body_cls.k[j] * s
            row.update(limit_Nm=round(float(sim.limit[j]), 2), T_10min=round(T_AMB + rise * (1 - math.exp(-600 / body_cls.tau[j])), 1),
                       T_steady=round(T_AMB + rise, 1))
        rows[j] = row
    return dict(yaw_cmd_deg=round(math.degrees(yaw), 1), yaw_got_deg=round(yaw_got, 1), arm=arm, ramp_s=T_RAMP, fell_at=fell,
                peak_waist_yaw_Nm=round(peak_yaw, 3), peak_ankle_roll_Nm=round(peak_ar, 3), hold=rows, tilt_end=round(sim.tilt_deg(), 1))


def main():
    import day
    sim = G.get_sim()
    m = sim.m
    j = m.joint("waist_yaw").id
    a = next(i for i in range(m.nu) if m.actuator(i).name == "waist_yaw")
    info = dict(range_deg=[round(math.degrees(x), 1) for x in m.jnt_range[j]], forcerange_Nm=[float(x) for x in m.actuator_forcerange[a]],
                kp=float(m.actuator_gainprm[a, 0]), armature=float(m.dof_armature[m.jnt_dofadr[j]]),
                parent=m.body(m.jnt_bodyid[j]).name, total_mass_kg=round(float(sum(m.body_mass)), 3))
    print("现有的转腰关节：", info, flush=True)
    sp = arm_forward(sim)
    print(f"手臂朝正前方：肩俯仰 {sp[0]:.2f} 弧度（方向向量 x={sp[1]:.2f}, z={sp[2]:.2f}）", flush=True)
    B = day.Body(sim)
    eff = []
    for yaw, arm, ramp in ((0.0, None, 1.5), (YAW, None, 1.5), (-YAW, None, 1.5), (YAW, "l", 1.5), (YAW, "r", 1.5), (0.0, "l", 1.5),
                           (YAW, None, 0.5), (YAW, None, 0.25), (YAW, "r", 0.25)):
        r = effort(sim, B, yaw, arm, sp[0], ramp)
        print(r, flush=True)
        eff.append(r)
    rch = reach(sim)
    print(json.dumps(rch, ensure_ascii=False), flush=True)
    json.dump(dict(note="仿真（mujoco 3.17，整体模型 figure55_full.xml，CPU）；温度是 day.py 热模型的估算",
                   joint=info, arm_forward_shoulder_pitch=sp[0], effort=eff, reach=rch),
              open(os.path.join(HERE, "waist_yaw_study.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
