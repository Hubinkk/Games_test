"""
手不离地的起身（2026-11-10，育苗人教的方法）

育苗人的话：两只手撑地的同时，膝盖弯曲，小腿蹬地站起来，躯体拱起来也要发力；用另一边的手臂撑住地面保持平衡。
花然以前的做法是跪直以后只靠两条腿站，手早早离开了地面，所以往后倒。

这里的动作是手工一步步排出来的关键帧（不是搜索，也不是强化学习）：
  俯卧撑（pushup.py 的手臂路径）→ 身体拱起来 → 脚收到身体下面（手一直撑着）→ 重心移到脚上 → 松手 → 站起

用法：
    python getup_hands.py            # 跑一次，打印每个关键帧结束时的状态；成功就写 getup_best_hands.json
    python getup_hands.py --quiet
沿用整体模型的扭矩上限，加 10 rad/s 转速上限。
"""
import json
import math
import os
import sys

import mujoco
import numpy as np

import agility
import figure_test
import full_build as fb
import full_demo as FD
from balance import Balance
from figure_test import stand_pose

HERE = os.path.dirname(os.path.abspath(__file__))
_SIM = None


def get_sim():
    global _SIM
    if _SIM is None:
        figure_test.MODEL = fb.OUT
        _SIM = FD.FullSim()
        agility.add_speed_limit(_SIM)
    return _SIM


def sm(a):
    a = max(0.0, min(1.0, a))
    return 0.5 - 0.5 * math.cos(math.pi * a)


def P(arm, L, R=None, wp=0.0, roll=(0.0, 0.0), neck=-0.3, arm_r=None):
    """arm = (肩俯仰, 肩侧举, 肩转, 肘)；L / R = (髋, 膝, 踝)；roll = (左髋侧摆, 右髋侧摆)"""
    R = L if R is None else R
    arm_r = arm if arm_r is None else arm_r
    p = stand_pose(0.05)
    p["l_shoulder_pitch"], p["l_shoulder_roll"], p["l_shoulder_yaw"], p["l_elbow"] = arm[0], -arm[1], -arm[2], arm[3]
    p["r_shoulder_pitch"], p["r_shoulder_roll"], p["r_shoulder_yaw"], p["r_elbow"] = arm_r
    p["l_hip_pitch"], p["l_knee"], p["l_ankle_pitch"] = L
    p["r_hip_pitch"], p["r_knee"], p["r_ankle_pitch"] = R
    p["l_hip_roll"], p["r_hip_roll"] = roll
    p["waist_pitch"] = wp
    p["neck_pitch"] = neck
    return p


def default_keyframes():
    px = json.load(open(os.path.join(HERE, "pushup_best.json"), encoding="utf-8"))["x"]
    arm = [tuple(px[f"{j}{i}"] for j in ("sp", "roll", "yaw", "el")) for i in range(3)]
    flat = (0.0, 0.05, 0.0)
    lie = P((0.0, -0.15, 0.0, -0.2), flat)
    ks = [(0.0, lie, "趴着"), (1.0, lie, "趴着")]
    t = 1.3
    ks.append((t, lie, "趴着"))
    for i, dt_ in enumerate((1.0, 1.0, 1.3)):
        t += dt_
        ks.append((t, P(arm[i], flat), f"俯卧撑 {i+1}"))
    return ks, arm, t


def simulate(ks, hold=1.5, verbose=True, record=None, t_free=None, start="prone", relax_neck=True, init=None):
    """ks: [(时间, 姿势, 名字)]。t_free：从这一刻起算"手可以离地"，接上平衡控制。"""
    sim = get_sim()
    m, d = sim.m, sim.d

    def target(t):
        for i in range(1, len(ks)):
            t0, p0 = ks[i - 1][:2]
            t1, p1 = ks[i][:2]
            if t <= t1:
                a = sm((t - t0) / (t1 - t0))
                return {k: p0.get(k, 0.0) + (p1.get(k, 0.0) - p0.get(k, 0.0)) * a for k in set(p0) | set(p1)}
        return dict(ks[-1][1])

    sim.reset()
    if start == "prone":      # 脸朝下趴着，头朝 +x
        d.qpos[0:3] = [0, 0, 0.06]
        d.qpos[3:7] = [math.cos(math.pi / 4), 0, math.sin(math.pi / 4), 0]
    else:                     # 仰面躺着，脚朝 +x（背后有背包，所以垫高一点再落下）
        d.qpos[0:3] = [0, 0, 0.10]
        d.qpos[3:7] = [math.cos(math.pi / 4), 0, -math.sin(math.pi / 4), 0]
    if init is not None:      # 直接从某个姿势开始（只用来单独检验后半段）
        init(sim)
    mujoco.mj_forward(m, d)
    bal = Balance(sim)
    bal.prev = None
    dt = m.opt.timestep
    t_end = ks[-1][0] + hold
    t_free = ks[-2][0] if t_free is None else t_free
    neck_i = [m.actuator(n).id for n in ("neck_pitch", "neck_low") if n in sim.act]
    neck_base = m.actuator_forcerange[neck_i].copy()
    if relax_neck:
        m.actuator_forcerange[neck_i] = neck_base * 0.05
    body_of = lambda g: m.body(m.geom_bodyid[g]).name
    marks = {round(k[0], 3): k[2] for k in ks[1:]}
    peak = {}
    balancing, restored, ki = False, False, 1
    hist = []
    trim = getattr(sim, "trim", None)
    for k in range(int(t_end / dt)):
        t = k * dt
        if trim is not None:          # 自动配平只在最后一段（走向站姿）打开
            trim.enabled = t >= ks[-2][0]
        tgt = target(t)
        if not restored and t >= t_free:
            m.actuator_forcerange[neck_i] = neck_base
            restored = True
        if t > t_free and sim.tilt_deg() < 35:
            balancing = True
        if balancing:
            tgt = bal.adjust(tgt)
        sim.set_targets(tgt)
        sim.step()
        if record is not None:
            record(t)
        for n, v in sim.torques().items():
            if n[:2] not in ("L_", "R_") and sim.limit.get(n, 0.0) > 1e-6 and not n.endswith(("_quad", "_ham")):
                peak[n] = max(peak.get(n, 0.0), abs(v) / sim.limit[n])
        if k % int(0.25 / dt) == 0:
            hist.append((round(t, 2), round(float(d.qpos[2]) * 100, 1), round(sim.tilt_deg(), 0)))
        if verbose and ki < len(ks) and t >= ks[ki][0] - dt / 2:
            load = {"手": 0.0, "脚": 0.0, "其它": 0.0}
            f6 = np.zeros(6)
            for c in range(d.ncon):
                con = d.contact[c]
                b1, b2 = body_of(con.geom1), body_of(con.geom2)
                other = b2 if b1 == "world" else b1 if b2 == "world" else None
                if other is None:
                    continue
                mujoco.mj_contactForce(m, d, c, f6)
                key = "手" if other[:2] in ("L_", "R_") else "脚" if other.endswith("_foot") else "其它"
                load[key] += f6[0]
                if key == "其它":
                    load.setdefault("碰地的", set()).add(other)
            com = d.subtree_com[0]
            kx = [d.xpos[m.body(f"{s}_shin").id] for s in "lr"]
            fx = [d.xpos[m.body(f"{s}_foot").id] for s in "lr"]
            hx = [d.xpos[m.body(f"{s}_palm").id] for s in "LR"]
            print(f"{ks[ki][0]:5.2f}s {ks[ki][2]:<10} 骨盆高 {d.qpos[2]*100:5.1f} cm 倾斜 {sim.tilt_deg():4.0f}°  "
                  f"重心 x {com[0]*100:6.1f} 骨盆 x {d.qpos[0]*100:5.1f} 左膝 x {kx[0][0]*100:5.1f} z {kx[0][2]*100:4.1f} 右膝 x {kx[1][0]*100:5.1f} z {kx[1][2]*100:4.1f}  左脚 x {fx[0][0]*100:6.1f} z {fx[0][2]*100:4.1f}  右脚 x {fx[1][0]*100:6.1f} z {fx[1][2]*100:4.1f}  "
                  f"手 x {hx[0][0]*100:6.1f} z {hx[0][2]*100:4.1f}  受力 手 {load['手']:4.1f} 脚 {load['脚']:4.1f} 其它 {load['其它']:4.1f} {sorted(load.get('碰地的', []))}", flush=True)
            ki += 1
    if trim is not None:
        trim.enabled = True
    m.actuator_forcerange[neck_i] = neck_base
    z, tilt = float(d.qpos[2]), sim.tilt_deg()
    ok = z > 0.20 and tilt < 20
    top = sorted(peak.items(), key=lambda kv: -kv[1])[:5]
    if verbose:
        print(f"结束：骨盆高 {z*100:.1f} cm，倾斜 {tilt:.0f}°，{'站起来了' if ok else '没站起来'}；最吃力的关节（占上限）：" +
              "，".join(f"{n} {v*100:.0f}%" for n, v in top))
    return dict(ok=bool(ok), z_cm=z * 100, tilt=tilt, hist=hist, peak=dict(top))


# ---------------- 第三版身体上已经验证的后半段：蹲着手撑地 → 松手 → 站起 ----------------
# 姿势来自 flex_study 的扫描（第三版身体）：髋 132°、膝 143°、踝勾 34°、腰 57°，手掌在脚掌中心前 4 cm
SQUAT = dict(hp=-2.3, kn=2.5, an=-0.6, wp=1.0, sp=-1.2, el=-0.4)


def squat_pose(sp=None, el=None, hp=None, kn=None, an=None, wp=None):
    c = dict(SQUAT)
    for k, v in dict(sp=sp, el=el, hp=hp, kn=kn, an=an, wp=wp).items():
        if v is not None:
            c[k] = v
    p = P((c["sp"], -0.25, 0.0, c["el"]), (c["hp"], c["kn"], c["an"]), wp=c["wp"], neck=-0.3)
    p["neck_low"] = -0.3
    return p


def squat_to_stand(verbose=True, record=None):
    """从"蹲着、手掌撑地"开始：手把重量交给腿，然后靠大腿站起来。育苗人：手臂撑起身体之后，把重心交给别的部分。"""
    import flex_study as F
    sim = get_sim()
    m = sim.m
    B = F.Body(mujoco.MjModel.from_xml_path(fb.OUT))

    def init(sim):
        B.flat(SQUAT["hp"], SQUAT["kn"], SQUAT["an"], SQUAT["wp"], 0.0, SQUAT["sp"], SQUAT["el"])
        d = sim.d
        d.qpos[:] = B.d.qpos
        fg = m.geom("l_foot").id
        d.qpos[2] -= B.d.geom_xpos[fg][2] - m.geom_size[fg][2] - 0.0005
        d.qvel[:] = 0
    S0 = squat_pose()
    mid = stand_pose(1.6)
    mid["waist_pitch"] = 0.3
    ks = [(0.0, S0, "蹲着手撑地"), (0.8, S0, "蹲着手撑地"),
          (1.8, squat_pose(sp=-0.6, el=-0.3, hp=-2.1, kn=2.3, an=-0.75, wp=0.7), "松手，重心交给腿"),
          (3.0, mid, "半蹲"), (5.0, stand_pose(), "站起")]
    return simulate(ks, hold=1.5, init=init, relax_neck=False, t_free=0.8, verbose=verbose, record=record)


# ---------------- 前半段：从趴着到"蹲着手撑地"，手一步步收回来（育苗人 2026-11-10 的建议） ----------------
# 弓腰收腿 → 手举到头两侧 → 两手撑起（头离地，手和脚尖撑着）→ 两只手轮流抬起、往脚的方向挪 → 蹲着手撑地 → 松手 → 站起
WALK0 = dict(sp0=-2.4, el_push=-0.2, N=3, lift=-1.3, Tm=0.5, Tp=0.4, sp_end=-1.2, wp_mid=1.0, an_mid=-1.0, hp_mid=-2.3, both=0, Ts=1.0)
WALK_RANGES = dict(sp0=(-2.8, -1.8), el_push=(-0.6, 0.0), N=(2, 5), lift=(-2.2, -0.6), Tm=(0.25, 0.8), Tp=(0.2, 0.7), sp_end=(-1.6, -0.8),
                   wp_mid=(0.3, 1.0), an_mid=(-1.0, -0.5), hp_mid=(-2.3, -1.8), both=(0, 1), Ts=(0.6, 1.6))


def hand_walk_keyframes(V):
    def pose(spl, ell, spr, elr, leg=(-2.3, 2.5, -1.0), wp=1.0, neck=-0.3):
        p = P((spl, -0.25, 0.0, ell), leg, wp=wp, neck=neck, arm_r=(spr, -0.25, 0.0, elr))
        p["neck_low"] = neck
        return p
    lie = pose(0.0, -0.2, 0.0, -0.2, leg=(0.0, 0.05, 0.0), wp=0.0, neck=0.0)
    ks = [(0.0, lie, "趴着"), (1.0, lie, "趴着")]
    t = 1.0
    t += 1.6; ks.append((t, pose(-0.5, -2.3, -0.5, -2.3), "弓腰收腿"))
    t += 1.0; ks.append((t, pose(V["sp0"], -2.3, V["sp0"], -2.3), "手举到头两侧"))
    t += 1.2; ks.append((t, pose(V["sp0"], V["el_push"], V["sp0"], V["el_push"]), "两手撑起"))
    N = int(round(V["N"]))
    d = (V["sp_end"] - V["sp0"]) / N
    spl = spr = V["sp0"]
    e = V["el_push"]
    for i in range(N):
        a = (i + 1) / N                        # 手一步步收回来的同时，腰、髋、踝慢慢走向中间的姿势
        leg = (-2.3 + (V["hp_mid"] + 2.3) * a, 2.5, -1.0 + (V["an_mid"] + 1.0) * a)
        wp = 1.0 + (V["wp_mid"] - 1.0) * a
        if V["both"] >= 0.5:                   # 两只手一起挪（对比用）
            t += V["Tm"]; ks.append((t, pose(spl + d, V["lift"], spr + d, V["lift"], leg, wp), f"两手抬起挪 {i+1}"))
            t += V["Tp"]; ks.append((t, pose(spl + d, e, spr + d, e, leg, wp), f"两手撑下 {i+1}"))
            spl = spr = spl + d
        else:                                  # 轮流：一只手撑着，另一只抬起、挪近、撑下
            t += V["Tm"]; ks.append((t, pose(spl + d, V["lift"], spr, e, leg, wp), f"左手抬起挪 {i+1}"))
            t += V["Tp"]; ks.append((t, pose(spl + d, e, spr, e, leg, wp), f"左手撑下 {i+1}"))
            spl += d
            t += V["Tm"]; ks.append((t, pose(spl, e, spr + d, V["lift"], leg, wp), f"右手抬起挪 {i+1}"))
            t += V["Tp"]; ks.append((t, pose(spl, e, spr + d, e, leg, wp), f"右手撑下 {i+1}"))
            spr += d
    S = squat_pose()
    t += V["Ts"]; ks.append((t, S, "蹲着手撑地"))
    t += 0.8; ks.append((t, S, "蹲着手撑地"))
    t_free = t
    t += 1.0; ks.append((t, squat_pose(sp=-0.6, el=-0.3, hp=-2.1, kn=2.3, an=-0.75, wp=0.7), "松手，重心交给腿"))
    mid = stand_pose(1.6)
    mid["waist_pitch"] = 0.3
    t += 1.2; ks.append((t, mid, "半蹲"))
    t += 2.0; ks.append((t, stand_pose(), "站起"))
    return ks, t_free


def hand_walk(V, verbose=False, record=None):
    ks, t_free = hand_walk_keyframes(V)
    r = simulate(ks, hold=1.5, relax_neck=False, t_free=t_free, verbose=verbose, record=record)
    prog = max((min(z, 26.0) / 26.0 + math.cos(math.radians(min(tl, 180))) for tt, z, tl in r["hist"] if tt > 4.8), default=0.0)
    final = min(r["z_cm"], 26.0) / 26.0 + math.cos(math.radians(min(r["tilt"], 180)))
    r["score"] = 0.5 * prog + final + (2.0 if r["ok"] else 0.0)
    return r


def _eval(V):
    r = hand_walk(V)
    return r["score"], V, dict(ok=r["ok"], z_cm=round(r["z_cm"], 1), tilt=round(r["tilt"]))


def search(n0=32, rounds=5):
    from multiprocessing import Pool
    rng = np.random.default_rng(3)

    def sample(center=None, width=1.0):
        V = {}
        for k, (lo, hi) in WALK_RANGES.items():
            V[k] = float(rng.uniform(lo, hi)) if center is None else float(np.clip(center[k] + rng.normal(0, 0.18 * width) * (hi - lo), lo, hi))
        return V
    with Pool(4) as pool:
        res = pool.map(_eval, [dict(WALK0)] + [sample() for _ in range(n0 - 1)])
        res.sort(key=lambda x: -x[0])
        best = res[0]
        print("随机搜索最好的 3 个：", [(round(s, 2), r) for s, _, r in res[:3]], flush=True)
        for k in range(rounds):
            more = pool.map(_eval, [sample(best[1], 1.0 / (k + 1)) for _ in range(16)])
            more.sort(key=lambda x: -x[0])
            if more[0][0] > best[0]:
                best = more[0]
            print(f"细化第 {k+1} 轮：得分 {best[0]:.2f}，{best[2]}", flush=True)
            json.dump(dict(V=best[1], score=best[0], result=best[2]), open(os.path.join(HERE, "getup_hands_walk_best.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            if best[2]["ok"] and k >= 1:
                break
    return best


if __name__ == "__main__":
    if "--search" in sys.argv:
        b = search()
        print("最好的参数：", {k: round(v, 2) for k, v in b[1].items()}, b[2])
        sys.exit(0)
    r = squat_to_stand(verbose="--quiet" not in sys.argv)
    json.dump(dict(segment="蹲着手撑地 → 站起", ok=r["ok"], z_cm=round(r["z_cm"], 1), tilt=round(r["tilt"], 1), hist=r["hist"], squat=SQUAT),
              open(os.path.join(HERE, "getup_hands_result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
