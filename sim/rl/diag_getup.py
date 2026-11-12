"""
起身诊断：同一份策略在完整身体和简化身体上跑同一段，每 0.1 秒记一次，看两边从哪一刻开始不一样。

用法（完整身体一律用 mujoco 3.17：~/venv-sys/bin/python）：
    python diag_getup.py full  [策略.npz] --fric=1.5 [--from=2.5 --to=6.0]
    python diag_getup.py simple [策略.npz] --fric=1.5 [--k=0.2 --rest=0.15 --wk=1.0] [--from= --to=]
    python diag_getup.py names                      # 打印参考动作里的关节名
    两种都可以加 --ref=turnL（或 turnR，转头的参考，make_ref_turn.py）、--head=0.7（头和地面之间单独的摩擦）

每行记：骨盆高、倾斜；肩俯仰和肘的实际角/目标角（左右）；手臂关节里用到力气上限的比例；
各部位压在地上的法向力（手指、拇指、手掌、前臂、上臂、头颈、躯干、大腿/膝、小腿、脚）；
手掌和手指接触里"切向力 / (摩擦系数 × 法向力)"的最大值（接近 1 = 在滑，明显小于 1 = 粘住）；手掌在地面上的水平速度。
完整身体的跑法和 eval_full.py 的 run 一样（不加扰动）；简化身体的跑法和 getup_env.py 一样（从趴着开始，CPU 版 MuJoCo）。
结果另存 diag_<full|simple>_<摩擦>[_k..].json。
"""
import json
import math
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from policy_np import Policy     # noqa: E402

ACT_SCALE, SLEW = 0.3, 0.2
SHOW = ("l_shoulder_pitch", "l_elbow", "r_shoulder_pitch", "r_elbow")
ARM = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")
PARTS = ("fing", "thumb", "palm", "fore", "upper", "head", "torso", "thigh", "shin", "foot")


def arg(name, default):
    for a in sys.argv[1:]:
        if a.startswith(f"--{name}="):
            return type(default)(a.split("=", 1)[1])
    return default


def part(name):
    n = name.lower()
    if "thumb" in n:
        return "thumb"
    if n in ("l_hand", "r_hand"):
        return "palm"
    if n in ("l_fingers", "r_fingers") or name[:2] in ("L_", "R_"):
        return "fing"
    for key, p in (("forearm", "fore"), ("upper_arm", "upper"), ("head", "head"), ("neck", "head"), ("thigh", "thigh"),
                   ("knee", "thigh"), ("shin", "shin"), ("foot", "foot"), ("ankle", "foot"), ("toe", "foot")):
        if key in n:
            return p
    return "torso"


class Probe:
    def __init__(self, m, names, limits):
        self.m = m
        self.names = names
        self.body_part = [part(m.body(b).name) for b in range(m.nbody)]
        self.floor = {g for g in range(m.ngeom) if m.geom_bodyid[g] == 0}
        jid = [m.joint(n).id for n in names]
        self.qa = np.array([m.jnt_qposadr[j] for j in jid])
        self.va = np.array([m.jnt_dofadr[j] for j in jid])
        self.lim = np.array(limits)
        self.arm_i = [i for i, n in enumerate(names) if n[2:] in ARM]
        self.pal = [m.body(b).id for b in ("l_hand", "r_hand")] if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "l_hand") >= 0 \
            else [m.body(b).id for b in ("L_hand", "R_hand")]

    def row(self, t, d, target):
        m = self.m
        w, x, y, _ = d.qpos[3:7]
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (x * x + y * y)))))
        force = {p: 0.0 for p in PARTS}
        slip = {"palm": 0.0, "fing": 0.0}
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            if (g1 in self.floor) == (g2 in self.floor):
                continue
            g = g2 if g1 in self.floor else g1
            p = self.body_part[m.geom_bodyid[g]]
            mujoco.mj_contactForce(m, d, i, f6)
            fn = max(f6[0], 0.0)
            force[p] += fn
            if p in ("palm", "fing", "thumb") and fn > 0.2:
                k = "palm" if p == "palm" else "fing"
                slip[k] = max(slip[k], float(np.hypot(f6[1], f6[2]) / (c.friction[0] * fn + 1e-9)))
        q = d.qpos[self.qa]
        tau = np.abs(d.qfrc_actuator[self.va]) / self.lim
        v = np.zeros(6)
        hv = []
        for b in self.pal:
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b, v, 0)
            hv.append(float(np.hypot(v[3], v[4])))
        out = dict(t=round(t, 2), z=round(float(d.qpos[2]) * 100, 1), tilt=round(tilt, 1))
        for n in SHOW:
            i = self.names.index(n)
            out[n] = [round(float(q[i]), 2), round(float(target[i]), 2)]
        ai = int(np.argmax(tau[self.arm_i]))
        out["arm_sat"] = [self.names[self.arm_i[ai]], round(float(tau[self.arm_i][ai]), 2)]
        allm = int(np.argmax(tau))
        out["max_sat"] = [self.names[allm], round(float(tau[allm]), 2)]
        out["N"] = {p: round(v_, 1) for p, v_ in force.items() if v_ > 0.05}
        out["slip"] = {k: round(v_, 2) for k, v_ in slip.items() if v_ > 0}
        out["hand_v"] = [round(a * 100, 1) for a in hv]       # cm/s
        return out


def bias_at(names, ts):
    """可行性试验：--bias=l_elbow+r_elbow:-1:1.5:2.7 在 1.5–2.7 秒给这些关节的动作加 -1（即目标偏 -0.3 弧度）"""
    out = np.zeros(len(names))
    b = arg("bias", "")
    if b:
        js, v, t0, t1 = b.split(":")
        if float(t0) <= ts <= float(t1):
            for j in js.split("+"):
                out[names.index(j)] = float(v)
    return out


def obs_of(d, pel, qa, va, last, r, t, T):
    ph = t / T
    return np.concatenate([-d.xmat[pel].reshape(3, 3)[2, :], d.qvel[3:6] * 0.25, d.qpos[qa], d.qvel[va] * 0.1, last, r,
                           [ph, math.sin(2 * math.pi * ph), math.cos(2 * math.pi * ph)]])


def run_full(pol, ref, friction, t_lo, t_hi):
    import getup_hands as G
    import tendon_knee as TK
    sim = G.get_sim()
    m, d = sim.m, sim.d
    if arg("cone", ""):
        m.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC if arg("cone", "") == "elliptic" else mujoco.mjtCone.mjCONE_PYRAMIDAL
    m.geom_friction[:, 0] = friction
    if arg("head", 0.0):                    # --head=0.7：头和地面之间单独用这个摩擦（同 eval_robust.py --head=）
        hg = m.geom("head").id
        m.geom_priority[hg] = 1
        m.geom_friction[hg, 0] = arg("head", 0.0)
    # 反事实试验：--part=thumb,fing:1.0 只把这些部位的摩擦改成 1.0（地面也得低于它才有效，MuJoCo 取两者较大的）
    pf = arg("part", "")
    if pf:
        parts, val = pf.split(":")
        parts = parts.split(",")
        for g in range(m.ngeom):
            if part(m.body(m.geom_bodyid[g]).name) in parts:
                m.geom_friction[g, 0] = float(val)
        for g in range(m.ngeom):
            if m.geom_bodyid[g] == 0:
                m.geom_friction[g, 0] = min(friction, float(val))
    names = ref["names"]
    q_ref = np.array(ref["q"])
    T = len(q_ref)
    jid = [m.joint(n).id for n in names]
    lo, hi = m.jnt_range[jid, 0], m.jnt_range[jid, 1]
    lim = []
    for j, n in zip(jid, names):
        a = [i for i in range(m.nu) if m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT and m.actuator_trnid[i, 0] == j]
        lim.append(TK.KNEE_LIMIT if n.endswith("knee") else float(m.actuator_forcerange[a[0], 1]))
    pr = Probe(m, names, lim)
    pel = m.body("pelvis").id
    sim.reset()
    d.qpos[0:3] = [0, 0, 0.06]
    d.qpos[3:7] = [math.cos(math.pi / 4), 0, math.sin(math.pi / 4), 0]
    mujoco.mj_forward(m, d)
    if getattr(sim, "trim", None) is not None:
        sim.trim.enabled = False
    n_sub = int(round(ref["dt"] / m.opt.timestep))
    last, target = np.zeros(len(names)), np.zeros(len(names))
    wi = names.index("waist_pitch")
    rows = []
    for t in range(T + 25):
        r = q_ref[min(t, T - 1)]
        a = pol(obs_of(d, pel, pr.qa, pr.va, last, r, t, T))
        a = a + bias_at(names, t * ref["dt"])
        a = np.clip(a, -1.0, 1.0)          # 加了偏置也不超过策略本来能改的 ±0.3 弧度
        want = np.clip(r + ACT_SCALE * a, lo, hi)
        target = target + np.clip(want - target, -SLEW, SLEW)
        pose = {n: float(v) for n, v in zip(names, target)}
        pose["waist_pitch"] = 2.0 * float(target[wi])
        for _ in range(n_sub):
            sim.set_targets(pose)
            sim.step()
        last = a
        ts = (t + 1) * ref["dt"]
        if t_lo - 1e-9 <= ts <= t_hi + 1e-9 and (t + 1) % 5 == 0:
            rows.append(pr.row(ts, d, target))
    if getattr(sim, "trim", None) is not None:
        sim.trim.enabled = True
    return rows, dict(z_cm=round(float(d.qpos[2]) * 100, 1), tilt=round(sim.tilt_deg()))


def run_simple(pol, ref, friction, t_lo, t_hi, k=0.2, rest=0.25, wk=1.0):
    m = mujoco.MjModel.from_xml_path(os.path.join(HERE, "huaran_rl.xml"))
    st = json.load(open(os.path.join(HERE, ref.get("state_file", "ref_state.json")), encoding="utf-8"))
    m.geom_friction[:, 0] *= friction
    if arg("head", 0.0):                    # 和 getup_env.py 的 HEAD_FRIC 一样：地面 0.01，头单独一个值（两面取大）
        m.geom_friction[m.geom("floor").id, 0] = 0.01
        m.geom_friction[m.geom("head").id, 0] = arg("head", 0.0)
    if arg("cone", ""):
        m.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC if arg("cone", "") == "elliptic" else mujoco.mjtCone.mjCONE_PYRAMIDAL
    if arg("dt", 0.0):
        m.opt.timestep = arg("dt", 0.0)
    # 手的反事实：--handsol 让手掌和手指板用完整身体手的接触参数；--nofing 让手指板不碰地
    hand_g = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name in ("l_hand", "r_hand", "l_fingers", "r_fingers")]
    if "--handsol" in sys.argv:
        m.geom_solref[hand_g] = [0.004, 1.0]
        m.geom_solimp[hand_g, :3] = [0.95, 0.99, 0.0005]
    if "--nofing" in sys.argv:
        for g in hand_g:
            if "fingers" in m.body(m.geom_bodyid[g]).name:
                m.geom_contype[g] = m.geom_conaffinity[g] = 0
    if arg("iters", 0):
        m.opt.iterations = arg("iters", 0)
        m.opt.ls_iterations = max(m.opt.ls_iterations, arg("iters", 0))
    for s in "lr":
        fj, wj = m.joint(f"{s}_fingers").id, m.joint(f"{s}_wrist").id
        m.jnt_stiffness[fj] = k
        m.qpos_spring[m.jnt_qposadr[fj]] = rest
        m.jnt_stiffness[wj] *= wk
    d = mujoco.MjData(m)
    names = ref["names"]
    q_ref = np.array(ref["q"])
    T = len(q_ref)
    jid = m.actuator_trnid[:, 0]
    assert [m.actuator(i).name for i in range(m.nu)] == names
    lo, hi = m.jnt_range[jid, 0], m.jnt_range[jid, 1]
    pr = Probe(m, [m.joint(j).name for j in jid], m.actuator_forcerange[:, 1])
    pr.names = names
    pel = m.body("pelvis").id
    d.qpos[:] = m.qpos0
    d.qpos[2] = st["z"][0] + 0.003
    d.qpos[3:7] = st["quat"][0]
    d.qpos[pr.qa] = st["q"][0]
    d.qvel[0:6] = st["vel"][0]
    d.qvel[pr.va] = st["qd"][0]
    mujoco.mj_forward(m, d)
    n_sub = int(round(ref["dt"] / m.opt.timestep))
    last, target = np.zeros(len(names)), q_ref[0].copy()
    rows = []
    for t in range(T + 25):
        r = q_ref[min(t, T - 1)]
        a = pol(obs_of(d, pel, pr.qa, pr.va, last, r, t, T))
        want = np.clip(r + ACT_SCALE * a, lo, hi)
        target = target + np.clip(want - target, -SLEW, SLEW)
        d.ctrl[:] = target
        for _ in range(n_sub):
            mujoco.mj_step(m, d)
        last = a
        ts = (t + 1) * ref["dt"]
        if t_lo - 1e-9 <= ts <= t_hi + 1e-9 and (t + 1) % 5 == 0:
            rows.append(pr.row(ts, d, target))
    w, x, y, _ = d.qpos[3:7]
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (x * x + y * y)))))
    return rows, dict(z_cm=round(float(d.qpos[2]) * 100, 1), tilt=round(tilt))


def show(rows):
    for r in rows:
        sp = " ".join(f"{n[0]}{'肩' if 'sh' in n else '肘'} {r[n][0]:+.2f}/{r[n][1]:+.2f}" for n in SHOW)
        print(f"{r['t']:5.1f}s  骨盆 {r['z']:5.1f}cm 倾斜 {r['tilt']:5.1f}°  {sp}  臂最吃力 {r['arm_sat'][0]} {r['arm_sat'][1]:.2f}"
              f"  全身最吃力 {r['max_sat'][0]} {r['max_sat'][1]:.2f}  地面力 {r['N']}  滑动比 {r['slip']}  手掌速度 {r['hand_v']} cm/s")


def main():
    mode = sys.argv[1]
    rname = arg("ref", "")                  # --ref=turnL / turnR：转头的参考（make_ref_turn.py）
    ref = json.load(open(os.path.join(HERE, f"ref_getup_{rname}.json" if rname else "ref_getup.json"), encoding="utf-8"))
    if rname:
        ref["state_file"] = f"ref_state_{rname}.json"
    if mode == "names":
        print(ref["names"], ref["dt"], ref["t_free"], len(ref["q"]))
        return
    pos = [a for a in sys.argv[2:] if not a.startswith("--")]
    path = pos[0] if pos else os.path.join(HERE, "getup_policy.npz")
    pol = Policy(path)
    f = arg("fric", 1.5)
    lo, hi = arg("from", 2.5), arg("to", 6.0)
    if mode == "full":
        rows, end = run_full(pol, ref, f, lo, hi)
        tag = (f"full_{f}" + (f"_{arg('cone', '')}" if arg("cone", "") else "")) + (f"_{arg('part', '').replace(',', '+').replace(':', '_')}" if arg("part", "") else "")
    else:
        k, rest, wk = arg("k", 0.2), arg("rest", 0.25), arg("wk", 1.0)
        rows, end = run_simple(pol, ref, f, lo, hi, k, rest, wk)
        tag = f"simple_{f}_k{k}_r{rest}_w{wk}" + (f"_{arg('cone', '')}" if arg("cone", "") else "") + ("_handsol" if "--handsol" in sys.argv else "") + ("_nofing" if "--nofing" in sys.argv else "")
    if rname or arg("head", 0.0):
        tag += (f"_{rname}" if rname else "") + (f"_head{arg('head', 0.0)}" if arg("head", 0.0) else "")
    show(rows)
    print(f"结束时：骨盆 {end['z_cm']} cm，倾斜 {end['tilt']}°（{'站起来了' if end['z_cm'] > 20 and end['tilt'] < 20 else '没站起来'}）")
    json.dump(dict(policy=os.path.basename(path), mode=mode, friction=f, end=end, rows=rows),
              open(os.path.join(HERE, f"diag_{tag}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=0)


if __name__ == "__main__":
    main()
