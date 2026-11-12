"""
稳不稳的检验：把策略放回完整身体，每种地面摩擦各跑几次，每次的起始姿势都偏一点（每个关节 ±0.05 弧度），
中途还会被推一下（胸口，3 N，持续 0.2 秒，方向和时刻随机）。eval_full.py 每种地面只跑一次、没有扰动。

用法（用项目的 mujoco 3.17，不需要显卡）：
    python eval_robust.py getup_policy.npz [--trials=4] [--nopush]      # 写 eval_robust_result.json

可选（2026-11-11 加，不给时和原来完全一样）：
    --neck=turn       起身的头 4.5 秒，头转向左边并低头（结果文件名加 _neckturn）。
    --head=1.0        头（geom "head"，唯一会碰地的头部几何体）和地面之间单独用这个摩擦。
                      做法：给头设 priority=1，MuJoCo 对 priority 高的一方直接用它自己的摩擦，不再"两面取大"。
                      结果写 eval_robust_head<值>.json，并多记 2.5 秒时的倾斜（头顶着地往前拱那一步）。
    --frics=0.6,1.8   换一组地面摩擦（默认 0.6, 0.8, 1.0, 1.2, 1.5）
    --procs=10        同时跑几个

可选（2026-11-12 加，"头偏向一边"练进起身，不给时和原来完全一样）：
    --ref=turn        用转头的参考动作 ref_getup_turnL/R.json（make_ref_turn.py），左右都跑；--side=L 或 R 只跑一边
    --push=5          推的力（牛顿，默认 3）
    --vary            起始姿势多样：脖子随机（左右转 ±0.6、上下 ±0.3、歪头 ±0.2 弧度），身体绕长轴随机歪 ±0.08 弧度
    --face            记头着地的位置：压力超过 0.5 N 的时间里，着地点离脸正前方 < 45° 的比例（和 ../head_contact.py 的"脸"一样）。
                      用了 --ref=turn 时自动打开
    --out=名字.json    结果文件名
    --mirror          转右时不用 ref_getup_turnR.json，而是把"转左"整条左右镜像：参考动作镜像（l/r 对调、横滚/偏航取反），
                      策略看到的观测也镜像、给出的动作再镜像回来（策略以为自己在转左）。见 mirror_lr.py
    --trace           每 0.1 秒记一行（脖子目标/实际/出力、身体横滚、头着地点），写进结果
"""
import json
import math
import os
import sys
from multiprocessing import Pool

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
ACT_SCALE, SLEW = 0.3, 0.2
FRICTIONS = (0.6, 0.8, 1.0, 1.2, 1.5)
NECK = dict(neck_yaw=1.3, neck_pitch=0.4, neck_low=0.3) if "--neck=turn" in sys.argv else None      # 见 ../head_contact.py
NECK_VARY = dict(neck_yaw=0.6, neck_pitch=0.3, neck_low=0.3, neck_roll=0.2)                         # --vary：脖子起始角随机的范围（±弧度）
ROLL_VARY = 0.08


def arg(name, default):
    return next((type(default)(a.split("=", 1)[1]) for a in sys.argv if a.startswith(f"--{name}=")), default)


def run(args):
    path, friction, seed, push_on = args[:4]
    head = args[4] if len(args) > 4 else None
    opt = args[5] if len(args) > 5 else {}
    side = opt.get("side")
    push_n = opt.get("push", 3.0)
    import getup_hands as G
    from policy_np import Policy
    pol = Policy(path)
    mirror = bool(opt.get("mirror")) and side == "R"
    ref = json.load(open(os.path.join(HERE, f"ref_getup_turn{'L' if mirror else side}.json" if side else "ref_getup.json"), encoding="utf-8"))
    if mirror:                                # 转右 = 把转左整条镜像；策略在镜像的世界里转左
        import mirror_lr as MR
        P = MR.Mirror(ref["names"])
        ref["q"] = [list(P.q(np.array(r))) for r in ref["q"]]
        pol0 = pol
        pol = lambda o: P.q(pol0(P.obs(o)))
    rng = np.random.default_rng(seed)
    sim = G.get_sim()
    m, d = sim.m, sim.d
    m.geom_friction[:, 0] = friction          # 所有接触面一起改（MuJoCo 取两个面里较大的摩擦）
    if head is not None:                      # 头单独指定：priority 高的一方用自己的摩擦
        hg = m.geom("head").id
        m.geom_priority[hg] = 1
        m.geom_friction[hg, 0] = head
    tilt25 = None
    names = ref["names"]
    q_ref = np.array(ref["q"])
    T = len(q_ref)
    jid = [m.joint(n).id for n in names]
    qa = np.array([m.jnt_qposadr[j] for j in jid])
    va = np.array([m.jnt_dofadr[j] for j in jid])
    lo, hi = m.jnt_range[jid, 0], m.jnt_range[jid, 1]
    pel, chest = m.body("pelvis").id, m.body("chest").id
    sim.reset()
    d.qpos[0:3] = [0, 0, 0.06]
    d.qpos[3:7] = [math.cos(math.pi / 4), 0, math.sin(math.pi / 4), 0]
    d.qpos[qa] = np.clip(d.qpos[qa] + rng.uniform(-0.05, 0.05, len(qa)), lo, hi)
    if opt.get("vary"):                       # 另用一个随机数，不改变原来那几个随机数的顺序
        r2 = np.random.default_rng(10_000 + seed)
        for n, w in NECK_VARY.items():
            i = names.index(n)
            d.qpos[qa[i]] = np.clip(d.qpos[qa[i]] + r2.uniform(-w, w), lo[i], hi[i])
        roll = r2.uniform(-ROLL_VARY, ROLL_VARY)            # 绕世界 x 轴（趴着时身体的长轴，头朝 +x）
        qr = np.array([math.cos(roll / 2), math.sin(roll / 2), 0, 0])
        q0 = d.qpos[3:7].copy()
        mujoco.mju_mulQuat(d.qpos[3:7], qr, q0)
        d.qpos[2] += 0.01
    mujoco.mj_forward(m, d)
    if getattr(sim, "trim", None) is not None:
        sim.trim.enabled = False
    n_sub = int(round(ref["dt"] / m.opt.timestep))
    last, target = np.zeros(len(names)), np.zeros(len(names))
    wi = names.index("waist_pitch")
    t_push = int(rng.uniform(4.0, 16.0) / ref["dt"]) if push_on else -1
    ang = rng.uniform(0, 2 * math.pi)
    stood_at, fell_at = None, None
    face = opt.get("face")
    if face:
        hg_, fl = m.geom("head").id, m.geom("floor").id
        hb = m.geom_bodyid[hg_]
        f6 = np.zeros(6)
        n_on, n_face = 0, 0
        face_bins = np.zeros(40)                 # 脸着地的时间按 0.5 秒分段（秒）
    trace = [] if opt.get("trace") else None
    if trace is not None:
        ny = names.index("neck_yaw")
        ny_act = next(i for i in range(m.nu) if m.actuator_trnid[i, 0] == jid[ny])
        hg_t, fl_t = m.geom("head").id, m.geom("floor").id
        f6t = np.zeros(6)
    for t in range(T + 25):
        r = q_ref[min(t, T - 1)]
        ph = t / T
        obs = np.concatenate([-d.xmat[pel].reshape(3, 3)[2, :], d.qvel[3:6] * 0.25, d.qpos[qa], d.qvel[va] * 0.1, last, r,
                              [ph, math.sin(2 * math.pi * ph), math.cos(2 * math.pi * ph)]])
        a = pol(obs)
        want = np.clip(r + ACT_SCALE * a, lo, hi)
        target = target + np.clip(want - target, -SLEW, SLEW)
        pose = {n: float(v) for n, v in zip(names, target)}
        pose["waist_pitch"] = 2.0 * float(target[wi])
        if NECK and t * ref["dt"] < 4.5:        # --neck=turn：起身的头 4.5 秒，头转向左边并低头（让侧面的头发着地，不让脸着地）
            pose.update(NECK)
        pushing = 0 <= t - t_push < 10
        for _ in range(n_sub):
            sim.set_targets(pose)
            d.xfrc_applied[chest, 0:2] = [push_n * math.cos(ang), push_n * math.sin(ang)] if pushing else [0.0, 0.0]
            sim.step()
            if face:
                for c in range(d.ncon):
                    con = d.contact[c]
                    if {int(con.geom1), int(con.geom2)} == {hg_, fl}:
                        mujoco.mj_contactForce(m, d, c, f6)
                        if f6[0] > 0.5:
                            u = d.xmat[hb].reshape(3, 3).T @ (con.pos - d.geom_xpos[hg_])
                            n_on += 1
                            if u[0] / max(np.linalg.norm(u), 1e-9) > math.cos(math.radians(45)):
                                n_face += 1
                                face_bins[min(int(d.time / 0.5), 39)] += m.opt.timestep
        last = a
        if trace is not None and t % 5 == 4 and t < 200:
            R = d.xmat[pel].reshape(3, 3)
            hb_t = m.geom_bodyid[hg_t]
            cang, cf = None, 0.0
            for c in range(d.ncon):
                con = d.contact[c]
                if {int(con.geom1), int(con.geom2)} == {hg_t, fl_t}:
                    mujoco.mj_contactForce(m, d, c, f6t)
                    if f6t[0] > cf:
                        u = d.xmat[hb_t].reshape(3, 3).T @ (con.pos - d.geom_xpos[hg_t])
                        cf, cang = float(f6t[0]), round(math.degrees(math.acos(np.clip(u[0] / max(np.linalg.norm(u), 1e-9), -1, 1))))
            trace.append(dict(t=round((t + 1) * ref["dt"], 1), yaw_want=round(float(target[ny]), 2), yaw=round(float(d.qpos[qa[ny]]), 2),
                              yaw_force=round(float(d.actuator_force[ny_act]), 3),
                              roll=round(math.degrees(math.asin(np.clip(R[2, 1], -1, 1)))),     # 骨盆左右轴的上下分量：身体绕长轴歪了多少（+ = 左边抬起）
                              z=round(float(d.qpos[2]) * 100, 1), head_N=round(cf, 2), head_ang=cang))   # head_ang：着地点离脸正前方的角度
        if t > 150 and fell_at is None and d.qpos[2] < 0.08:
            fell_at = round((t + 1) * ref["dt"], 1)
        if t + 1 == 125:
            tilt25 = round(sim.tilt_deg())
    d.xfrc_applied[chest, 0:2] = 0
    if getattr(sim, "trim", None) is not None:
        sim.trim.enabled = True
    z, tilt = float(d.qpos[2]), sim.tilt_deg()
    out = dict(friction=friction, seed=seed, ok=bool(z > 0.20 and tilt < 20), z_cm=round(z * 100, 1), tilt=round(tilt),
               fell_at=fell_at, push_at=round(t_push * ref["dt"], 1) if push_on else None)
    if head is not None:
        out.update(head=head, tilt_2_5s=tilt25)
    if side:
        out.update(side=side)
    if mirror:
        out.update(mirror=True)
    if trace is not None:
        out.update(trace=trace)
    if face:
        out.update(head_contact_s=round(n_on * m.opt.timestep, 2), face_s=round(n_face * m.opt.timestep, 3),
                   face_share=round(n_face / max(n_on, 1), 3),
                   face_when={f"{i * 0.5:g}": round(float(v), 2) for i, v in enumerate(face_bins) if v > 0})
    return out


def main():
    path = os.path.abspath(sys.argv[1])
    trials = arg("trials", 4)
    push_on = "--nopush" not in sys.argv
    head = next((float(a.split("=")[1]) for a in sys.argv if a.startswith("--head=")), None)
    frics = next((tuple(float(x) for x in a.split("=")[1].split(",")) for a in sys.argv if a.startswith("--frics=")), FRICTIONS)
    procs = arg("procs", 10)
    sides = list(arg("side", "LR")) if arg("ref", "") == "turn" else [None]
    opt = dict(push=arg("push", 3.0), vary="--vary" in sys.argv, face="--face" in sys.argv or arg("ref", "") == "turn",
               mirror="--mirror" in sys.argv, trace="--trace" in sys.argv)
    new = any(a.split("=")[0] in ("--ref", "--side", "--push", "--vary", "--face", "--out", "--mirror", "--trace") for a in sys.argv)
    if not new:
        jobs = [(path, f, s, push_on) + ((head,) if head is not None else ()) for f in frics for s in range(trials)]
    else:
        jobs = [(path, f, s, push_on, head, dict(opt, side=sd)) for sd in sides for f in frics for s in range(trials)]
    with Pool(min(procs, len(jobs))) as pool:
        res = pool.map(run, jobs)
    total = 0
    for sd in sides:
        for f in frics:
            rs = [r for r in res if r["friction"] == f and r.get("side") == sd]
            ok = sum(r["ok"] for r in rs)
            total += ok
            fails = "；".join(f"第 {r['seed']} 次 {r['fell_at']} 秒时倒（{r['push_at']} 秒被推"
                             + (f"，2.5 秒倾斜 {r['tilt_2_5s']}°" if head is not None else "") + "）" for r in rs if not r["ok"])
            fs = ""
            if rs and "face_share" in rs[0]:
                on = sum(r["head_contact_s"] for r in rs)
                fs = f"  头着地 {on / len(rs):.2f} 秒/次，落在脸上 {sum(r['face_s'] for r in rs) / max(on, 1e-9):.1%}（最多的一次 {max(r['face_share'] for r in rs):.0%}）"
            print((f"转向{'左' if sd == 'L' else '右'}、" if sd else "") + f"地面摩擦 {f}" + (f"、头 {head}" if head is not None else "")
                  + f"：{ok}/{len(rs)} 次站起来" + fs + (f"  失败：{fails}" if fails else ""), flush=True)
    print(f"合计 {total}/{len(res)}（起始姿势随机偏 ±0.05 弧度" + ("，脖子和身体也随机偏" if opt["vary"] else "")
          + (f"，中途被推一次 {opt['push']:g} N" if push_on else "，不推") + "）")
    if new:
        out = arg("out", "") or ("eval_robust" + (f"_head{head}" if head is not None else "") + ("_turn" if sides != [None] else "")
                                 + f"_push{opt['push']:g}" + ("_vary" if opt["vary"] else "") + ".json")
        json.dump(dict(policy=os.path.basename(path), push=push_on, push_N=opt["push"], head=head, vary=opt["vary"], runs=res),
                  open(os.path.join(HERE, out), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        return
    out = ("eval_robust_result" if head is None else f"eval_robust_head{head}") + ("_neckturn" if NECK else "") + ".json"
    json.dump(dict(policy=os.path.basename(path), push=push_on, head=head, runs=res) if head is not None else
              dict(policy=os.path.basename(path), push=push_on, runs=res),
              open(os.path.join(HERE, out), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
