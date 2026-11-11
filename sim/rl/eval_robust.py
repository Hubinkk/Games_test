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


def run(args):
    path, friction, seed, push_on = args[:4]
    head = args[4] if len(args) > 4 else None
    import getup_hands as G
    from policy_np import Policy
    pol = Policy(path)
    ref = json.load(open(os.path.join(HERE, "ref_getup.json"), encoding="utf-8"))
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
    mujoco.mj_forward(m, d)
    if getattr(sim, "trim", None) is not None:
        sim.trim.enabled = False
    n_sub = int(round(ref["dt"] / m.opt.timestep))
    last, target = np.zeros(len(names)), np.zeros(len(names))
    wi = names.index("waist_pitch")
    t_push = int(rng.uniform(4.0, 16.0) / ref["dt"]) if push_on else -1
    ang = rng.uniform(0, 2 * math.pi)
    stood_at, fell_at = None, None
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
            d.xfrc_applied[chest, 0:2] = [3.0 * math.cos(ang), 3.0 * math.sin(ang)] if pushing else [0.0, 0.0]
            sim.step()
        last = a
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
    return out


def main():
    path = os.path.abspath(sys.argv[1])
    trials = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--trials=")), 4)
    push_on = "--nopush" not in sys.argv
    head = next((float(a.split("=")[1]) for a in sys.argv if a.startswith("--head=")), None)
    frics = next((tuple(float(x) for x in a.split("=")[1].split(",")) for a in sys.argv if a.startswith("--frics=")), FRICTIONS)
    procs = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--procs=")), 10)
    jobs = [(path, f, s, push_on) + ((head,) if head is not None else ()) for f in frics for s in range(trials)]
    with Pool(min(procs, len(jobs))) as pool:
        res = pool.map(run, jobs)
    total = 0
    for f in frics:
        rs = [r for r in res if r["friction"] == f]
        ok = sum(r["ok"] for r in rs)
        total += ok
        fails = "；".join(f"第 {r['seed']} 次 {r['fell_at']} 秒时倒（{r['push_at']} 秒被推"
                         + (f"，2.5 秒倾斜 {r['tilt_2_5s']}°" if head is not None else "") + "）" for r in rs if not r["ok"])
        print(f"地面摩擦 {f}" + (f"、头 {head}" if head is not None else "") + f"：{ok}/{len(rs)} 次站起来" + (f"  失败：{fails}" if fails else ""))
    print(f"合计 {total}/{len(res)}（起始姿势随机偏 ±0.05 弧度" + ("，中途被推一次 3 N" if push_on else "，不推") + "）")
    out = ("eval_robust_result" if head is None else f"eval_robust_head{head}") + ("_neckturn" if NECK else "") + ".json"
    json.dump(dict(policy=os.path.basename(path), push=push_on, head=head, runs=res) if head is not None else
              dict(policy=os.path.basename(path), push=push_on, runs=res),
              open(os.path.join(HERE, out), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
