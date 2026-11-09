"""
摔倒后自己爬起来：从趴着（脸朝下）站起来的动作搜索。

用法：
    python getup.py            # 随机搜索 + 细化起身动作的关键帧，打印最好的结果，保存 getup_best.json

做法：起身动作由 4 个关键帧组成（手撑地 → 收腿跪起 → 往后坐到脚上 → 站起），
关键帧里的关节角度和时间是待搜索的参数。用和其他测试相同的整体模型、相同的扭矩上限，
看能不能在 10 秒内回到站立并保持 1.5 秒。这和真实机器人调起身动作的做法一样，只是由程序代劳。
"""
import json
import math
import os
import sys
from multiprocessing import Pool

import mujoco
import numpy as np

import figure_test
import full_build as fb
import full_demo as FD
from figure_test import stand_pose

HERE = os.path.dirname(os.path.abspath(__file__))
RANGES = {
    "t1": (0.6, 1.4), "sp1": (-1.8, -0.9), "el1": (-1.2, 0.0),
    "t2": (0.8, 2.0), "hp2": (-2.2, -1.2), "kn2": (1.6, 2.5), "an2": (-1.0, 0.4), "sp2": (-2.2, -1.0), "wp2": (-0.3, 0.6),
    "t3": (0.8, 2.0), "hp3": (-2.2, -1.4), "kn3": (1.8, 2.5), "an3": (-1.0, -0.2), "sp3": (-3.0, -1.2), "wp3": (-0.3, 0.6), "el3": (-1.5, 0.0),
    "t4": (1.2, 3.0),
}
_SIM = None


def sm(a):
    a = max(0.0, min(1.0, a))
    return 0.5 - 0.5 * math.cos(math.pi * a)


VARIANT = os.environ.get("GETUP_VARIANT", "")   # 诊断用："torque2"（扭矩加倍）、"range"（关节范围加大）、"both"


def apply_variant(sim, v):
    """诊断：放宽某一项限制，看是不是它卡住了起身。"""
    import tendon_knee as tk
    m = sim.m
    if v in ("torque2", "both"):
        for i in range(m.nu):
            n = m.actuator(i).name
            if n[:2] not in ("L_", "R_") and not n.endswith(("_quad", "_ham")):
                m.actuator_forcerange[i] *= 2
                sim.limit[n] = sim.limit.get(n, 0) * 2
        tk.KNEE_LIMIT *= 2
        tk.TMAX *= 2
        for i in range(m.nu):
            if m.actuator(i).name.endswith(("_quad", "_ham")):
                m.actuator_ctrlrange[i] = (-tk.TMAX, 0)
    if v in ("range", "both"):
        for j, lo, hi in (("waist_pitch", -0.3, 1.4), ("l_hip_pitch", -2.6, 0.7), ("r_hip_pitch", -2.6, 0.7),
                          ("l_knee", 0, 2.7), ("r_knee", 0, 2.7), ("l_ankle_pitch", -1.3, 1.0), ("r_ankle_pitch", -1.3, 1.0)):
            m.jnt_range[m.joint(j).id] = (lo, hi)
        RANGES.update(hp2=(-2.6, -1.2), hp3=(-2.6, -1.4), kn2=(1.6, 2.7), kn3=(1.8, 2.7), an3=(-1.3, -0.2), wp2=(-0.3, 1.4), wp3=(-0.3, 1.4))


def get_sim():
    global _SIM
    if _SIM is None:
        figure_test.MODEL = fb.OUT
        _SIM = FD.FullSim()
        if VARIANT:
            apply_variant(_SIM, VARIANT)
    return _SIM


def lying():
    p = stand_pose(0.05)
    for s in "lr":
        p[f"{s}_hip_pitch"] = 0.0
        p[f"{s}_ankle_pitch"] = 0.0
        p[f"{s}_shoulder_pitch"] = 0.0
    return p


def keyframes(x):
    def pose(sp, el, hp, kn, an, wp):
        p = lying()
        for s in "lr":
            p[f"{s}_shoulder_pitch"], p[f"{s}_elbow"] = sp, el
            p[f"{s}_hip_pitch"], p[f"{s}_knee"], p[f"{s}_ankle_pitch"] = hp, kn, an
        p["waist_pitch"] = wp
        return p
    t = 1.0
    ks = [(0.0, lying()), (t, lying())]
    t += x["t1"]; ks.append((t, pose(x["sp1"], x["el1"], 0.0, 0.05, 0.0, 0.0)))
    t += x["t2"]; ks.append((t, pose(x["sp2"], x["el1"], x["hp2"], x["kn2"], x["an2"], x["wp2"])))
    t += x["t3"]; ks.append((t, pose(x["sp3"], x["el3"], x["hp3"], x["kn3"], x["an3"], x["wp3"])))
    t += x["t4"]; ks.append((t, stand_pose()))
    return ks


def run(x, record=None):
    sim = get_sim()
    m, d = sim.m, sim.d
    ks = keyframes(x)

    def target(t):
        for i in range(1, len(ks)):
            t0, p0 = ks[i - 1]
            t1, p1 = ks[i]
            if t <= t1:
                a = sm((t - t0) / (t1 - t0))
                return {k: p0.get(k, 0.0) + (p1.get(k, 0.0) - p0.get(k, 0.0)) * a for k in set(p0) | set(p1)}
        return ks[-1][1]

    sim.reset()
    d.qpos[0:3] = [0, 0, 0.06]
    d.qpos[3:7] = [math.cos(math.pi / 4), 0, math.sin(math.pi / 4), 0]   # 脸朝下趴着
    mujoco.mj_forward(m, d)
    dt = m.opt.timestep
    t_end = ks[-1][0] + 1.5
    hist = []
    for k in range(int(t_end / dt)):
        t = k * dt
        sim.set_targets(target(t))
        sim.step()
        if record is not None:
            record(t)
        if k % int(0.25 / dt) == 0:
            hist.append((round(t, 2), round(float(d.qpos[2]) * 100, 1), round(sim.tilt_deg(), 0)))
    z, tilt = float(d.qpos[2]), sim.tilt_deg()
    ok = z > 0.20 and tilt < 20
    score = min(z, 0.26) / 0.26 + math.cos(math.radians(min(tilt, 180))) + (1.0 if ok else 0.0)
    return dict(score=score, ok=ok, z_cm=z * 100, tilt=tilt, t_total=ks[-1][0] - 1.0, hist=hist)


def sample(rng, center=None, width=1.0):
    x = {}
    for k, (lo, hi) in RANGES.items():
        if center is None:
            x[k] = float(rng.uniform(lo, hi))
        else:
            x[k] = float(np.clip(center[k] + rng.normal(0, 0.15 * width) * (hi - lo), lo, hi))
    return x


def evaluate(x):
    r = run(x)
    return r["score"], x, r


if __name__ == "__main__":
    n0 = int(sys.argv[1]) if len(sys.argv) > 1 else 48
    rng = np.random.default_rng(1)
    with Pool(4) as pool:
        res = pool.map(evaluate, [sample(rng) for _ in range(n0)])
        res.sort(key=lambda t: -t[0])
        print("随机搜索最好的 3 个：", [(round(s, 2), r["ok"], round(r["z_cm"], 1), round(r["tilt"])) for s, _, r in res[:3]], flush=True)
        best = res[0]
        for rnd in range(4):
            cands = [sample(rng, best[1], 1.0 / (rnd + 1)) for _ in range(16)]
            more = pool.map(evaluate, cands)
            more.sort(key=lambda t: -t[0])
            if more[0][0] > best[0]:
                best = more[0]
            print(f"细化第 {rnd+1} 轮：最好 {best[0]:.2f}，站起来了：{best[2]['ok']}，骨盆高 {best[2]['z_cm']:.1f} cm，倾斜 {best[2]['tilt']:.0f}°", flush=True)
    s, x, r = best
    print("最好的动作：", {k: round(v, 2) for k, v in x.items()})
    print("过程（时间, 骨盆高 cm, 倾斜°）：", r["hist"])
    json.dump(dict(x=x, result={k: v for k, v in r.items() if k != "hist"}, hist=r["hist"], variant=VARIANT),
              open(os.path.join(HERE, f"getup_best{'_' + VARIANT if VARIANT else ''}.json"), "w"), ensure_ascii=False, indent=1)
