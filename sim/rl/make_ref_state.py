"""
在完整身体上把成功的那次起身（getup_hands.hand_walk，松手后接手写的平衡控制）完整跑一遍，
每 20 ms 记下身体实际的状态：骨盆高、骨盆感觉到的"哪边是下"、26 个关节实际的角度和速度、骨盆的朝向和速度。
强化学习用它当"身体应该在哪"的参考（ref_getup.json 只有电机的目标角，没有身体的状态），
也用它让练习可以从动作的中间开始。

用法： python make_ref_state.py      # 写 ref_state.json
注意：要用项目的 mujoco 3.17 跑。同一条动作在 mujoco 4.0 上站不起来（2026-11-11 实测：9.2 cm，倾斜 107°）。
"""
import json
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import getup_hands as G          # noqa: E402


def main():
    ref = json.load(open(os.path.join(HERE, "ref_getup.json"), encoding="utf-8"))
    V = json.load(open(os.path.join(HERE, "..", "getup_hands_walk_best.json"), encoding="utf-8"))["V"]
    sim = G.get_sim()
    m, d = sim.m, sim.d
    names = ref["names"]
    jid = [m.joint(n).id for n in names]
    qa = np.array([m.jnt_qposadr[j] for j in jid])
    va = np.array([m.jnt_dofadr[j] for j in jid])
    pel = m.body("pelvis").id
    every = int(round(ref["dt"] / m.opt.timestep))
    rows, k = [], [0]
    r4 = lambda a: [round(float(v), 4) for v in a]

    def record(t):
        k[0] += 1
        if k[0] % every == 0:
            R = d.xmat[pel].reshape(3, 3)
            rows.append(dict(z=round(float(d.qpos[2]), 4), grav=r4(-R[2, :]), q=r4(d.qpos[qa]), quat=r4(d.qpos[3:7]),
                             vel=r4(d.qvel[0:6]), qd=r4(d.qvel[va])))
    r = G.hand_walk(V, verbose=False, record=record)
    T = len(ref["q"])
    rows = (rows + [rows[-1]] * T)[:T]
    out = dict(dt=ref["dt"], names=names, mujoco=mujoco.__version__, ok=r["ok"], z_cm=round(r["z_cm"], 1), tilt=round(r["tilt"], 1))
    for key in ("z", "grav", "q", "quat", "vel", "qd"):
        out[key] = [x[key] for x in rows]
    json.dump(out, open(os.path.join(HERE, "ref_state.json"), "w", encoding="utf-8"))
    print(f"完整身体上的这次起身（mujoco {mujoco.__version__}）：{'站起来了' if r['ok'] else '没站起来'}，骨盆高 {r['z_cm']:.1f} cm，倾斜 {r['tilt']:.0f}°；记下 {len(rows)} 帧")
    for t in (0, 240, 421, 512, 609, 659, 719, 819, T - 1):
        print(f"  {t*ref['dt']:5.2f}s 骨盆高 {rows[t]['z']*100:5.1f} cm  下方向(骨盆坐标) {rows[t]['grav']}")


if __name__ == "__main__":
    main()
