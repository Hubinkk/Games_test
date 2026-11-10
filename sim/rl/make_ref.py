"""
把成功的起身动作（../getup_hands.py 的 hand_walk）导出成参考轨迹，并在简化身体上回放检查。

用法：
    python make_ref.py        # 写 ref_getup.json（每 20 ms 一组 26 个电机的目标角）；打印回放结果

参考轨迹只是"目标角度随时间怎么变"，不含平衡控制。强化学习在它上面学修正量。
"""
import json
import math
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import getup_hands as G          # noqa: E402

DT_REF = 0.02
W_MAX = 10.0


def export():
    V = json.load(open(os.path.join(HERE, "..", "getup_hands_walk_best.json"), encoding="utf-8"))["V"]
    ks, t_free = G.hand_walk_keyframes(V)
    m = mujoco.MjModel.from_xml_path(os.path.join(HERE, "huaran_rl.xml"))
    names = [m.actuator(i).name for i in range(m.nu)]
    T = ks[-1][0] + 1.5

    def target(t):
        for i in range(1, len(ks)):
            t0, p0 = ks[i - 1][:2]
            t1, p1 = ks[i][:2]
            if t <= t1:
                a = G.sm((t - t0) / (t1 - t0))
                return {k: p0.get(k, 0.0) + (p1.get(k, 0.0) - p0.get(k, 0.0)) * a for k in set(p0) | set(p1)}
        return dict(ks[-1][1])
    rows = []
    for k in range(int(T / DT_REF) + 1):
        p = target(k * DT_REF)
        p["waist_pitch"] = p.get("waist_pitch", 0.0) * 0.5          # 腰两段：每段一半
        rows.append([round(float(p.get(n, 0.0)), 4) for n in names])
    out = dict(dt=DT_REF, names=names, t_free=t_free, marks=[[k[0], k[2]] for k in ks], q=rows)
    json.dump(out, open(os.path.join(HERE, "ref_getup.json"), "w", encoding="utf-8"), ensure_ascii=False)
    return out


def replay(ref, friction=1.0, verbose=True):
    m = mujoco.MjModel.from_xml_path(os.path.join(HERE, "huaran_rl.xml"))
    d = mujoco.MjData(m)
    m.geom_friction[m.geom("floor").id][0] = friction
    d.qpos[0:3] = [0, 0, 0.06]
    d.qpos[3:7] = [math.cos(math.pi / 4), 0, math.sin(math.pi / 4), 0]
    mujoco.mj_forward(m, d)
    q = np.array(ref["q"])
    dt = m.opt.timestep
    cur = np.zeros(m.nu)
    pel = m.body("pelvis").id
    marks = list(ref["marks"])
    for k in range(int(len(q) * ref["dt"] / dt)):
        t = k * dt
        want = q[min(int(t / ref["dt"]), len(q) - 1)]
        cur += np.clip(want - cur, -W_MAX * dt, W_MAX * dt)
        d.ctrl[:] = cur
        mujoco.mj_step(m, d)
        if verbose and marks and t >= marks[0][0]:
            up = d.xmat[pel].reshape(3, 3)[2, 2]
            print(f"{marks[0][0]:5.2f}s {marks[0][1]:<10} 骨盆高 {d.qpos[2]*100:5.1f} cm，倾斜 {math.degrees(math.acos(max(-1, min(1, up)))):4.0f}°")
            marks.pop(0)
    up = d.xmat[pel].reshape(3, 3)[2, 2]
    tilt = math.degrees(math.acos(max(-1, min(1, up))))
    return float(d.qpos[2]) > 0.20 and tilt < 20, float(d.qpos[2]) * 100, tilt


if __name__ == "__main__":
    ref = export()
    print(f"参考轨迹：{len(ref['q'])} 帧，{len(ref['q'])*ref['dt']:.1f} 秒，{len(ref['names'])} 个电机")
    ok, z, tilt = replay(ref)
    print(f"在简化身体上直接回放（没有平衡控制）：{'站起来了' if ok else '没站起来'}，骨盆高 {z:.1f} cm，倾斜 {tilt:.0f}°")
    for f in (0.6, 0.8, 1.2):
        ok, z, tilt = replay(ref, f, verbose=False)
        print(f"  地面摩擦 {f}：{'站起来了' if ok else '没站起来'}，骨盆高 {z:.1f} cm，倾斜 {tilt:.0f}°")
