"""
趴着时把手挪到肩膀旁边、再撑起上身（俯卧撑）：搜索胳膊的三段动作。

用法：
    GETUP_METHOD=kneel python pushup.py     # 结果存在 pushup_best.json

背景（2026-11-11）：之前所有起身尝试里，手都落在屁股旁边，没能放到肩膀下面；
胳膊贴着身体直接弯肘时，前臂往地面里压，手会卡住。人是先把胳膊张开、贴着地面滑到肩旁再撑。
这里只搜胳膊（肩俯仰、肩侧摆、肩旋转、肘）三个关键帧，脖子放松；目标是胸口离地越高越好、手在肩膀下方。
"""
import json
import math
import os
from multiprocessing import Pool

import mujoco
import numpy as np

import getup

HERE = os.path.dirname(os.path.abspath(__file__))
J = ("sp", "roll", "yaw", "el")
RNG = {"sp": (-3.0, 0.8), "roll": (-2.6, 0.3), "yaw": (-1.5, 1.5), "el": (-2.4, 0.0)}
TIMES = (0.3, 1.3, 2.3, 3.6)       # 三段动作的起止时刻
HOLD = 1.0


def arms(p, sp, roll, yaw, el):
    p["r_shoulder_pitch"], p["l_shoulder_pitch"] = sp, sp
    p["r_shoulder_roll"], p["l_shoulder_roll"] = roll, -roll
    p["r_shoulder_yaw"], p["l_shoulder_yaw"] = yaw, -yaw
    p["r_elbow"], p["l_elbow"] = el, el


def run(x, record=None):
    sim = getup.get_sim()
    m, d = sim.m, sim.d
    sim.reset()
    d.qpos[0:3] = [0, 0, 0.06]
    d.qpos[3:7] = [math.cos(math.pi / 4), 0, math.sin(math.pi / 4), 0]
    mujoco.mj_forward(m, d)
    neck = m.actuator("neck_pitch").id
    nb = m.actuator_forcerange[neck].copy()
    m.actuator_forcerange[neck] = nb * 0.05
    keys = [(0.0, (0.0, 0.0, 0.0, 0.0))] + [(TIMES[i + 1], tuple(x[f"{j}{i}"] for j in J)) for i in range(3)]
    dt = m.opt.timestep
    T = TIMES[-1] + HOLD
    for k in range(int(T / dt)):
        t = k * dt
        cur = keys[-1][1]
        for i in range(1, len(keys)):
            if t <= keys[i][0]:
                t0 = TIMES[0] if i == 1 else keys[i - 1][0]
                a = getup.sm((t - t0) / (keys[i][0] - t0))
                cur = tuple(p0 + (p1 - p0) * a for p0, p1 in zip(keys[i - 1][1], keys[i][1]))
                break
        p = getup.lying()
        arms(p, *cur)
        sim.set_targets(p)
        sim.step()
        if record:
            record(t)
    m.actuator_forcerange[neck] = nb
    chest = float(d.xpos[m.body("chest").id][2])
    dx = float(d.xpos[m.body("R_palm").id][0] - d.xpos[m.body("r_upper_arm").id][0])
    return dict(chest_cm=chest * 100, hand_dx_cm=dx * 100, score=chest * 100 - 0.15 * abs(dx * 100))


def sample(rng, c=None, w=1.0):
    x = {}
    for i in range(3):
        for j in J:
            lo, hi = RNG[j]
            k = f"{j}{i}"
            x[k] = float(rng.uniform(lo, hi)) if c is None else float(np.clip(c[k] + rng.normal(0, 0.15 * w) * (hi - lo), lo, hi))
    return x


def ev(x):
    r = run(x)
    return r["score"], x, r


if __name__ == "__main__":
    rng = np.random.default_rng(3)
    # 人的做法作为一个起点：先张开（侧摆），再转、弯肘把手带到肩旁，最后伸肘撑起
    human = dict(sp0=0.0, roll0=-1.3, yaw0=0.0, el0=0.0,
                 sp1=-0.9, roll1=-1.3, yaw1=1.0, el1=-1.8,
                 sp2=-1.4, roll2=-0.4, yaw2=0.3, el2=-0.3)
    with Pool(4) as pool:
        res = pool.map(ev, [human] + [sample(rng) for _ in range(119)])
        res.sort(key=lambda t: -t[0])
        best = res[0]
        print("随机：", [(round(r["chest_cm"], 1), round(r["hand_dx_cm"], 1)) for _, _, r in res[:4]], flush=True)
        for rnd in range(5):
            more = pool.map(ev, [sample(rng, best[1], 1 / (rnd + 1)) for _ in range(24)])
            more.sort(key=lambda t: -t[0])
            if more[0][0] > best[0]:
                best = more[0]
            print(f"细化 {rnd+1}：胸口 {best[2]['chest_cm']:.1f} cm，手在肩前后 {best[2]['hand_dx_cm']:.1f} cm", flush=True)
    json.dump(dict(x=best[1], result=best[2]), open(os.path.join(HERE, "pushup_best.json"), "w"), ensure_ascii=False, indent=1)
    print("最好：", {k: round(v, 2) for k, v in best[1].items()})
