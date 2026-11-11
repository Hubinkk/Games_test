"""
把训练好的策略放回完整的身体（有手指、绳驱膝的整体模型）里检验：它在没见过的、更细的身体上还站不站得起来。

用法：
    python eval_full.py [getup_policy.npz]        # 在几种地面摩擦下各跑一次，写 eval_full_result.json

训练用的是简化身体（huaran_rl.xml）。两个身体的差别（手指、膝的拉线、仿真步长）就是这一步要过的关。
注意："只照参考动作做"在这里本来就站不起来（0/5）。原来成功的那次（getup_hands.py）在松手以后接了手写的平衡控制；
这里故意不接，因为松手以后怎么保持平衡正是策略要学的。已核对：接上平衡控制后，同一条参考动作站到 28.1 cm、倾斜 0°。
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
from policy_np import Policy     # noqa: E402

ACT_SCALE, SLEW = 0.3, 0.2


def run(pol, ref, friction=1.0, use_policy=True):
    sim = G.get_sim()
    m, d = sim.m, sim.d
    # 两个面接触时 MuJoCo 取两者中较大的摩擦。身体各处本来是 1.0，只改地面的话 0.6、0.8 和 1.0 其实是同一种情况
    # （2026-11-11 发现）。所以把所有接触面一起改，这样"地面摩擦 0.6"才真的是 0.6。
    m.geom_friction[:, 0] = friction
    names = ref["names"]
    q_ref = np.array(ref["q"])
    T = len(q_ref)
    jid = [m.joint(n).id for n in names]
    qa = np.array([m.jnt_qposadr[j] for j in jid])
    va = np.array([m.jnt_dofadr[j] for j in jid])
    lo, hi = m.jnt_range[jid, 0], m.jnt_range[jid, 1]
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
    for t in range(T + 25):
        r = q_ref[min(t, T - 1)]
        ph = t / T
        obs = np.concatenate([-d.xmat[pel].reshape(3, 3)[2, :], d.qvel[3:6] * 0.25, d.qpos[qa], d.qvel[va] * 0.1, last, r,
                              [ph, math.sin(2 * math.pi * ph), math.cos(2 * math.pi * ph)]])
        a = pol(obs) if use_policy else np.zeros(len(names))
        want = np.clip(r + ACT_SCALE * a, lo, hi)
        target = target + np.clip(want - target, -SLEW, SLEW)
        pose = {n: float(v) for n, v in zip(names, target)}
        pose["waist_pitch"] = 2.0 * float(target[wi])        # 整体模型的接口里，腰写的是两段加起来的角度
        for _ in range(n_sub):
            sim.set_targets(pose)
            sim.step()
        last = a
    if getattr(sim, "trim", None) is not None:
        sim.trim.enabled = True
    z, tilt = float(d.qpos[2]), sim.tilt_deg()
    return dict(friction=friction, ok=bool(z > 0.20 and tilt < 20), z_cm=round(z * 100, 1), tilt=round(tilt))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else os.path.join(HERE, "getup_policy.npz")
    ref = json.load(open(os.path.join(HERE, "ref_getup.json"), encoding="utf-8"))
    pol = Policy(path)
    out = {"policy": [], "reference_only": []}
    for f in (0.6, 0.8, 1.0, 1.2, 1.5):
        a = run(pol, ref, f, True)
        b = run(pol, ref, f, False)
        out["policy"].append(a)
        out["reference_only"].append(b)
        print(f"地面摩擦 {f}：只照参考动作做 {'站起来了' if b['ok'] else '没站起来'}（{b['z_cm']} cm）；"
              f"加上学到的修正 {'站起来了' if a['ok'] else '没站起来'}（{a['z_cm']} cm，倾斜 {a['tilt']}°）", flush=True)
    print(f"加上修正后：{sum(r['ok'] for r in out['policy'])}/5；只照参考：{sum(r['ok'] for r in out['reference_only'])}/5")
    json.dump(out, open(os.path.join(HERE, "eval_full_result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
