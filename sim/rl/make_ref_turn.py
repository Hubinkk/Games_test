"""
把"起身时头偏向一边"写进参考动作（2026-11-12，docs/next-local.md 任务一）。

原来的做法（sim/getup_rl.py 的 TURN_HEAD）是起身的头 4.5 秒把脖子固定成"转向左边并低头"，不听策略的。
这里把同样的姿势写进参考动作，让策略自己学会转头；左右各一份：

    ref_getup_turnL.json / ref_getup_turnR.json      电机的目标角（和 ref_getup.json 同样的格式）
    ref_state_turnL.json / ref_state_turnR.json      身体状态的参考：原来的 ref_state.json，脖子换成转头（见 make_state 的说明）

只改脖子的三个角（neck_yaw、neck_pitch、neck_low），身体其余部分和原来的参考动作一模一样（不重新设计起身动作）：
    0–0.15 秒   从原来的角度平滑转到 neck_yaw = ±1.3、neck_pitch = 0.4、neck_low = 0.3（左为 +）
    0.3–4.5 秒  保持
    4.5–5.0 秒  平滑回到原来的角度
策略从观测里的"参考动作这一步的角"知道该转哪边（neck_yaw 的正负），观测的 113 个数不变。

用法（用项目的 mujoco 3.17）：  python make_ref_turn.py
旧的 ref_getup.json、ref_state.json 不动（旧策略 getup_policy.npz / run3_final.npz 还用它们）。
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import getup_hands as G          # noqa: E402

TURN = dict(neck_yaw=1.3, neck_pitch=0.4, neck_low=0.3)      # 和 getup_rl.TURN_HEAD 一样
T_IN, T_HOLD, T_OUT = 0.15, 4.5, 5.0      # 转进去要快：趴着开始时脸贴着地，转得越慢脸着地越久（0.3 秒时右边约 12%）


def weight(t):
    """0 = 原来的脖子角度，1 = 转头的姿势。"""
    if t < T_IN:
        return G.sm(t / T_IN)
    if t <= T_HOLD:
        return 1.0
    if t < T_OUT:
        return 1.0 - G.sm((t - T_HOLD) / (T_OUT - T_HOLD))
    return 0.0


def turn_of(side):
    s = 1.0 if side == "L" else -1.0
    return dict(neck_yaw=s * TURN["neck_yaw"], neck_pitch=TURN["neck_pitch"], neck_low=TURN["neck_low"])


def apply(pose, t, side):
    """在一组目标角上加转头（pose 用整体模型接口的名字；脖子三个角不受腰的 ×2 影响）。"""
    w = weight(t)
    if w <= 0:
        return pose
    pose = dict(pose)
    for k, v in turn_of(side).items():
        p0 = pose.get(k, 0.0)
        pose[k] = p0 + (v - p0) * w
    return pose


def make_ref(side):
    ref = json.load(open(os.path.join(HERE, "ref_getup.json"), encoding="utf-8"))
    names = ref["names"]
    q = np.array(ref["q"])
    for k in range(len(q)):
        p = apply({n: float(q[k, i]) for i, n in enumerate(names) if n.startswith("neck")}, k * ref["dt"], side)
        for n, v in p.items():
            q[k, names.index(n)] = round(v, 4)
    ref["q"] = q.tolist()
    ref["side"] = side
    ref["turn"] = dict(turn_of(side), t_in=T_IN, t_hold=T_HOLD, t_out=T_OUT)
    json.dump(ref, open(os.path.join(HERE, f"ref_getup_turn{side}.json"), "w", encoding="utf-8"), ensure_ascii=False)
    return ref


def make_state(side):
    """身体状态的参考：用原来成功那次的 ref_state.json，只把脖子三个关节的角换成转头后的目标角（速度用差分）。

    为什么不重新录：手写的起身加上转头以后，在完整身体上站不起来了（2026-11-12 实测：左右都是骨盆高 4.6 cm、倾斜 92°，
    见 record_state）；而学到的策略（run3）加转头覆盖是站得起来的（eval_robust.py --neck=turn）。
    所以身体其余部分仍然跟原来那次，脖子跟转头。
    """
    st = json.load(open(os.path.join(HERE, "ref_state.json"), encoding="utf-8"))
    ref = json.load(open(os.path.join(HERE, f"ref_getup_turn{side}.json"), encoding="utf-8"))
    names = ref["names"]
    q, qd = np.array(st["q"]), np.array(st["qd"])
    rq = np.array(ref["q"])
    for n in TURN:
        i = names.index(n)
        q[:, i] = rq[:, i]
        qd[:, i] = np.gradient(rq[:, i], ref["dt"])
    st["q"] = np.round(q, 4).tolist()
    st["qd"] = np.round(qd, 4).tolist()
    st["side"] = side
    st["note"] = "ref_state.json 的身体 + 转头后的脖子目标角（手写起身加转头站不起来，所以没重新录）"
    json.dump(st, open(os.path.join(HERE, f"ref_state_turn{side}.json"), "w", encoding="utf-8"))
    return st


def record_state(side):
    """（只用来检查）完整身体上照着手写的起身（带转头）跑一遍。2026-11-12：左右都站不起来，所以 make_state 不用它。"""
    import mujoco
    ref = json.load(open(os.path.join(HERE, f"ref_getup_turn{side}.json"), encoding="utf-8"))
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
    orig = sim.set_targets
    sim.set_targets = lambda pose: orig(apply(pose, d.time, side))
    try:
        r = G.hand_walk(V, verbose=False, record=record)
    finally:
        sim.set_targets = orig
    T = len(ref["q"])
    rows = (rows + [rows[-1]] * T)[:T]
    out = dict(dt=ref["dt"], names=names, side=side, mujoco=mujoco.__version__, ok=r["ok"], z_cm=round(r["z_cm"], 1), tilt=round(r["tilt"], 1))
    for key in ("z", "grav", "q", "quat", "vel", "qd"):
        out[key] = [x[key] for x in rows]
    yi = names.index("neck_yaw")
    print(f"转向{'左' if side == 'L' else '右'}：完整身体上照着做（mujoco {mujoco.__version__}）{'站起来了' if r['ok'] else '没站起来'}，"
          f"骨盆高 {r['z_cm']:.1f} cm，倾斜 {r['tilt']:.0f}°；neck_yaw 实际角 1/2/4/6 秒："
          + " ".join(f"{rows[int(s / ref['dt'])]['q'][yi]:+.2f}" for s in (1, 2, 4, 6)), flush=True)
    return out


if __name__ == "__main__":
    for s in "LR":
        make_ref(s)
        make_state(s)
        if "--check-hand" in sys.argv:
            record_state(s)
    print("写好了 ref_getup_turnL/R.json、ref_state_turnL/R.json")
