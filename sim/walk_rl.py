"""
用强化学习学会的走路（sim/rl/walk_policy.npz）走一段指定的距离（给 day.py 的 --walk=rl 用）。

控制循环照 sim/rl/eval_walk.py 的 run()：
  - 观测 49 个数：重力方向、骨盆角速度×0.25、指令×CMD_SCALE、腿上 12 个关节角（减站姿）、角速度×0.05、上一步输出、节拍的 cos/sin
  - 目标角 = 站姿 + 0.5 × 策略输出，限在关节范围内；每 20 ms 一次；数字都从 rl/walk_common.py 取
  - 朝向控制（eval_walk.py 里 opts["hold"] 那一层）：按指令的转速累加"该朝哪边"，偏了就在转向指令上补 HOLD_K × 偏差，限 ±0.8 rad/s
  - 走路期间自动配平（sim.trim）关掉，交回时打开
和 eval_walk.py 不同的地方：起始姿势不加随机偏差、节拍从 0 开始（一天里的这一段是确定的）；没有推。

一段的顺序：
  1. 指令 15 cm/s 往前走，直到骨盆沿 x 走出 dist（默认 11.2 cm = 手写静步行在一天里那一段实际走出的距离）
  2. 指令 0 站 2 秒（策略还在控制）
  3. 交回（打开自动配平）：1 秒里策略继续跑（指令 0），它给的目标姿势按比例混进 day.py 平时站着的姿势（stand_pose + 平衡控制），
     比例从 0 平滑升到 1。第一版（"freeze"）是策略停下、目标角从它最后的姿势过渡过去：第 1 轮策略可以，
     第 2 轮策略站着时姿势左右不对称、要一直调（髋侧 0.26、两膝 0.13 和 0.59 弧度），一停就倒（0.57 秒，过渡 1–3 秒都一样）
  4. 交回后再站 2 秒，确认没摔

用法（单独跑一次）：
    python walk_rl.py              # 走 11.2 cm；打印结果
    python walk_rl.py --heat=10    # 另外：连续按 15 cm/s 走 10 秒，打印腿上各电机的平均铜损（给 day_walk_compare.py 用的同一个函数）
"""
import math
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "rl"))
import walk_common as W          # noqa: E402
from policy_np import Policy     # noqa: E402

HOLD_K = 1.5                     # 和 rl/eval_walk.py 相同：朝向每偏 1 弧度，转向指令补 1.5 rad/s
STATIC_DX = 0.112                # 手写静步行（walk.walk(sim, b, 4, 0.03, T=2.4)）在一天里实际走出 11.2 cm（digital_huaran_day.js 第 5 天）
V_CMD = 0.15
STAND_S, BLEND_S, HOLD_S = 2.0, 1.0, 2.0
MAX_WALK_S = 15.0
HANDOFF = os.environ.get("WALK_HANDOFF", "mix")   # 交回方式，见 run()；day_walk_compare.py 用环境变量另跑一次 "freeze" 对照
_POL = None


def policy():
    global _POL
    if _POL is None:
        # 环境变量 WALK_POLICY 可以换一份策略（文件名，相对 sim/rl/），验收新一轮时用，不动 walk_policy.npz
        _POL = Policy(os.path.join(HERE, "rl", os.environ.get("WALK_POLICY", "walk_policy.npz")))
    return _POL


def est_seconds(dist=STATIC_DX, v=V_CMD):
    """day.py 排时间表用的估计（起步约 0.5 秒）。"""
    return dist / v + 0.5 + STAND_S + BLEND_S + HOLD_S


def _smooth(a):
    a = max(0.0, min(1.0, a))
    return 0.5 - 0.5 * math.cos(math.pi * a)


class Walker:
    def __init__(self, sim):
        self.sim = sim
        m, d = sim.m, sim.d
        self.pol = policy()
        jid = [m.joint(n).id for n in W.LEGS]
        self.qa = np.array([m.jnt_qposadr[j] for j in jid])
        self.va = np.array([m.jnt_dofadr[j] for j in jid])
        self.lo, self.hi = m.jnt_range[jid, 0], m.jnt_range[jid, 1]
        self.pel = m.body("pelvis").id
        self.dp = W.default_pose()
        self.q0 = np.array([self.dp.get(n, 0.0) for n in W.LEGS])
        self.n_sub = int(round(W.DT / m.opt.timestep))
        self.cscale = np.array(W.CMD_SCALE)
        self.last = np.zeros(12)
        self.t = 0
        self.yaw0 = self.yaw()
        self.yaw_ref, self.yaw_acc, self.yaw_prev = self.yaw0, 0.0, self.yaw0
        self.pose = dict(self.dp)
        self.fell_at = None
        self.trim = getattr(sim, "trim", None)
        if self.trim is not None:
            self.trim.enabled = False

    def yaw(self):
        R = self.sim.d.xmat[self.pel].reshape(3, 3)
        return math.atan2(R[1, 0], R[0, 0])

    def fallen(self):
        d = self.sim.d
        up = d.xmat[self.pel].reshape(3, 3)[2, 2]
        return d.qpos[2] < W.FALL_Z or up < math.cos(math.radians(W.FALL_DEG)) or not np.all(np.isfinite(d.qpos))

    def step(self, cmd, on_substep=None, mix=None):
        """策略走 20 ms。cmd = (前进 m/s, 侧移 m/s, 转向 rad/s)。mix(pose) 不为空时，每个小步把策略的目标姿势交给它改一下再发出去（交回时用）。"""
        sim, d = self.sim, self.sim.d
        cmd = np.array(cmd, dtype=float)
        if cmd.any():                       # 朝向控制（eval_walk.py 的 hold）
            self.yaw_ref += cmd[2] * W.DT
            err = (self.yaw_ref - self.yaw() + math.pi) % (2 * math.pi) - math.pi
            cmd[2] = float(np.clip(cmd[2] + HOLD_K * err, -0.8, 0.8))
        R = d.xmat[self.pel].reshape(3, 3)
        ph = self.t * W.DT * W.GAIT_HZ
        a2 = 2 * math.pi * np.array([ph % 1.0, (ph + 0.5) % 1.0])
        obs = np.concatenate([-R[2, :], d.qvel[3:6] * 0.25, cmd * self.cscale, d.qpos[self.qa] - self.q0, d.qvel[self.va] * 0.05,
                              self.last, np.cos(a2), np.sin(a2)])
        a = self.pol(obs)
        want = np.clip(self.q0 + W.ACT_SCALE * a, self.lo, self.hi)
        pose = dict(self.dp)
        pose.update({n: float(v) for n, v in zip(W.LEGS, want)})
        self.pose = pose
        for _ in range(self.n_sub):
            sim.set_targets(mix(pose) if mix else pose)
            sim.step()
            if on_substep:
                on_substep()
        self.last = a
        self.t += 1
        y = self.yaw()
        self.yaw_acc += (y - self.yaw_prev + math.pi) % (2 * math.pi) - math.pi
        self.yaw_prev = y
        if self.fell_at is None and self.fallen():
            self.fell_at = round(self.t * W.DT, 2)


def run(sim, bal=None, dist=STATIC_DX, v=V_CMD, reset=True, handoff=None):
    """走 dist 米再站住、交回。返回结果字典（时间都是仿真秒）。
    handoff："mix"（默认，HANDOFF）= 交回的 1 秒里策略还在跑，它的目标姿势按比例混进平时的站姿；
             "freeze" = 2026-11-16 第一版：策略停下，目标角从它最后的姿势过渡到平时的站姿（第 2 轮策略这样会在 0.57 秒摔倒）。"""
    handoff = handoff or HANDOFF
    from balance import Balance
    from figure_test import stand_pose
    d = sim.d
    if reset:
        sim.reset()
    mujoco.mj_forward(sim.m, d)
    w = Walker(sim)
    x0, y0 = float(d.qpos[0]), float(d.qpos[1])
    # 1. 走
    while float(d.qpos[0]) - x0 < dist and w.t * W.DT < MAX_WALK_S and w.fell_at is None:
        w.step((v, 0.0, 0.0))
    walk_s = w.t * W.DT
    reached = float(d.qpos[0]) - x0 >= dist
    # 2. 指令 0 站 2 秒
    for _ in range(int(round(STAND_S / W.DT))):
        w.step((0.0, 0.0, 0.0))
    dx, dy = float(d.qpos[0]) - x0, float(d.qpos[1]) - y0
    heading = math.degrees(w.yaw_acc)
    policy_s = w.t * W.DT
    # 3. 交回：目标角在 1 秒内从策略的最后姿势平滑过渡到平时的站姿（加平衡控制），打开自动配平
    bal = bal or Balance(sim)
    bal.prev = None
    if w.trim is not None:
        w.trim.enabled = True
    dt = sim.m.opt.timestep
    handoff_fell = None
    t_h = 0.0
    if handoff == "mix":
        # 策略继续跑（指令 0），它的目标姿势和平时的站姿按 a 混合，a 在 BLEND_S 秒里从 0 平滑升到 1
        nb = int(round(BLEND_S / W.DT))
        for k in range(nb):
            a = _smooth((k + 1) / nb)

            def mix(pose, a=a):
                p_to = bal.adjust(stand_pose())
                return {n: pose.get(n, 0.0) + (p_to.get(n, 0.0) - pose.get(n, 0.0)) * a for n in set(pose) | set(p_to)}
            w.step((0.0, 0.0, 0.0), mix=mix)
            t_h += W.DT
            if handoff_fell is None and (sim.fallen() or w.fallen()):
                handoff_fell = round(t_h, 2)
        p_from, n_blend = {}, 0
    else:
        p_from = dict(w.pose)
        n_blend = int(round(BLEND_S / dt))
    n_hold = int(round(HOLD_S / dt))
    for k in range(n_blend + n_hold):
        a = _smooth(k / n_blend) if n_blend else 1.0
        p_to = bal.adjust(stand_pose())
        keys = set(p_from) | set(p_to)
        sim.set_targets({n: p_from.get(n, 0.0) + (p_to.get(n, 0.0) - p_from.get(n, 0.0)) * a for n in keys})
        sim.step()
        t_h += dt
        if handoff_fell is None and (sim.fallen() or w.fallen()):
            handoff_fell = round(t_h, 2)
    return dict(reached=bool(reached), walk_s=round(walk_s, 2), dx_cm=round(dx * 100, 2), dy_cm=round(dy * 100, 2),
                heading_deg=round(heading, 1), fell_at=w.fell_at, handoff_fell_at=handoff_fell,
                policy_s=round(policy_s, 2), total_s=round(policy_s + t_h, 2),
                end_tilt=round(sim.tilt_deg(), 1), end_z_cm=round(float(d.qpos[2]) * 100, 1))


def mean_tau2(sim, joints, seconds=10.0, v=V_CMD, skip=1.0, mean_out=None):
    """连续按 v 走 seconds 秒（从站姿开始，前 skip 秒不算），返回各关节扭矩平方的平均值（N·m²）和走出的距离。
    mean_out 给一个字典时，另外填进各关节扭矩的平均值（带正负号，看是不是一直偏向一边）。"""
    sim.reset()
    w = Walker(sim)
    acc, n = {j: 0.0 for j in joints}, [0]
    acc1 = {j: 0.0 for j in joints}

    def sub():
        if w.t * W.DT >= skip:
            tq = sim.torques()
            for j in joints:
                x = float(tq.get(j, 0.0))
                acc[j] += x ** 2
                acc1[j] += x
            n[0] += 1
    x0 = float(sim.d.qpos[0])
    while w.t * W.DT < seconds and w.fell_at is None:
        w.step((v, 0.0, 0.0), sub)
    if w.trim is not None:
        w.trim.enabled = True
    if mean_out is not None:
        mean_out.update({j: acc1[j] / max(1, n[0]) for j in joints})
    return {j: acc[j] / max(1, n[0]) for j in joints}, float(sim.d.qpos[0]) - x0, w.fell_at


if __name__ == "__main__":
    import getup_hands as G
    sim = G.get_sim()
    print(run(sim))
    h = next((float(a.split("=")[1]) for a in sys.argv if a.startswith("--heat=")), None)
    if h:
        legs = [f"{s}_{j}" for s in "lr" for j in ("hip_pitch", "hip_roll", "knee", "ankle_pitch", "ankle_roll")]
        t2, dx, fell = mean_tau2(sim, legs, h)
        print(f"连续走 {h} 秒：{dx*100:.0f} cm，摔 {fell}；扭矩平方平均：", {k: round(v, 4) for k, v in t2.items()})
