"""
起身的强化学习环境（MJX，在显卡上同时跑几千个花然）。

花然照着一条成功过的起身动作练，策略学的是"在参考动作上加多少修正"：
    电机目标 = 参考动作此刻的角度（ref_getup.json） + 0.3 弧度 × 策略的输出
沿用整体模型的扭矩上限；目标角每 20 ms 最多变 0.2 弧度（= 10 rad/s 的转速上限）。

第二版（2026-11-11）的改动，原因见 wiki/灵活度.md：
  - 奖励改成"身体的状态跟不跟得上成功那次"（ref_state.json：骨盆高、朝向、关节实际角度），
    不再只看电机目标角。第一版只跟目标角，在简化身体上那条动作本来就撑不起来，练了 2500 万步没有起色。
  - 练习可以从动作的中间开始（七成的身体从随机的时刻开始，三成从趴着开始），这样后半段（松手、站起）也练得到。
  - 歪得离参考太远就算这一局结束，重新开始。

每一局的随机变化：地面摩擦 0.5–1.7、起始姿势的小偏差、偶尔被推一下。
（第四轮 2026-11-11：摩擦上限 1.5 → 1.7。第三轮的策略在训练用的身体上，摩擦 1.5 也只有 54% 站起来——
 1.5 在范围的最边上，练得少；另外简化身体的摩擦锥改成和完整身体一样的 elliptic，见 build_rl_model.py。）
它能感觉到的（观测）：身体朝哪边歪、转得多快、每个关节的角度和速度、上一步自己的输出、现在该做到哪一步。
"""
import json
import math
import os

import jax
import jax.numpy as jnp
import mujoco
from brax.envs.base import Env, State
from mujoco import mjx

ACT_SCALE = 0.3
SLEW = 0.2                 # 每个控制步（20 ms）目标角最多变化的弧度
FRIC = (0.5, 1.7)
PUSH_PROB, PUSH_N = 0.002, 3.0     # 每步有 0.2% 的机会被推一下（3 N，持续这一步）
PUSH_STEPS = 1             # 推一下持续几步（20 ms 一步）。转头那一轮（2026-11-12）用 10 步 = 0.2 秒、3–6 N，对上 eval_robust.py --push=5
# 以下是"头偏向一边"那一轮（2026-11-12，train_getup.py --ref=turn）加的，默认关着：
HEAD_FRIC = None           # (lo, hi)：头和地面之间单独的摩擦，每局随机。做法：地面的摩擦设得很低（0.01），
                           # 身体各处 = fric、头 = 这个值；MuJoCo 两面取大，所以身体碰地用 fric、头碰地用头的值（和完整身体上 priority 的效果一样）
VARY = False               # 起始姿势多样：脖子随机（每局开头重新抽），趴着开始时身体绕长轴歪一点
NECK_VARY = (0.6, 0.3, 0.3, 0.2)   # neck_yaw、neck_pitch、neck_low、neck_roll 的 ±弧度（和 eval_robust.py --vary 一样）
ROLL_VARY = 0.08
FACE_W = 0.0               # 脸着地的扣分（每步）。"脸着地"：头的球心离地 < 半径 + 3 mm，且脸的正前方离正下方 < 45°
                           # （和 eval_robust.py --face 的"着地点离脸正前方 < 45°"是同一件事）
P_PRONE = 0.3              # 从趴着开始的比例；其余从参考动作的随机时刻开始
FALL_DEG = 60.0            # 朝向和参考差这么多就算倒了
FALL_Z = 0.07              # 骨盆比参考低这么多也算倒了


class GetUp(Env):
    def __init__(self, xml_path, ref_path, n_frames=10, state_path=None, mid_start=True):
        """ref_path 可以是一份，也可以是几份（比如转左、转右各一份）：每个身体开始时抽一份。
        state_path 同样；不给时用同目录的 ref_state.json（一份时）或和 ref_getup_turnX.json 配对的 ref_state_turnX.json。"""
        self.mj = mujoco.MjModel.from_xml_path(xml_path)
        self.mx = mjx.put_model(self.mj)
        ref_paths = [ref_path] if isinstance(ref_path, str) else list(ref_path)
        if state_path is None:
            state_paths = [os.path.join(os.path.dirname(p), os.path.basename(p).replace("ref_getup", "ref_state")) for p in ref_paths]
        else:
            state_paths = [state_path] if isinstance(state_path, str) else list(state_path)
        refs = [json.load(open(p, encoding="utf-8")) for p in ref_paths]
        ref = refs[0]
        assert ref["names"] == [self.mj.actuator(i).name for i in range(self.mj.nu)]
        self.ref = jnp.array([r["q"] for r in refs], dtype=jnp.float32)        # [几份, T, 电机]
        self.S = len(refs)
        self.T = int(self.ref.shape[1])
        self.t_free = int(ref["t_free"] / ref["dt"])
        self.n_frames = n_frames
        self.dt = float(self.mj.opt.timestep) * n_frames
        assert abs(self.dt - ref["dt"]) < 1e-9
        sts = [json.load(open(p, encoding="utf-8")) for p in state_paths]
        for st in sts:
            assert st["names"] == ref["names"] and len(st["z"]) == self.T and st["ok"]
        f32 = lambda k: jnp.array([st[k] for st in sts], dtype=jnp.float32)
        self.s_z, self.s_grav, self.s_q, self.s_quat, self.s_vel, self.s_qd = (f32(k) for k in ("z", "grav", "q", "quat", "vel", "qd"))
        self.neck_i = jnp.array([ref["names"].index(n) for n in ("neck_yaw", "neck_pitch", "neck_low", "neck_roll")])
        self.floor_g = self.mj.geom("floor").id
        self.head_g = self.mj.geom("head").id
        self.head_r = float(self.mj.geom_size[self.head_g, 0])
        self.mid_start = mid_start
        jid = self.mj.actuator_trnid[:, 0]
        self.act_q = jnp.array(self.mj.jnt_qposadr[jid])
        self.act_v = jnp.array(self.mj.jnt_dofadr[jid])
        self.lo = jnp.array(self.mj.jnt_range[jid, 0], dtype=jnp.float32)
        self.hi = jnp.array(self.mj.jnt_range[jid, 1], dtype=jnp.float32)
        self.frc = jnp.array(self.mj.actuator_forcerange[:, 1], dtype=jnp.float32)
        self.pelvis = self.mj.body("pelvis").id
        self.chest = self.mj.body("chest").id
        self.q0 = jnp.array(self.mj.qpos0, dtype=jnp.float32)
        self.fric0 = self.mx.geom_friction
        # 手的软硬每一局随机变：手指的刚度、手指放松时弯多少、手腕的刚度（相对标称值的倍数）
        fj = [self.mj.joint(f"{s}_fingers").id for s in "lr"]
        wj = [self.mj.joint(f"{s}_wrist").id for s in "lr"]
        self.fing_j, self.wrist_j = jnp.array(fj), jnp.array(wj)
        self.fing_q = jnp.array(self.mj.jnt_qposadr[fj])
        self.stiff0, self.spring0 = self.mx.jnt_stiffness, self.mx.qpos_spring
        self.wrist_k = float(self.mj.jnt_stiffness[wj[0]])

    def _hand(self, key):
        """随机一只'手感'：[手指刚度 0.05–0.5 N·m/rad（按比例均匀）, 手指放松角 0–0.3 rad, 手腕刚度 ×0.5–2]"""
        u = jax.random.uniform(key, (3,))
        return jnp.array([0.05 * 10.0 ** u[0], 0.3 * u[1], 0.5 * 4.0 ** u[2]])

    # ---- brax 需要的属性 ----
    @property
    def observation_size(self):
        return 3 + 3 + 2 * self.mj.nu + self.mj.nu + self.mj.nu + 3

    @property
    def action_size(self):
        return self.mj.nu

    @property
    def backend(self):
        return "mjx"

    # ---- 观测 ----
    def _obs(self, d, last_act, t, side):
        R = d.xmat[self.pelvis]
        grav = -R[2, :]                                   # 重力方向在骨盆坐标里：它感觉到的"哪边是下"
        ph = t.astype(jnp.float32) / self.T
        ref = self.ref[side, jnp.clip(t, 0, self.T - 1)]  # 转左还是转右，策略从这里（参考的 neck_yaw）看得出来
        return jnp.concatenate([grav, d.qvel[3:6] * 0.25, d.qpos[self.act_q], d.qvel[self.act_v] * 0.1, last_act, ref,
                                jnp.array([ph, jnp.sin(2 * jnp.pi * ph), jnp.cos(2 * jnp.pi * ph)])])

    def _measure(self, d):
        R = d.xmat[self.pelvis]
        return d.qpos[2], R[2, 2], -R[2, :]

    def reset(self, rng):
        rng, k1, k2, k3, k4, k5, k6, k7, k8 = jax.random.split(rng, 9)
        # 这个身体每一局从参考动作的哪一刻开始（训练库在一局结束后会回到同一个开头，所以每个身体的开头是固定的；几千个身体合起来盖住整条动作）
        t0 = jax.random.randint(k3, (), 0, self.T - 100)
        t0 = jnp.where(jax.random.uniform(k4, ()) < P_PRONE, 0, t0) if self.mid_start else jnp.zeros((), dtype=jnp.int32)
        side = jax.random.randint(k6, (), 0, self.S)      # 用哪一份参考（转左 / 转右）；同理每个身体固定，几千个身体各占一半
        quat = self.s_quat[side, t0]
        z0 = self.s_z[side, t0] + 0.003
        if VARY:                                          # 趴着开始时，身体绕世界 x 轴（趴着时的长轴）歪一点，抬高 1 cm 让它落下
            roll = jax.random.uniform(k7, (), minval=-ROLL_VARY, maxval=ROLL_VARY) * (t0 == 0)
            qr = jnp.array([jnp.cos(roll / 2), jnp.sin(roll / 2), 0.0, 0.0])
            quat = self._qmul(qr, quat)
            z0 = z0 + 0.01 * (t0 == 0)
        q = self.q0.at[2].set(z0).at[3:7].set(quat)
        q = q.at[self.act_q].set(self.s_q[side, t0] + jax.random.uniform(k1, (self.mj.nu,), minval=-0.05, maxval=0.05))
        v = jnp.zeros(self.mj.nv, dtype=jnp.float32).at[0:6].set(self.s_vel[side, t0]).at[self.act_v].set(self.s_qd[side, t0])
        d = mjx.make_data(self.mx).replace(qpos=q, qvel=v)
        d = mjx.forward(self.mx, d)
        zero = jnp.zeros(self.mj.nu, dtype=jnp.float32)
        info = dict(t=t0, t0=t0, k=jnp.zeros((), dtype=jnp.int32), target=self.ref[side, t0], last_act=zero, rng=rng,
                    fric=jax.random.uniform(k2, (), minval=FRIC[0], maxval=FRIC[1]), hand=self._hand(k5), side=side,
                    hfric=self._hfric(k8), push_v=jnp.zeros(2), push_left=jnp.zeros((), dtype=jnp.int32))
        z, up, _ = self._measure(d)
        metrics = dict(height=z, up=up, success=jnp.zeros(()), track=jnp.zeros(()), fell=jnp.zeros(()), face=jnp.zeros(()))
        return State(d, self._obs(d, zero, t0, side), jnp.zeros(()), jnp.zeros(()), metrics, info)

    @staticmethod
    def _qmul(a, b):
        w1, x1, y1, z1 = a
        w2, x2, y2, z2 = b
        return jnp.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                          w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])

    def _hfric(self, key):
        return jax.random.uniform(key, (), minval=HEAD_FRIC[0], maxval=HEAD_FRIC[1]) if HEAD_FRIC else jnp.zeros(())

    def step(self, state, action):
        info = state.info
        k = info["steps"].astype(jnp.int32) if "steps" in info else info["k"]      # 训练时由包装器计步（自动重开时会清零）
        t0 = info["t0"]
        t = t0 + k
        first = k == 0
        rng, k1, k2, k3, k4, k5, k6 = jax.random.split(info["rng"], 7)
        side = info["side"]
        fric = jnp.where(first, jax.random.uniform(k1, (), minval=FRIC[0], maxval=FRIC[1]), info["fric"])
        hand = jnp.where(first, self._hand(k4), info["hand"])
        hfric = jnp.where(first, self._hfric(k5), info["hfric"])
        prev = jnp.where(first, self.ref[side, t0], info["target"])
        last_act = jnp.where(first, jnp.zeros_like(info["last_act"]), info["last_act"])
        action = jnp.clip(action, -1.0, 1.0)
        tc = jnp.clip(t, 0, self.T - 1)
        ref = self.ref[side, tc]
        want = jnp.clip(ref + ACT_SCALE * action, self.lo, self.hi)
        target = prev + jnp.clip(want - prev, -SLEW, SLEW)
        gf = self.fric0 * jnp.array([fric, 1.0, 1.0])
        if HEAD_FRIC:                     # 地面很滑、头单独一个值：碰地时身体用 fric，头用 hfric（两面取大）
            gf = gf.at[self.floor_g, 0].set(0.01).at[self.head_g, 0].set(hfric)
        mx = self.mx.replace(geom_friction=gf,
                             jnt_stiffness=self.stiff0.at[self.fing_j].set(hand[0]).at[self.wrist_j].set(self.wrist_k * hand[2]),
                             qpos_spring=self.spring0.at[self.fing_q].set(hand[1]))
        d0 = state.pipeline_state
        if VARY:                          # 每局开头把脖子重新随机一次（训练库重开时会回到同一个开头，所以在这里抽）
            nq = self.act_q[self.neck_i]
            noise = jax.random.uniform(k6, (4,), minval=-1.0, maxval=1.0) * jnp.array(NECK_VARY)
            neck = jnp.clip(self.s_q[side, t0][self.neck_i] + noise, self.lo[self.neck_i], self.hi[self.neck_i])
            d0 = d0.replace(qpos=jnp.where(first, d0.qpos.at[nq].set(neck), d0.qpos))
        # 推：开始以后持续 PUSH_STEPS 步，方向随机；PUSH_STEPS = 1 时和原来一样（每个方向 ±PUSH_N 均匀）
        left = jnp.where(first, 0, info["push_left"])
        start = (left == 0) & (jax.random.uniform(k2, ()) < PUSH_PROB)
        if PUSH_STEPS > 1:
            u = jax.random.uniform(k3, (2,))
            new_v = PUSH_N * (0.6 + 0.6 * u[0]) * jnp.array([jnp.cos(2 * jnp.pi * u[1]), jnp.sin(2 * jnp.pi * u[1])])
        else:
            new_v = PUSH_N * jax.random.uniform(k3, (2,), minval=-1.0, maxval=1.0)
        push_v = jnp.where(start, new_v, info["push_v"])
        left = jnp.where(start, PUSH_STEPS, jnp.maximum(left - 1, 0))
        push = jnp.where(left > 0, 1.0, 0.0) * push_v
        xfrc = jnp.zeros_like(d0.xfrc_applied).at[self.chest, 0:2].set(push)
        d = d0.replace(ctrl=target, xfrc_applied=xfrc)
        d = jax.lax.fori_loop(0, self.n_frames, lambda _, x: mjx.step(mx, x), d)
        z, up, grav = self._measure(d)
        q = d.qpos[self.act_q]
        # 身体的状态跟不跟得上成功的那次
        t2c = jnp.clip(t + 1, 0, self.T - 1)
        dz = z - self.s_z[side, t2c]
        cosg = jnp.clip(jnp.dot(grav, self.s_grav[side, t2c]), -1.0, 1.0)
        r_z = jnp.exp(-300.0 * dz ** 2)                               # 差 5 cm 时剩 0.47
        r_g = jnp.exp(-4.0 * (1.0 - cosg))                            # 朝向差 30° 时剩 0.58
        r_q = jnp.exp(-3.0 * jnp.mean((q - self.s_q[side, t2c]) ** 2))
        track = r_q
        late = (t >= self.t_free).astype(jnp.float32)                 # 松手以后才算"站着"
        success = ((z > 0.20) & (up > math.cos(math.radians(20.0)))).astype(jnp.float32)
        effort = jnp.mean((d.actuator_force / self.frc) ** 2)
        smooth = jnp.mean((action - last_act) ** 2)
        reward = 0.6 * r_z + 0.8 * r_g + 0.3 * r_q + 2.0 * late * success - 0.05 * effort - 0.05 * smooth
        hx = d.geom_xmat[self.head_g]                                 # 头的坐标轴在世界里；第 0 列是脸的正前方
        face = ((d.geom_xpos[self.head_g, 2] < self.head_r + 0.003) & (hx[2, 0] < -math.cos(math.radians(45.0)))).astype(jnp.float32)
        reward = reward - FACE_W * face
        bad = jnp.any(jnp.isnan(d.qpos)) | jnp.any(jnp.isnan(d.qvel))
        fell = (cosg < math.cos(math.radians(FALL_DEG))) | (dz < -FALL_Z)
        done = bad | fell
        reward = jnp.where(done, 0.0, reward) * 0.02
        info = dict(info)
        info.update(t=t + 1, k=k + 1, target=target, last_act=action, rng=rng, fric=fric, hand=hand, hfric=hfric,
                    push_v=push_v, push_left=left)
        metrics = dict(state.metrics)          # 训练库的包装器会往里加自己的项，不能整个换掉
        metrics.update(height=z, up=up, success=success * late, track=track, fell=fell.astype(jnp.float32), face=face)
        return state.replace(pipeline_state=d, obs=self._obs(d, action, t + 1, side), reward=reward, done=done.astype(jnp.float32),
                             metrics=metrics, info=info)
