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
P_PRONE = 0.3              # 从趴着开始的比例；其余从参考动作的随机时刻开始
FALL_DEG = 60.0            # 朝向和参考差这么多就算倒了
FALL_Z = 0.07              # 骨盆比参考低这么多也算倒了


class GetUp(Env):
    def __init__(self, xml_path, ref_path, n_frames=10, state_path=None, mid_start=True):
        self.mj = mujoco.MjModel.from_xml_path(xml_path)
        self.mx = mjx.put_model(self.mj)
        ref = json.load(open(ref_path, encoding="utf-8"))
        assert ref["names"] == [self.mj.actuator(i).name for i in range(self.mj.nu)]
        self.ref = jnp.array(ref["q"], dtype=jnp.float32)
        self.T = int(self.ref.shape[0])
        self.t_free = int(ref["t_free"] / ref["dt"])
        self.n_frames = n_frames
        self.dt = float(self.mj.opt.timestep) * n_frames
        assert abs(self.dt - ref["dt"]) < 1e-9
        st = json.load(open(state_path or os.path.join(os.path.dirname(ref_path), "ref_state.json"), encoding="utf-8"))
        assert st["names"] == ref["names"] and len(st["z"]) == self.T and st["ok"]
        f32 = lambda k: jnp.array(st[k], dtype=jnp.float32)
        self.s_z, self.s_grav, self.s_q, self.s_quat, self.s_vel, self.s_qd = (f32(k) for k in ("z", "grav", "q", "quat", "vel", "qd"))
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
    def _obs(self, d, last_act, t):
        R = d.xmat[self.pelvis]
        grav = -R[2, :]                                   # 重力方向在骨盆坐标里：它感觉到的"哪边是下"
        ph = t.astype(jnp.float32) / self.T
        ref = self.ref[jnp.clip(t, 0, self.T - 1)]
        return jnp.concatenate([grav, d.qvel[3:6] * 0.25, d.qpos[self.act_q], d.qvel[self.act_v] * 0.1, last_act, ref,
                                jnp.array([ph, jnp.sin(2 * jnp.pi * ph), jnp.cos(2 * jnp.pi * ph)])])

    def _measure(self, d):
        R = d.xmat[self.pelvis]
        return d.qpos[2], R[2, 2], -R[2, :]

    def reset(self, rng):
        rng, k1, k2, k3, k4, k5 = jax.random.split(rng, 6)
        # 这个身体每一局从参考动作的哪一刻开始（训练库在一局结束后会回到同一个开头，所以每个身体的开头是固定的；几千个身体合起来盖住整条动作）
        t0 = jax.random.randint(k3, (), 0, self.T - 100)
        t0 = jnp.where(jax.random.uniform(k4, ()) < P_PRONE, 0, t0) if self.mid_start else jnp.zeros((), dtype=jnp.int32)
        q = self.q0.at[2].set(self.s_z[t0] + 0.003).at[3:7].set(self.s_quat[t0])
        q = q.at[self.act_q].set(self.s_q[t0] + jax.random.uniform(k1, (self.mj.nu,), minval=-0.05, maxval=0.05))
        v = jnp.zeros(self.mj.nv, dtype=jnp.float32).at[0:6].set(self.s_vel[t0]).at[self.act_v].set(self.s_qd[t0])
        d = mjx.make_data(self.mx).replace(qpos=q, qvel=v)
        d = mjx.forward(self.mx, d)
        zero = jnp.zeros(self.mj.nu, dtype=jnp.float32)
        info = dict(t=t0, t0=t0, k=jnp.zeros((), dtype=jnp.int32), target=self.ref[t0], last_act=zero, rng=rng,
                    fric=jax.random.uniform(k2, (), minval=FRIC[0], maxval=FRIC[1]), hand=self._hand(k5))
        z, up, _ = self._measure(d)
        metrics = dict(height=z, up=up, success=jnp.zeros(()), track=jnp.zeros(()), fell=jnp.zeros(()))
        return State(d, self._obs(d, zero, t0), jnp.zeros(()), jnp.zeros(()), metrics, info)

    def step(self, state, action):
        info = state.info
        k = info["steps"].astype(jnp.int32) if "steps" in info else info["k"]      # 训练时由包装器计步（自动重开时会清零）
        t0 = info["t0"]
        t = t0 + k
        first = k == 0
        rng, k1, k2, k3, k4 = jax.random.split(info["rng"], 5)
        fric = jnp.where(first, jax.random.uniform(k1, (), minval=FRIC[0], maxval=FRIC[1]), info["fric"])
        hand = jnp.where(first, self._hand(k4), info["hand"])
        prev = jnp.where(first, self.ref[t0], info["target"])
        last_act = jnp.where(first, jnp.zeros_like(info["last_act"]), info["last_act"])
        action = jnp.clip(action, -1.0, 1.0)
        tc = jnp.clip(t, 0, self.T - 1)
        ref = self.ref[tc]
        want = jnp.clip(ref + ACT_SCALE * action, self.lo, self.hi)
        target = prev + jnp.clip(want - prev, -SLEW, SLEW)
        mx = self.mx.replace(geom_friction=self.fric0 * jnp.array([fric, 1.0, 1.0]),
                             jnt_stiffness=self.stiff0.at[self.fing_j].set(hand[0]).at[self.wrist_j].set(self.wrist_k * hand[2]),
                             qpos_spring=self.spring0.at[self.fing_q].set(hand[1]))
        push = jnp.where(jax.random.uniform(k2, ()) < PUSH_PROB, 1.0, 0.0) * PUSH_N * jax.random.uniform(k3, (2,), minval=-1.0, maxval=1.0)
        xfrc = jnp.zeros_like(state.pipeline_state.xfrc_applied).at[self.chest, 0:2].set(push)
        d = state.pipeline_state.replace(ctrl=target, xfrc_applied=xfrc)
        d = jax.lax.fori_loop(0, self.n_frames, lambda _, x: mjx.step(mx, x), d)
        z, up, grav = self._measure(d)
        q = d.qpos[self.act_q]
        # 身体的状态跟不跟得上成功的那次
        t2c = jnp.clip(t + 1, 0, self.T - 1)
        dz = z - self.s_z[t2c]
        cosg = jnp.clip(jnp.dot(grav, self.s_grav[t2c]), -1.0, 1.0)
        r_z = jnp.exp(-300.0 * dz ** 2)                               # 差 5 cm 时剩 0.47
        r_g = jnp.exp(-4.0 * (1.0 - cosg))                            # 朝向差 30° 时剩 0.58
        r_q = jnp.exp(-3.0 * jnp.mean((q - self.s_q[t2c]) ** 2))
        track = r_q
        late = (t >= self.t_free).astype(jnp.float32)                 # 松手以后才算"站着"
        success = ((z > 0.20) & (up > math.cos(math.radians(20.0)))).astype(jnp.float32)
        effort = jnp.mean((d.actuator_force / self.frc) ** 2)
        smooth = jnp.mean((action - last_act) ** 2)
        reward = 0.6 * r_z + 0.8 * r_g + 0.3 * r_q + 2.0 * late * success - 0.05 * effort - 0.05 * smooth
        bad = jnp.any(jnp.isnan(d.qpos)) | jnp.any(jnp.isnan(d.qvel))
        fell = (cosg < math.cos(math.radians(FALL_DEG))) | (dz < -FALL_Z)
        done = bad | fell
        reward = jnp.where(done, 0.0, reward) * 0.02
        info = dict(info)
        info.update(t=t + 1, k=k + 1, target=target, last_act=action, rng=rng, fric=fric, hand=hand)
        metrics = dict(state.metrics)          # 训练库的包装器会往里加自己的项，不能整个换掉
        metrics.update(height=z, up=up, success=success * late, track=track, fell=fell.astype(jnp.float32))
        return state.replace(pipeline_state=d, obs=self._obs(d, action, t + 1), reward=reward, done=done.astype(jnp.float32),
                             metrics=metrics, info=info)
