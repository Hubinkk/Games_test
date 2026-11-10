"""
起身的强化学习环境（MJX，在显卡上同时跑几千个花然）。

花然照着一条成功过的起身动作（ref_getup.json）练，策略学的是"在参考动作上加多少修正"：
    电机目标 = 参考动作此刻的角度 + 0.3 弧度 × 策略的输出
沿用整体模型的扭矩上限；目标角每 20 ms 最多变 0.2 弧度（= 10 rad/s 的转速上限）。

每一局的随机变化：地面摩擦 0.5–1.5、起始姿势的小偏差、偶尔被推一下。
它能感觉到的（观测）：身体朝哪边歪、转得多快、每个关节的角度和速度、上一步自己的输出、现在该做到哪一步。
"""
import json
import math

import jax
import jax.numpy as jnp
import mujoco
from brax.envs.base import Env, State
from mujoco import mjx

ACT_SCALE = 0.3
SLEW = 0.2                 # 每个控制步（20 ms）目标角最多变化的弧度
FRIC = (0.5, 1.5)
PUSH_PROB, PUSH_N = 0.002, 3.0     # 每步有 0.2% 的机会被推一下（3 N，持续这一步）


class GetUp(Env):
    def __init__(self, xml_path, ref_path, n_frames=10):
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
        jid = self.mj.actuator_trnid[:, 0]
        self.act_q = jnp.array(self.mj.jnt_qposadr[jid])
        self.act_v = jnp.array(self.mj.jnt_dofadr[jid])
        self.lo = jnp.array(self.mj.jnt_range[jid, 0], dtype=jnp.float32)
        self.hi = jnp.array(self.mj.jnt_range[jid, 1], dtype=jnp.float32)
        self.frc = jnp.array(self.mj.actuator_forcerange[:, 1], dtype=jnp.float32)
        self.pelvis = self.mj.body("pelvis").id
        self.chest = self.mj.body("chest").id
        q0 = jnp.array(self.mj.qpos0, dtype=jnp.float32)
        q0 = q0.at[0:3].set(jnp.array([0.0, 0.0, 0.06]))
        self.q0 = q0.at[3:7].set(jnp.array([math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0]))   # 脸朝下趴着
        self.fric0 = self.mx.geom_friction

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
        return d.qpos[2], R[2, 2]

    def reset(self, rng):
        rng, k1, k2 = jax.random.split(rng, 3)
        q = self.q0.at[self.act_q].add(jax.random.uniform(k1, (self.mj.nu,), minval=-0.05, maxval=0.05))
        d = mjx.make_data(self.mx).replace(qpos=q)
        d = mjx.forward(self.mx, d)
        zero = jnp.zeros(self.mj.nu, dtype=jnp.float32)
        t = jnp.zeros((), dtype=jnp.int32)
        info = dict(t=t, target=zero, last_act=zero, rng=rng,
                    fric=jax.random.uniform(k2, (), minval=FRIC[0], maxval=FRIC[1]))
        z, up = self._measure(d)
        metrics = dict(height=z, up=up, success=jnp.zeros(()), track=jnp.zeros(()))
        return State(d, self._obs(d, zero, t), jnp.zeros(()), jnp.zeros(()), metrics, info)

    def step(self, state, action):
        info = state.info
        t = info["steps"].astype(jnp.int32) if "steps" in info else info["t"]      # 训练时由包装器计步（自动重开时会清零）
        first = t == 0
        rng, k1, k2, k3 = jax.random.split(info["rng"], 4)
        fric = jnp.where(first, jax.random.uniform(k1, (), minval=FRIC[0], maxval=FRIC[1]), info["fric"])
        prev = jnp.where(first, jnp.zeros_like(info["target"]), info["target"])
        last_act = jnp.where(first, jnp.zeros_like(info["last_act"]), info["last_act"])
        action = jnp.clip(action, -1.0, 1.0)
        ref = self.ref[jnp.clip(t, 0, self.T - 1)]
        want = jnp.clip(ref + ACT_SCALE * action, self.lo, self.hi)
        target = prev + jnp.clip(want - prev, -SLEW, SLEW)
        mx = self.mx.replace(geom_friction=self.fric0 * jnp.array([fric, 1.0, 1.0]))
        push = jnp.where(jax.random.uniform(k2, ()) < PUSH_PROB, 1.0, 0.0) * PUSH_N * jax.random.uniform(k3, (2,), minval=-1.0, maxval=1.0)
        xfrc = jnp.zeros_like(state.pipeline_state.xfrc_applied).at[self.chest, 0:2].set(push)
        d = state.pipeline_state.replace(ctrl=target, xfrc_applied=xfrc)
        d = jax.lax.fori_loop(0, self.n_frames, lambda _, x: mjx.step(mx, x), d)
        z, up = self._measure(d)
        q = d.qpos[self.act_q]
        track = jnp.exp(-2.0 * jnp.mean((q - ref) ** 2))
        late = (t >= self.t_free).astype(jnp.float32)                 # 松手以后才要求站直
        gate = 0.2 + 0.8 * late
        stand = gate * (jnp.clip(z / 0.28, 0.0, 1.0) + 0.5 * (up + 1.0))
        success = ((z > 0.20) & (up > math.cos(math.radians(20.0)))).astype(jnp.float32)
        effort = jnp.mean((d.actuator_force / self.frc) ** 2)
        smooth = jnp.mean((action - last_act) ** 2)
        reward = 0.5 * track + stand + 3.0 * late * success - 0.05 * effort - 0.05 * smooth
        bad = jnp.any(jnp.isnan(d.qpos)) | jnp.any(jnp.isnan(d.qvel))
        reward = jnp.where(bad, 0.0, reward) * 0.02
        t2 = t + 1
        info = dict(info)
        info.update(t=t2, target=target, last_act=action, rng=rng, fric=fric)
        metrics = dict(height=z, up=up, success=success * late, track=track)
        return state.replace(pipeline_state=d, obs=self._obs(d, action, t2), reward=reward, done=bad.astype(jnp.float32),
                             metrics=metrics, info=info)
