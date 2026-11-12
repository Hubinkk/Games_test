"""
走路的强化学习环境（MJX，在显卡上同时跑几千个花然）。

和起身不同，这里没有参考动作。给它一个指令（往前多快、往侧面多快、转多快），它自己学怎么迈步：
    腿上 12 个电机的目标角 = 站姿 + 0.5 弧度 × 策略的输出
其余电机（腰、脖子、手臂）固定在站姿：手臂自然下垂。以后再放开。
沿用整体模型的扭矩上限；目标角按 10 rad/s 的转速上限追（每 2 ms 最多变 0.02 弧度，和完整身体里的做法一样）。

结构借自 MuJoCo Playground 里人形机器人"按指令速度行走"那一类环境（只借写法，数字按花然的尺寸重定）：
  - 一个固定节拍（每秒 2 步周期）告诉它"现在该哪只脚抬"；抬脚高度跟不跟得上节拍有奖励。指令为零时两脚都不抬。
  - 奖励：身体的速度跟不跟得上指令、躯干直不直、脚落地后滑不滑、动作抖不抖、费不费力。
  - 摔了（骨盆太低或躯干歪过 40°）这一局结束。

它能感觉到的（观测，只用实物上会有的）：重力方向和转动速度（IMU）、腿上 12 个关节的角度和速度、上一步自己的输出、指令、节拍。
训练时这些读数都加一点噪声。给"打分的网络"（只在训练时用）额外看真实速度、脚高、地面摩擦和被推的力。

每一局随机变：地面摩擦 0.5–1.7（所有接触面一起变）、每个电机的刚度 ±15%、起始姿势小偏差；
平均每 4 秒被推一次（胸口，水平方向，最大 6 N，持续 0.2 秒）；指令每 5 秒换一次。
"""
import math

import jax
import jax.numpy as jnp
import mujoco
from brax.envs.base import Env, State
from mujoco import mjx

# 站姿、动作幅度、节拍这些和检验脚本共用的数字都在 walk_common.py（两边必须一样）
from walk_common import (ACT_SCALE, CMD_SCALE, FALL_DEG, FALL_Z, FOOT_HALF, GAIT_HZ, LEGS, RATE, STEP_H, SWING,
                         default_pose, stand_height)

FRIC = (0.5, 1.7)
KP_RANGE = (0.85, 1.15)
PUSH_PROB, PUSH_MAX, PUSH_STEPS = 0.005, 6.0, 10
CMD_EVERY = 250                      # 每 5 秒换一次指令
VX, VY, WZ = (-0.10, 0.25), (-0.06, 0.06), (-0.8, 0.8)
NOISE = dict(grav=0.03, gyro=0.2, q=0.02, qd=1.0)


class Walk(Env):
    def __init__(self, xml_path, n_frames=10, noise=True):
        self.mj = mujoco.MjModel.from_xml_path(xml_path)
        self.mx = mjx.put_model(self.mj)
        self.n_frames = n_frames
        self.dt = float(self.mj.opt.timestep) * n_frames
        self.noise = noise
        names = [self.mj.actuator(i).name for i in range(self.mj.nu)]
        dp = default_pose()
        self.default = jnp.array([dp.get(n, 0.0) for n in names], dtype=jnp.float32)
        self.leg = jnp.array([names.index(n) for n in LEGS])
        jid = self.mj.actuator_trnid[:, 0]
        self.act_q = jnp.array(self.mj.jnt_qposadr[jid])
        self.lo = jnp.array(self.mj.jnt_range[jid, 0], dtype=jnp.float32)
        self.hi = jnp.array(self.mj.jnt_range[jid, 1], dtype=jnp.float32)
        self.leg_q = jnp.array(self.mj.jnt_qposadr[jid])[self.leg]
        self.leg_v = jnp.array(self.mj.jnt_dofadr[jid])[self.leg]
        self.leg_frc = jnp.array(self.mj.actuator_forcerange[:, 1], dtype=jnp.float32)[self.leg]
        self.leg_default = self.default[self.leg]
        # 希望尽量保持在站姿附近的关节：髋转、髋侧摆、踝侧摆（每条腿的第 0、1、5 个）
        self.pose_w = jnp.array([1.0, 1.0, 0.0, 0.0, 0.0, 1.0] * 2, dtype=jnp.float32)
        self.pelvis = self.mj.body("pelvis").id
        self.chest = self.mj.body("chest").id
        self.feet = jnp.array([self.mj.geom("l_foot").id, self.mj.geom("r_foot").id])
        self.q0 = jnp.array(self.mj.qpos0, dtype=jnp.float32)
        self.z0 = stand_height()
        self.fric0 = self.mx.geom_friction
        self.gain0, self.bias0 = self.mx.actuator_gainprm, self.mx.actuator_biasprm
        self.cmd_scale = jnp.array(CMD_SCALE, dtype=jnp.float32)

    # ---- brax 需要的属性 ----
    @property
    def observation_size(self):
        return {"state": (49,), "privileged_state": (49 + 24,)}

    @property
    def action_size(self):
        return len(LEGS)

    @property
    def backend(self):
        return "mjx"

    # ---- 小工具 ----
    def _cmd(self, key):
        u = jax.random.uniform(key, (5,))
        c = jnp.array([VX[0] + u[0] * (VX[1] - VX[0]), VY[0] + u[1] * (VY[1] - VY[0]), WZ[0] + u[2] * (WZ[1] - WZ[0])])
        c = jnp.where(u[3] < 0.30, c * jnp.array([1.0, 0.0, 0.0]), c)      # 三成只往前后走
        c = jnp.where(u[4] < 0.15, jnp.zeros(3), c)                        # 一成半站着不动
        return c.astype(jnp.float32)

    def _rz(self, u):
        """节拍 u（0–1）时这只脚该离地多高：前 SWING 这一段抬起再落下，其余时间踩在地上。"""
        s = jnp.clip(u / SWING, 0.0, 1.0)
        return STEP_H * 0.5 * (1.0 - jnp.cos(2.0 * jnp.pi * s)) * (u < SWING)

    def _phases(self, k, ph0):
        ph = ph0 + k.astype(jnp.float32) * self.dt * GAIT_HZ
        return jnp.array([ph % 1.0, (ph + 0.5) % 1.0])                     # 左脚、右脚

    def _sense(self, d):
        R = d.xmat[self.pelvis]
        yaw = jnp.arctan2(R[1, 0], R[0, 0])
        c, s = jnp.cos(yaw), jnp.sin(yaw)
        vx, vy = d.qvel[0], d.qvel[1]
        gyro = d.qvel[3:6]                                                 # 骨盆自己坐标里的转动速度
        return dict(grav=-R[2, :], up=R[2, 2], gyro=gyro, z=d.qpos[2], vz=d.qvel[2],
                    v=jnp.array([c * vx + s * vy, -s * vx + c * vy]),      # 朝着它面对的方向量的前进、侧移速度
                    wz=(R @ gyro)[2], q=d.qpos[self.leg_q], qd=d.qvel[self.leg_v],
                    foot=d.geom_xpos[self.feet])

    def _obs(self, s, last_act, cmd, phases, key, fric, kp, push):
        ang = 2.0 * jnp.pi * phases
        clock = jnp.concatenate([jnp.cos(ang), jnp.sin(ang)])
        cmd_o = cmd * self.cmd_scale
        clean = jnp.concatenate([s["grav"], s["gyro"] * 0.25, cmd_o, s["q"] - self.leg_default, s["qd"] * 0.05, last_act, clock])
        if self.noise:
            k1, k2, k3, k4 = jax.random.split(key, 4)
            n = lambda kk, shape, a: jax.random.uniform(kk, shape, minval=-a, maxval=a)
            noisy = jnp.concatenate([s["grav"] + n(k1, (3,), NOISE["grav"]), (s["gyro"] + n(k2, (3,), NOISE["gyro"])) * 0.25, cmd_o,
                                     s["q"] - self.leg_default + n(k3, (12,), NOISE["q"]), (s["qd"] + n(k4, (12,), NOISE["qd"])) * 0.05,
                                     last_act, clock])
        else:
            noisy = clean
        fz = s["foot"][:, 2] - FOOT_HALF
        priv = jnp.concatenate([clean, s["v"] * 5.0, jnp.array([s["vz"] * 5.0, s["wz"], (s["z"] - self.z0) * 10.0]),
                                fz * 20.0, (fz < 0.004).astype(jnp.float32), jnp.array([fric]), push / 5.0, kp - 1.0])
        return {"state": noisy, "privileged_state": priv}

    def reset(self, rng):
        rng, k1, k2, k3, k4, k5 = jax.random.split(rng, 6)
        q = self.q0.at[2].set(self.z0 + 0.002).at[self.act_q].set(self.default)
        q = q.at[self.leg_q].add(jax.random.uniform(k1, (12,), minval=-0.03, maxval=0.03))
        d = mjx.make_data(self.mx).replace(qpos=q)
        d = mjx.forward(self.mx, d)
        zero = jnp.zeros(12, dtype=jnp.float32)
        cmd = self._cmd(k2)
        ph0 = jax.random.uniform(k3, ())
        fric = jax.random.uniform(k4, (), minval=FRIC[0], maxval=FRIC[1])
        kp = jnp.ones(12, dtype=jnp.float32)
        push = jnp.zeros(2, dtype=jnp.float32)
        k0 = jnp.zeros((), dtype=jnp.int32)
        info = dict(k=k0, target=self.default, last_act=zero, rng=rng, fric=fric, kp=kp, cmd=cmd, ph0=ph0,
                    push=push, push_left=jnp.zeros((), dtype=jnp.int32))
        s = self._sense(d)
        obs = self._obs(s, zero, cmd, self._phases(k0, ph0), k5, fric, kp, push)
        zf = jnp.zeros(())
        metrics = dict(vx=zf, err=zf, werr=zf, fell=zf, height=s["z"], lift=zf)
        return State(d, obs, jnp.zeros(()), jnp.zeros(()), metrics, info)

    def step(self, state, action):
        info = state.info
        k = info["steps"].astype(jnp.int32) if "steps" in info else info["k"]      # 训练时由包装器计步（自动重开时会清零）
        first = k == 0
        rng, k1, k2, k3, k4, k5, k6, k7, k8 = jax.random.split(info["rng"], 9)
        # 这一局的随机量：一局的第一步重新抽
        fric = jnp.where(first, jax.random.uniform(k1, (), minval=FRIC[0], maxval=FRIC[1]), info["fric"])
        kp = jnp.where(first, jax.random.uniform(k2, (12,), minval=KP_RANGE[0], maxval=KP_RANGE[1]), info["kp"])
        ph0 = jnp.where(first, jax.random.uniform(k3, ()), info["ph0"])
        cmd = jnp.where(first | (k % CMD_EVERY == 0), self._cmd(k4), info["cmd"])
        prev = jnp.where(first, self.default, info["target"])
        last_act = jnp.where(first, jnp.zeros_like(info["last_act"]), info["last_act"])
        # 推：开始以后持续 PUSH_STEPS 步
        left = jnp.where(first, 0, info["push_left"])
        start = (left == 0) & (jax.random.uniform(k5, ()) < PUSH_PROB)
        u = jax.random.uniform(k6, (2,))
        new_push = PUSH_MAX * u[0] * jnp.array([jnp.cos(2 * jnp.pi * u[1]), jnp.sin(2 * jnp.pi * u[1])])
        push = jnp.where(start, new_push, jnp.where(left > 0, info["push"], jnp.zeros(2)))
        left = jnp.where(start, PUSH_STEPS - 1, jnp.maximum(left - 1, 0))

        action = jnp.clip(action, -1.0, 1.0)
        want = self.default.at[self.leg].add(ACT_SCALE * action)
        want = jnp.clip(want, self.lo, self.hi)
        g = jnp.ones(self.mj.nu, dtype=jnp.float32).at[self.leg].set(kp)
        mx = self.mx.replace(geom_friction=self.fric0 * jnp.array([fric, 1.0, 1.0]),
                             actuator_gainprm=self.gain0.at[:, 0].set(self.gain0[:, 0] * g),
                             actuator_biasprm=self.bias0.at[:, 1].set(self.bias0[:, 1] * g))
        d0 = state.pipeline_state
        foot0 = d0.geom_xpos[self.feet]
        xfrc = jnp.zeros_like(d0.xfrc_applied).at[self.chest, 0:2].set(push)
        d0 = d0.replace(xfrc_applied=xfrc)
        r = RATE * float(self.mj.opt.timestep)

        def sub(_, c):
            d, tgt = c
            tgt = tgt + jnp.clip(want - tgt, -r, r)
            return mjx.step(mx, d.replace(ctrl=tgt)), tgt

        d, target = jax.lax.fori_loop(0, self.n_frames, sub, (d0, prev))
        s = self._sense(d)
        k1n = k + 1
        phases = self._phases(k1n, ph0)
        moving = (jnp.sum(jnp.abs(cmd)) > 1e-4).astype(jnp.float32)
        # ---- 奖励 ----
        verr = jnp.sum((cmd[:2] - s["v"]) ** 2)
        werr = (cmd[2] - s["wz"]) ** 2
        r_lin = jnp.exp(-verr / 0.006)                       # 速度差 5 cm/s 时剩 0.66，差 10 cm/s 时剩 0.19
        r_ang = jnp.exp(-werr / 0.1)
        fz = s["foot"][:, 2] - FOOT_HALF
        want_z = self._rz(phases) * moving
        r_step = jnp.exp(-jnp.sum((fz - want_z) ** 2) / 0.0002)    # 一只脚差 1 cm 时剩 0.61
        contact = (fz < 0.004).astype(jnp.float32)
        fv = (s["foot"][:, :2] - foot0[:, :2]) / self.dt
        slip = jnp.sum(contact * jnp.sum(fv ** 2, axis=1))
        tilt = jnp.sum(s["grav"][:2] ** 2)
        wob = jnp.sum(s["gyro"][:2] ** 2)
        dz = (s["z"] - self.z0) ** 2
        rate = jnp.mean((action - last_act) ** 2)
        effort = jnp.mean((d.actuator_force[self.leg] / self.leg_frc) ** 2)
        dq = s["q"] - self.leg_default
        pose = jnp.sum(self.pose_w * dq ** 2)
        still = (1.0 - moving) * (jnp.mean(dq ** 2) + 0.01 * jnp.mean(s["qd"] ** 2))
        reward = (1.5 * r_lin + 0.8 * r_ang + 1.0 * r_step + 0.2
                  - 2.0 * tilt - 0.02 * wob - 100.0 * dz - 5.0 * slip - 0.05 * rate - 0.02 * effort - 0.3 * pose - 1.0 * still)
        bad = jnp.any(jnp.isnan(d.qpos)) | jnp.any(jnp.isnan(d.qvel))
        fell = (s["z"] < FALL_Z) | (s["up"] < math.cos(math.radians(FALL_DEG)))
        done = bad | fell
        reward = jnp.where(done, 0.0, jnp.clip(reward, 0.0, 10.0)) * self.dt
        info = dict(info)
        info.update(k=k1n, target=target, last_act=action, rng=rng, fric=fric, kp=kp, cmd=cmd, ph0=ph0, push=push, push_left=left)
        metrics = dict(state.metrics)          # 训练库的包装器会往里加自己的项，不能整个换掉
        metrics.update(vx=s["v"][0], err=jnp.sqrt(verr), werr=jnp.sqrt(werr), fell=fell.astype(jnp.float32), height=s["z"],
                       lift=jnp.max(fz))
        obs = self._obs(s, action, cmd, phases, k7, fric, kp, push)
        return state.replace(pipeline_state=d, obs=obs, reward=reward, done=done.astype(jnp.float32), metrics=metrics, info=info)
