"""
强化学习：从单膝跪地站起来（花然起身的最后一步）

用法：
    GETUP_METHOD=kneel3 python rl_rise.py            # 训练，结果存在 rl_rise_policy.json
    GETUP_METHOD=kneel3 python rl_rise.py --eval     # 只评估已有策略

背景（2026-11-12）：单膝跪地起身已经能俯卧撑、跪直、移重心、跨腿（getup.py，kneel3），
只在"从单膝跪地站起来"这一下往后倒。这一下只有一个膝盖和一只脚着地，需要边动边修正平衡，
按预先排好的姿势走的方法到头了。这里训练一个**反馈策略**：每 10 ms 看一次身体（倾斜、角速度、
重心相对脚的位置、关节角度和速度），在原来的动作上加一点修正。

方法：ARS（增强随机搜索，Mania 等 2018）。线性策略 + 观测归一化；每一轮在策略参数上加若干随机扰动，
各跑一次，按得分好坏的差值更新参数。不需要 GPU，4 个 CPU 核并行。
起点：kneel3 动作走到"单膝跪地"那一刻的仿真状态（加一点随机扰动，让策略学会应付不同的晃动）。
扭矩上限和 10 rad/s 转速上限都和其他测试一样。
"""
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import mujoco
import numpy as np

import getup
from balance import Balance
from figure_test import stand_pose

HERE = os.path.dirname(os.path.abspath(__file__))
CTRL = ["l_hip_pitch", "r_hip_pitch", "l_hip_roll", "r_hip_roll", "l_knee", "r_knee",
        "l_ankle_pitch", "r_ankle_pitch", "l_ankle_roll", "r_ankle_roll", "waist_pitch",
        "l_shoulder_pitch", "r_shoulder_pitch"]
ACT_SCALE = 0.5           # 每个关节最多修正 ±0.5 rad
POLICY_DT = 0.01          # 每 10 ms 决策一次
HOLD = 1.5                # 站起后还要站稳 1.5 s
W_MAX = 10.0              # 转速上限 rad/s
_ENV = None


class Env:
    def __init__(self):
        self.sim = getup.get_sim()
        self.m, self.d = self.sim.m, self.sim.d
        best = json.load(open(os.path.join(HERE, "getup_best_kneel3.json"), encoding="utf-8"))["x"]
        self.ks = getup.keyframes_for(best)
        # 第二版（2026-11-12）：上一版的起点其实不是真正的单膝跪地（右脚在左膝后面）。
        # 改为从"跪直"那一刻开始（两膝着地、上身直立，已验证稳定），用按人的做法重新设计的单膝跪地动作：
        # 右大腿抬到水平、小腿垂直，脚平放在左膝前方；上身前倾、胳膊往前送来配重；再站起来。
        kneel_t, kneel = self.ks[-5]
        self.t_rise = kneel_t
        half = dict(kneel)
        lift = dict(kneel)
        lift.update(r_hip_pitch=-1.5, r_knee=1.7, r_ankle_pitch=-0.2)
        half.update(r_hip_pitch=-1.45, r_knee=1.5, r_ankle_pitch=0.0, waist_pitch=0.3,
                    l_shoulder_pitch=-1.2, r_shoulder_pitch=-1.2, l_elbow=-0.2, r_elbow=-0.2)
        t = kneel_t
        self.ks = [kp for kp in self.ks if kp[0] <= kneel_t]
        for dt_, p in ((0.9, lift), (0.9, half), (1.6, stand_pose())):
            t += dt_
            self.ks.append((t, p))
        self.T5 = self.ks[-1][0] - self.t_rise
        self.bal = Balance(self.sim)
        self.state0, self.targets0 = self._capture()
        self.nq = len(CTRL)
        self.obs_dim = 3 + 3 + 1 + 2 + 2 * self.nq + 1

    def _target(self, t):
        ks = self.ks
        for i in range(1, len(ks)):
            t0, p0 = ks[i - 1]
            t1, p1 = ks[i]
            if t <= t1:
                a = getup.sm((t - t0) / (t1 - t0))
                return {k: p0.get(k, 0.0) + (p1.get(k, 0.0) - p0.get(k, 0.0)) * a for k in set(p0) | set(p1)}
        return ks[-1][1]

    def _capture(self):
        """按 kneel3 动作走到单膝跪地那一刻，存下仿真状态。"""
        sim, m, d = self.sim, self.m, self.d
        sim.reset()
        d.qpos[0:3] = [0, 0, 0.06]
        d.qpos[3:7] = [math.cos(math.pi / 4), 0, math.sin(math.pi / 4), 0]
        mujoco.mj_forward(m, d)
        neck = m.actuator("neck_pitch").id
        self.neck_base = m.actuator_forcerange[neck].copy()
        dt = m.opt.timestep
        last = None
        for k in range(int(self.t_rise / dt)):
            t = k * dt
            m.actuator_forcerange[neck] = self.neck_base * 0.05
            tgt = self._target(t)
            sim.set_targets(tgt)
            sim.step()
            last = tgt
        size = mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_INTEGRATION)
        st = np.zeros(size)
        mujoco.mj_getState(m, d, st, mujoco.mjtState.mjSTATE_INTEGRATION)
        return st, last

    def jq(self, n):
        return float(self.d.qpos[self.m.jnt_qposadr[self.m.joint(n).id]])

    def jv(self, n):
        return float(self.d.qvel[self.m.jnt_dofadr[self.m.joint(n).id]])

    def obs(self, t):
        m, d = self.m, self.d
        R = d.xmat[m.body("pelvis").id].reshape(3, 3)
        g = R.T @ np.array([0, 0, -1.0])                    # 重力在骨盆坐标里的方向
        w = d.qvel[3:6]
        z = d.qpos[2]
        com = d.subtree_com[m.body("pelvis").id]
        feet = 0.5 * (d.xpos[m.body("l_foot").id] + d.xpos[m.body("r_foot").id])
        rel = com[:2] - feet[:2]
        q = [self.jq(n) for n in CTRL]
        v = [self.jv(n) for n in CTRL]
        return np.concatenate([g, w, [z], rel, q, v, [t / (self.T5 + HOLD)]])

    def rollout(self, W, mean, std, noise=0.0, seed=0, record=None):
        """跑一次。返回总得分、是否站起来、骨盆高、倾斜。"""
        rng = np.random.default_rng(seed)
        sim, m, d = self.sim, self.m, self.d
        mujoco.mj_setState(m, d, self.state0, mujoco.mjtState.mjSTATE_INTEGRATION)
        if noise:
            d.qvel[:] += rng.normal(0, noise, size=d.qvel.shape)
        mujoco.mj_forward(m, d)
        neck = m.actuator("neck_pitch").id
        m.actuator_forcerange[neck] = self.neck_base
        self.bal.prev = None
        dt = m.opt.timestep
        n_sub = int(round(POLICY_DT / dt))
        T = self.T5 + HOLD
        last = dict(self.targets0)
        total, obs_log = 0.0, []
        a = np.zeros(self.nq)
        balancing = False
        for k in range(int(T / dt)):
            t = k * dt
            if k % n_sub == 0:
                o = self.obs(t)
                obs_log.append(o)
                a = np.clip(W @ ((o - mean) / std), -1, 1) * ACT_SCALE
            tgt = self._target(self.t_rise + t)
            if not balancing and sim.tilt_deg() < 35:
                balancing = True
            if balancing:
                tgt = self.bal.adjust(tgt)
            for i, n in enumerate(CTRL):
                tgt[n] = tgt.get(n, 0.0) + a[i]
            for n, v in list(tgt.items()):        # 转速上限
                if n.startswith("hand_"):
                    continue
                p = last.get(n, v)
                tgt[n] = p + max(-W_MAX * dt, min(W_MAX * dt, v - p))
                last[n] = tgt[n]
            sim.set_targets(tgt)
            sim.step()
            if record:
                record(t)
            if k % n_sub == 0:
                z = float(d.qpos[2])
                up = math.cos(math.radians(min(sim.tilt_deg(), 90)))
                total += (min(z, 0.26) / 0.26 + up) * POLICY_DT
                if z < 0.05:
                    total -= 2.0
                    break
        z, tilt = float(d.qpos[2]), sim.tilt_deg()
        ok = z > 0.20 and tilt < 20
        total += (3.0 if ok else 0.0) + min(z, 0.26) / 0.26 + math.cos(math.radians(min(tilt, 90)))
        return total, ok, z, tilt, obs_log


def _get_env():
    global _ENV
    if _ENV is None:
        _ENV = Env()
    return _ENV


def worker(args):
    W, mean, std, seed = args
    env = _get_env()
    r, ok, z, tilt, obs = env.rollout(W, mean, std, noise=0.15, seed=seed)
    return r, ok, z, tilt, np.array(obs)


def train(iters=40, ndir=12, top=6, step=0.03, nu=0.05, seed=0):
    env = _get_env()
    rng = np.random.default_rng(seed)
    W = np.zeros((env.nq, env.obs_dim))
    mean = np.zeros(env.obs_dim)
    std = np.ones(env.obs_dim)
    obs_all = []
    hist = []
    t0 = time.time()
    with Pool(4) as pool:
        base = pool.map(worker, [(W, mean, std, 1000 + i) for i in range(4)])
        print(f"策略为零（只用原来的动作）：站起来 {sum(b[1] for b in base)}/4，平均得分 {np.mean([b[0] for b in base]):.2f}", flush=True)
        for b in base:
            obs_all.append(b[4])
        allo = np.concatenate(obs_all)
        mean, std = allo.mean(0), allo.std(0) + 1e-3
        for it in range(iters):
            deltas = rng.normal(size=(ndir, *W.shape))
            jobs = []
            for i in range(ndir):
                s = int(rng.integers(1e9))
                jobs.append((W + nu * deltas[i], mean, std, s))
                jobs.append((W - nu * deltas[i], mean, std, s))
            res = pool.map(worker, jobs)
            rp = np.array([res[2 * i][0] for i in range(ndir)])
            rm = np.array([res[2 * i + 1][0] for i in range(ndir)])
            order = np.argsort(-np.maximum(rp, rm))[:top]
            sig = np.concatenate([rp[order], rm[order]]).std() + 1e-6
            W = W + step / (top * sig) * sum((rp[i] - rm[i]) * deltas[i] for i in order)
            for r in res:
                obs_all.append(r[4])
            allo = np.concatenate(obs_all[-200:])
            mean, std = allo.mean(0), allo.std(0) + 1e-3
            okrate = np.mean([r[1] for r in res])
            hist.append(dict(it=it + 1, mean_reward=float(np.mean(rp + rm) / 2), ok_rate=float(okrate)))
            print(f"第 {it+1:2d} 轮：平均得分 {np.mean(rp+rm)/2:.2f}，带扰动的尝试里站起来的比例 {okrate*100:.0f}%（{time.time()-t0:.0f} s）", flush=True)
            json.dump(dict(W=W.tolist(), mean=mean.tolist(), std=std.tolist(), hist=hist, ctrl=CTRL, act_scale=ACT_SCALE, version=2),
                      open(os.path.join(HERE, "rl_rise_policy.json"), "w"), ensure_ascii=False)
    return W, mean, std, hist


def evaluate(n=8):
    pol = json.load(open(os.path.join(HERE, "rl_rise_policy.json"), encoding="utf-8"))
    W, mean, std = np.array(pol["W"]), np.array(pol["mean"]), np.array(pol["std"])
    with Pool(4) as pool:
        zero = pool.map(worker, [(np.zeros_like(W), mean, std, 500 + i) for i in range(n)])
        res = pool.map(worker, [(W, mean, std, 500 + i) for i in range(n)])
    print(f"原来的动作（不修正）：站起来 {sum(r[1] for r in zero)}/{n}")
    print(f"加上学到的反馈策略：站起来 {sum(r[1] for r in res)}/{n}")
    return sum(r[1] for r in zero), sum(r[1] for r in res), n


if __name__ == "__main__":
    if "--eval" in sys.argv:
        evaluate()
    elif "--time" in sys.argv:
        env = _get_env()
        t = time.time()
        r = env.rollout(np.zeros((env.nq, env.obs_dim)), np.zeros(env.obs_dim), np.ones(env.obs_dim))
        print(f"一次尝试 {time.time()-t:.1f} s（仿真 {env.T5 + HOLD:.1f} s）：站起来 {r[1]}，骨盆 {r[2]*100:.1f} cm，倾斜 {r[3]:.0f}°")
    else:
        iters = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--iters=")), 40))
        train(iters=iters)
        evaluate()
