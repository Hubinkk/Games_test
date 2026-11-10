"""
自动配平（L1）：花然用自己的身体数据，把"身体变了"带来的重心偏移用腰补回来。

它有两份身体数据：设计动作时的身体（参考，figure55_full_ref.xml）和现在的身体。
对马上要摆的姿势，在两份"草稿身体"里各算一次重心；现在的重心比参考偏多少，就让腰弯多少把它补回来。
每 20 ms 重算一次。不用强化学习，不用传感器。测试见 trim.py。
"""
import mujoco
import numpy as np

EVERY = 0.02         # 每 20 ms 重算一次
MAX_TRIM = 0.25      # 腰最多修正这么多弧度


class _Draft:
    """一份草稿身体：只用来"想"，不影响真的身体。"""

    def __init__(self, m):
        self.m, self.d = m, mujoco.MjData(m)
        self.adr = {m.joint(j).name: m.jnt_qposadr[j] for j in range(m.njnt) if m.jnt_type[j] == 3}
        self.feet = [m.geom(f"{s}_foot").id for s in ("l", "r")]
        self.pelvis = m.body("pelvis").id

    def com_x(self, pose, waist_extra=0.0):
        d = self.d
        d.qpos[:] = self.m.qpos0
        for k, v in pose.items():
            a = self.adr.get(k)
            if a is not None:
                d.qpos[a] = v
        d.qpos[self.adr["waist_pitch"]] += waist_extra
        mujoco.mj_kinematics(self.m, d)
        mujoco.mj_comPos(self.m, d)
        sup = np.mean([d.geom_xpos[g] for g in self.feet], axis=0)
        return float(d.subtree_com[self.pelvis][0] - sup[0])


class Trim:
    """包在 sim.set_targets 外面：每次给姿势时，顺便把腰修正到重心回到参考身体的位置。"""

    def __init__(self, sim, ref_path):
        self.sim = sim
        self.now = _Draft(sim.m)
        self.ref = _Draft(mujoco.MjModel.from_xml_path(ref_path))
        self.delta, self.t_last = 0.0, -1.0
        self.history = []
        self.orig = sim.set_targets
        sim.set_targets = self.set_targets

    def solve(self, pose):
        x0 = self.now.com_x(pose)
        k = (self.now.com_x(pose, 0.05) - x0) / 0.05      # 腰前弯 1 弧度，重心前移多少米
        if abs(k) < 1e-4:
            return 0.0
        return float(np.clip((self.ref.com_x(pose) - x0) / k, -MAX_TRIM, MAX_TRIM))

    def set_targets(self, pose):
        t = self.sim.d.time
        if t - self.t_last >= EVERY or t < self.t_last:
            self.delta = self.solve({k: v for k, v in pose.items() if not k.startswith("hand_")})
            self.t_last = t
            self.history.append(self.delta)
        pose = dict(pose)
        pose["waist_pitch"] = pose.get("waist_pitch", 0.0) + self.delta
        self.orig(pose)
