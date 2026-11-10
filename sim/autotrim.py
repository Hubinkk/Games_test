"""
自动配平（L1）：花然用自己的身体数据，把"身体变了"带来的重心偏移补回来。

它有两份身体数据：设计动作时的身体（参考，figure55_full_ref.xml）和现在的身体。
对马上要摆的姿势，在两份"草稿身体"里各算一次重心；现在的重心比参考偏多少，就补多少：
  用腰补（脚踝的修正写好了但关着：走路时会往后倒）。
每 20 ms 重算一次。只在身体大致直立时起作用（趴着、起身途中不管）。不用强化学习，不用传感器。测试见 trim.py。
"""
import mujoco
import numpy as np

EVERY = 0.02         # 每 20 ms 重算一次
MAX_ANKLE = 0.0      # 脚踝的修正先关掉：第三版身体上试过，走路时动脚踝会往后倒；只用腰则通过（2026-11-10）
MAX_TRIM = 0.25      # 腰最多修正这么多弧度（总的弯曲角）
UPRIGHT = 45.0       # 倾斜超过这个角度就不配平


class _Draft:
    """一份草稿身体：只用来"想"，不影响真的身体。"""

    def __init__(self, m):
        self.m, self.d = m, mujoco.MjData(m)
        self.adr = {m.joint(j).name: m.jnt_qposadr[j] for j in range(m.njnt) if m.jnt_type[j] == 3}
        self.feet = [m.geom(f"{s}_foot").id for s in ("l", "r")]
        self.pelvis = m.body("pelvis").id
        self.two = "waist_pitch2" in self.adr

    def com(self, pose, waist_extra=0.0):
        d = self.d
        d.qpos[:] = self.m.qpos0
        for k, v in pose.items():
            a = self.adr.get(k)
            if a is not None:
                d.qpos[a] = v
        w = pose.get("waist_pitch", 0.0) + waist_extra        # 姿势里写的是腰总的弯曲角
        if self.two:
            d.qpos[self.adr["waist_pitch"]] = d.qpos[self.adr["waist_pitch2"]] = w / 2
        else:
            d.qpos[self.adr["waist_pitch"]] = w
        mujoco.mj_kinematics(self.m, d)
        mujoco.mj_comPos(self.m, d)
        sup = np.mean([d.geom_xpos[g] for g in self.feet], axis=0)
        c = d.subtree_com[self.pelvis]
        return float(c[0] - sup[0]), float(c[2] - sup[2])

    def com_x(self, pose, waist_extra=0.0):
        return self.com(pose, waist_extra)[0]


class Trim:
    """包在 sim.set_targets 外面：每次给姿势时，顺便把重心修正回参考身体的位置。"""

    def __init__(self, sim, ref_path):
        self.sim = sim
        self.now = _Draft(sim.m)
        self.ref = _Draft(mujoco.MjModel.from_xml_path(ref_path))
        self.delta, self.ankle, self.t_last = 0.0, 0.0, -1.0
        self.enabled = True       # 起身这类"在现在的身体上设计的动作"自己关掉它：参考身体做不出那些姿势，比较没有意义
        self.history = []
        self.orig = sim.set_targets
        sim.set_targets = self.set_targets

    def solve(self, pose):
        x0, h = self.now.com(pose)
        need = self.ref.com_x(pose) - x0                   # 重心要往前挪多少米
        ankle = float(np.clip(need / max(h, 0.05), -MAX_ANKLE, MAX_ANKLE))
        need -= ankle * h
        k = (self.now.com_x(pose, 0.05) - x0) / 0.05       # 腰前弯 1 弧度，重心前移多少米
        waist = float(np.clip(need / k, -MAX_TRIM, MAX_TRIM)) if abs(k) > 1e-4 else 0.0
        return ankle, waist

    def set_targets(self, pose):
        t = self.sim.d.time
        if t - self.t_last >= EVERY or t < self.t_last:
            if self.enabled and self.sim.tilt_deg() < UPRIGHT:
                self.ankle, self.delta = self.solve({k: v for k, v in pose.items() if not k.startswith("hand_")})
            else:
                self.ankle, self.delta = 0.0, 0.0
            self.t_last = t
            self.history.append(self.delta)
        pose = dict(pose)
        pose["waist_pitch"] = pose.get("waist_pitch", 0.0) + self.delta
        for s in ("l", "r"):                                # 脚踝多勾一点 = 从脚踝起整体前倾
            pose[f"{s}_ankle_pitch"] = pose.get(f"{s}_ankle_pitch", 0.0) - self.ankle
        self.orig(pose)
