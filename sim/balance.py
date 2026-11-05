"""
重心反馈平衡控制器（"脊髓层"的最简版本）

思路：实时测量全身重心（COM）相对支撑脚的位置和速度，
      微调髋、踝的目标角度，把重心拉回支撑区域上方。
  - 左右方向：同时调整两条腿的髋侧摆和踝侧摆（骨盆整体平移）
  - 前后方向：调整踝俯仰（踝策略）

这是手写的经典控制，不是学出来的策略。它的作用是给出一个基准：
之后用强化学习训练出的平衡，应该比它更好。
"""
import numpy as np


class Balance:
    def __init__(self, sim, kp_y=12.0, kd_y=0.8, kp_x=4.0, kd_x=0.4, sign_x=-1.0, roll_max=0.4):
        self.sim = sim
        m = sim.m
        self.pelvis = m.body("pelvis").id
        self.foot = {s: m.geom(f"{s}_foot").id for s in ("l", "r")}
        self.kp_y, self.kd_y, self.kp_x, self.kd_x, self.sign_x = kp_y, kd_y, kp_x, kd_x, sign_x
        self.roll_max = roll_max
        self.prev = None

    def com_state(self, dt):
        d = self.sim.d
        com = d.subtree_com[self.pelvis].copy()
        vel = np.zeros(3) if self.prev is None else (com - self.prev) / dt
        self.prev = com
        return com, vel

    def support(self, stance):
        d = self.sim.d
        ps = [d.geom_xpos[self.foot[s]] for s in stance]
        return np.mean(ps, axis=0)

    def adjust(self, pose, stance=("l", "r"), dt=0.002):
        com, vel = self.com_state(dt)
        sup = self.support(stance)
        ex, ey = sup[0] - com[0], sup[1] - com[1]
        # 左右：腿侧摆角为负时，骨盆向 +y 平移
        roll = -(self.kp_y * ey - self.kd_y * vel[1])
        roll = float(np.clip(roll, -self.roll_max, self.roll_max))
        for s in ("l", "r"):
            pose[f"{s}_hip_roll"] = pose.get(f"{s}_hip_roll", 0.0) + roll
            pose[f"{s}_ankle_roll"] = pose.get(f"{s}_ankle_roll", 0.0) - roll
        # 前后：踝策略
        dp = self.sign_x * (self.kp_x * ex - self.kd_x * vel[0])
        dp = float(np.clip(dp, -0.3, 0.3))
        for s in stance:
            pose[f"{s}_ankle_pitch"] = pose.get(f"{s}_ankle_pitch", 0.0) + dp
        return pose
