"""
左右镜像（2026-11-12，"头偏向一边"）：把一个姿势、一条参考动作或一份观测按身体的正中面镜像。

身体是左右对称的（完整身体在 qpos0 时，每对 l_/r_ 关节的转轴和位置互为镜像，误差 0；见 diag_out/turn/mir1.py）。
镜像以后：l_ 和 r_ 对调；绕前后轴（roll）和上下轴（yaw）转的关节角取反，绕左右轴（pitch、膝、肘）转的不变。
这样学会"转左"的策略可以直接用来"转右"：把真实的观测镜像给它看，它给出的动作再镜像回来。

观测的排法和 getup_env.py 一样（113 个数）：
  重力在骨盆坐标里 3、骨盆角速度 3、关节角 26、关节角速度 26、上一步动作 26、参考 26、相位 3。
骨盆坐标在 qpos0 时和世界一致，左右轴是 y：重力取 (x, -y, z)；角速度是"轴向量"，取 (-x, y, -z)。
"""
import numpy as np

NAMES = ['waist_yaw', 'waist_pitch', 'neck_yaw', 'neck_pitch', 'neck_low', 'neck_roll',
         'l_shoulder_pitch', 'l_shoulder_roll', 'l_shoulder_yaw', 'l_elbow', 'r_shoulder_pitch', 'r_shoulder_roll', 'r_shoulder_yaw', 'r_elbow',
         'l_hip_yaw', 'l_hip_roll', 'l_hip_pitch', 'l_knee', 'l_ankle_pitch', 'l_ankle_roll',
         'r_hip_yaw', 'r_hip_roll', 'r_hip_pitch', 'r_knee', 'r_ankle_pitch', 'r_ankle_roll']
FLIP = ("_yaw", "_roll")          # 名字以这些结尾的关节，镜像时取反


def partner(n):
    if n.startswith("l_"):
        return "r_" + n[2:]
    if n.startswith("r_"):
        return "l_" + n[2:]
    return n


class Mirror:
    def __init__(self, names=NAMES):
        self.names = list(names)
        self.perm = np.array([self.names.index(partner(n)) for n in self.names])
        self.sign = np.array([-1.0 if n.endswith(FLIP) else 1.0 for n in self.names])
        n = len(self.names)
        self.n = n

    def q(self, x):
        """关节角 / 角速度 / 动作 / 参考：一组 26 个数。"""
        x = np.asarray(x, dtype=float)
        return self.sign * x[..., self.perm]

    def obs(self, o):
        o = np.asarray(o, dtype=float)
        n = self.n
        g = o[0:3] * np.array([1.0, -1.0, 1.0])
        w = o[3:6] * np.array([-1.0, 1.0, -1.0])
        parts = [g, w] + [self.q(o[6 + k * n: 6 + (k + 1) * n]) for k in range(4)] + [o[6 + 4 * n:]]
        return np.concatenate(parts)
