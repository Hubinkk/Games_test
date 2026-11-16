"""走路的训练环境（walk_env.py）和检验（eval_walk.py）共用的数字。只用标准库，两边的 Python 环境都能读。"""
import math
import os

LEGS = [f"{s}_{j}" for s in "lr" for j in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch", "ankle_roll")]
KNEE0 = 0.4                          # 站姿：膝弯 0.4 弧度，髋和踝各 -0.2
LEAN = float(os.environ.get("WALK_LEAN", "0.07"))   # 站姿从脚踝整个人往前倾这么多弧度：重心偏后，不前倾会往后倒
ACT_SCALE = 0.5                     # 目标角 = 站姿 + 0.5 弧度 × 策略的输出
RATE = 10.0                          # 转速上限 rad/s
DT = 0.02                            # 策略每 20 ms 出一次目标
GAIT_HZ = 2.0                        # 每秒几个完整的步周期（左右各迈一步算一个）
SWING = 0.4                          # 一个周期里每只脚抬着的比例
STEP_H = 0.025                       # 抬脚的目标高度（米）
FOOT_HALF = 0.006                    # 脚掌厚度的一半
CMD_SCALE = (5.0, 10.0, 1.0)         # 指令（前进 m/s、侧移 m/s、转 rad/s）放进观测时乘的系数
FALL_Z, FALL_DEG = 0.17, 40.0        # 骨盆低于 17 cm 或躯干歪过 40° 算摔


# 第 2 轮（2026-11-16）：上身不归走路的策略管，由别处指挥（以后是 VR 里育苗人的头和手）。
# 训练和检验时让上身在下面的范围里随意摆，腿要照样走稳。范围比关节的极限小一些。
UPPER = {
    "waist_yaw": (-0.5, 0.5), "waist_pitch": (-0.05, 0.2),
    "neck_yaw": (-1.0, 1.0), "neck_pitch": (-0.4, 0.4), "neck_low": (-0.3, 0.3), "neck_roll": (-0.3, 0.3),
    "l_shoulder_pitch": (-2.0, 0.6), "l_shoulder_roll": (-0.1, 1.5), "l_shoulder_yaw": (-1.0, 1.0), "l_elbow": (-2.0, 0.0),
    "r_shoulder_pitch": (-2.0, 0.6), "r_shoulder_roll": (-1.5, 0.1), "r_shoulder_yaw": (-1.0, 1.0), "r_elbow": (-2.0, 0.0),
}
UPPER_EVERY = 100                    # 每 2 秒换一次上身的目标姿势
UPPER_REST = 0.3                     # 三成的时候回到自然下垂


def default_pose():
    p = {}
    for s in "lr":
        p[f"{s}_hip_pitch"], p[f"{s}_knee"], p[f"{s}_ankle_pitch"] = -KNEE0 / 2, KNEE0, -KNEE0 / 2 - LEAN
    p["l_shoulder_roll"], p["r_shoulder_roll"] = 0.15, -0.15
    p["l_elbow"] = p["r_elbow"] = -0.2
    return p


def stand_height():
    return 0.285 - 2 * 0.115 * (1 - math.cos(KNEE0 / 2))
