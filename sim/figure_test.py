"""
55 cm 人偶机器人 · 物理验证脚本

用法：
    pip install mujoco
    python figure_test.py           # 只跑测试并打印结果
    python figure_test.py --view    # 同时打开 3D 窗口观看

测试顺序：
    1. 站立 3 秒：能不能站稳，各关节需要多大扭矩
    2. 深蹲：膝从 0.3 rad 弯到 1.6 rad，在最低点停 2 秒，再站起来
    3. 推一下：站立时在胸口水平推 0.1 秒，看能不能站住
       （这里只有关节 PD 控制，还没有平衡控制器，所以推倒了也是正常的）

结果会对照 docs/knee-joint-v1.md 里的目标：
    膝关节持续扭矩 ≥ 1.2 N·m，峰值 ≥ 3 N·m；设计估算为持续约 1.35、峰值约 2.8
"""
import argparse
import math
import os
import time

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "figure55.xml")

KNEE_CONT_TARGET = 1.2   # N·m，持续扭矩需求
KNEE_PEAK_TARGET = 3.0   # N·m，峰值扭矩需求
KNEE_CONT_DESIGN = 1.35  # N·m，膝关节 v1 的持续扭矩估算


def stand_pose(knee=0.3):
    """脚掌放平、躯干直立的蹲姿：髋 = 踝 = -膝/2，髋正好在踝的正上方。"""
    p = {}
    for s in ("l", "r"):
        p[f"{s}_hip_pitch"] = -knee / 2
        p[f"{s}_knee"] = knee
        p[f"{s}_ankle_pitch"] = -knee / 2
    p["l_shoulder_roll"] = 0.15
    p["r_shoulder_roll"] = -0.15
    p["l_elbow"] = p["r_elbow"] = -0.2
    return p


def squat_knee(t):
    """深蹲轨迹：0–3 s 下蹲，3–5 s 停在最低点，5–8 s 站起。"""
    lo, hi = 0.3, 1.6
    if t < 3:
        a = t / 3
    elif t < 5:
        a = 1.0
    elif t < 8:
        a = 1 - (t - 5) / 3
    else:
        a = 0.0
    a = 0.5 - 0.5 * math.cos(math.pi * a)  # 平滑起止
    return lo + (hi - lo) * a


class Sim:
    def __init__(self, view=False):
        self.m = mujoco.MjModel.from_xml_path(MODEL)
        self.d = mujoco.MjData(self.m)
        self.act = {self.m.actuator(i).name: i for i in range(self.m.nu)}
        self.limit = {n: float(self.m.actuator_forcerange[i][1]) for n, i in self.act.items()}
        self.view = None
        if view:
            import mujoco.viewer
            self.view = mujoco.viewer.launch_passive(self.m, self.d)
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.m, self.d)
        pose = stand_pose()
        for name, val in pose.items():
            jid = self.m.joint(name).id
            self.d.qpos[self.m.jnt_qposadr[jid]] = val
        self.d.qpos[2] = 0.285 - 2 * 0.115 * (1 - math.cos(0.15))
        self.set_targets(pose)
        mujoco.mj_forward(self.m, self.d)

    def set_targets(self, pose):
        self.d.ctrl[:] = 0
        for name, val in pose.items():
            if name in self.act:
                self.d.ctrl[self.act[name]] = val

    def step(self, n=1):
        for _ in range(n):
            mujoco.mj_step(self.m, self.d)
        if self.view is not None:
            self.view.sync()
            time.sleep(self.m.opt.timestep * n)

    def torques(self):
        return {n: float(self.d.actuator_force[i]) for n, i in self.act.items()}

    def fallen(self):
        z = self.d.qpos[2]
        w, x, y, _ = self.d.qpos[3:7]
        up_z = 1 - 2 * (x * x + y * y)  # 躯干朝上方向的 z 分量
        return z < 0.12 or up_z < math.cos(0.6)

    def tilt_deg(self):
        w, x, y, _ = self.d.qpos[3:7]
        up_z = max(-1.0, min(1.0, 1 - 2 * (x * x + y * y)))
        return math.degrees(math.acos(up_z))


def run_phase(sim, seconds, pose_fn, push=None):
    """运行一段时间，返回每个执行器的最大 |扭矩|、扭矩时间序列（膝）和是否摔倒。"""
    dt = sim.m.opt.timestep
    steps = int(seconds / dt)
    peak = {n: 0.0 for n in sim.act}
    knee_series = []
    fell_at = None
    chest = sim.m.body("chest").id
    for k in range(steps):
        t = k * dt
        sim.set_targets(pose_fn(t))
        sim.d.xfrc_applied[chest] = 0
        if push and push[0] <= t < push[0] + push[1]:
            sim.d.xfrc_applied[chest][0] = push[2]
        sim.step()
        tq = sim.torques()
        for n, v in tq.items():
            peak[n] = max(peak[n], abs(v))
        knee_series.append((t, abs(tq["l_knee"]), abs(tq["r_knee"])))
        if fell_at is None and sim.fallen():
            fell_at = t
    return peak, knee_series, fell_at


def report_peaks(sim, peak, names):
    print(f"  {'关节':<16}{'最大扭矩':>10}{'上限':>8}{'占用':>8}")
    for n in names:
        lim = sim.limit[n]
        use = peak[n] / lim * 100
        flag = "  ← 顶到上限" if use >= 98 else ("  ← 接近上限" if use >= 80 else "")
        print(f"  {n:<16}{peak[n]:>9.2f}{lim:>8.1f}{use:>7.0f}%{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", action="store_true", help="打开 3D 窗口观看")
    args = ap.parse_args()

    sim = Sim(view=args.view)
    total = float(sum(sim.m.body_mass))
    print(f"模型：{os.path.basename(MODEL)}  总质量 {total:.2f} kg，关节 {sim.m.nu} 个\n")
    legs = [f"{s}_{j}" for s in ("l", "r") for j in ("hip_pitch", "knee", "ankle_pitch", "hip_roll")]

    # 1. 站立
    print("【1】站立 3 秒")
    peak, ks, fell = run_phase(sim, 3.0, lambda t: stand_pose())
    steady = sim.torques()
    print(f"  结果：{'摔倒（第 %.2f 秒）' % fell if fell is not None else '站稳'}，"
          f"骨盆高度 {sim.d.qpos[2]*100:.1f} cm，躯干倾斜 {sim.tilt_deg():.1f}°")
    print(f"  稳定后膝关节扭矩：左 {abs(steady['l_knee']):.2f} / 右 {abs(steady['r_knee']):.2f} N·m")
    report_peaks(sim, peak, legs)

    # 2. 深蹲
    print("\n【2】深蹲（膝 0.3 → 1.6 rad，最低点停 2 秒）")
    sim.reset()
    run_phase(sim, 1.0, lambda t: stand_pose())
    peak, ks, fell = run_phase(sim, 9.0, lambda t: stand_pose(squat_knee(t)))
    hold = [max(l, r) for t, l, r in ks if 3.5 <= t < 5.0]
    hold_avg = sum(hold) / len(hold)
    knee_peak = max(peak["l_knee"], peak["r_knee"])
    print(f"  结果：{'摔倒（第 %.2f 秒）' % fell if fell is not None else '完成，没有摔倒'}")
    print(f"  最低点保持时膝关节平均扭矩：{hold_avg:.2f} N·m（双腿分担）")
    report_peaks(sim, peak, legs)

    # 3. 推一下
    print("\n【3】站立时在胸口水平推 0.1 秒")
    for force in (1.0, 2.0, 4.0):
        sim.reset()
        _, _, fell = run_phase(sim, 3.0, lambda t: stand_pose(), push=(1.0, 0.1, force))
        print(f"  推力 {force:.0f} N：{'摔倒' if fell is not None else '站住了'}")

    # 对照膝关节 v1
    print("\n【对照 docs/knee-joint-v1.md】")
    print(f"  深蹲保持所需扭矩 {hold_avg:.2f} N·m  vs 持续扭矩需求 {KNEE_CONT_TARGET} / 设计估算 {KNEE_CONT_DESIGN}"
          f"  → {'够用' if hold_avg <= KNEE_CONT_DESIGN else '不够，需要加大减速比或电机'}")
    print(f"  深蹲过程峰值扭矩 {knee_peak:.2f} N·m  vs 模型上限 {sim.limit['l_knee']:.1f}"
          f"  → {'有余量' if knee_peak < 0.8 * sim.limit['l_knee'] else '余量不足'}")
    print("  说明：这里只有双腿深蹲。单腿支撑、起身和跌倒后爬起需要的扭矩会更大，")
    print(f"        所以 {KNEE_PEAK_TARGET} N·m 的峰值目标仍然需要台架实测来确认。")

    if sim.view is not None:
        print("\n测试结束。关闭窗口即可退出。")
        while sim.view.is_running():
            sim.set_targets(stand_pose(squat_knee(sim.d.time % 9.0)))
            sim.step()


if __name__ == "__main__":
    main()
