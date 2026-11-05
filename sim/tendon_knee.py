"""
绳驱膝关节测试：把膝关节改成"绳轮＋肌腱线"驱动，测线上要承受多大的拉力

用法：
    python tendon_knee.py

结构（每条腿）：
    - 大腿上端有一个绳轮（电机藏在中空的大腿骨里）
    - 同一个绳轮上反向绕两根线：
        前线 = 股四头肌：经过膝盖前方，拉动小腿伸直
        后线 = 腘绳肌：经过膝盖后方，拉动小腿弯曲
    - 膝盖处有一个半径 1 cm 的滑轮（相当于髌骨），线绕着它走
    - 两根线都有预紧力，所以关节不会松垮

控制方法：
    先用和原模型相同的 PD 增益算出膝关节需要的扭矩，
    再换算成两根线的拉力（只能拉，不能推），交给 MuJoCo 的肌腱模型去执行。
    所以力臂随角度的变化、线绕滑轮的几何关系，都是 MuJoCo 实际计算出来的。
"""
import os

import mujoco
import numpy as np

import figure_test
from balance import Balance
from figure_test import Sim, run_phase, squat_knee, stand_pose

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "figure55.xml")
DST = os.path.join(HERE, "figure55_tendon.xml")

PULLEY_R = 0.010   # 膝部滑轮半径（m），即肌腱力臂
PRETENSION = 10.0  # 每根线的预紧力（N）
TMAX = 400.0       # 仿真里允许的最大拉力（N），只用来防止数值失控
KNEE_KP, KNEE_KV, KNEE_LIMIT = 20.0, 0.5, 2.8


def build():
    s = open(SRC, encoding="utf-8").read()
    for side in ("l", "r"):
        # 大腿：绳轮出线点（前、后）
        a = f'<body name="{side}_shin" pos="0 0 -0.115">'
        thigh_sites = (f'<site name="{side}_spool_f" pos="0.012 0 -0.02" size="0.003"/>\n'
                       f'            <site name="{side}_spool_b" pos="-0.012 0 -0.02" size="0.003"/>\n'
                       f'            <site name="{side}_side_f" pos="0.03 0 -0.115" size="0.002"/>\n'
                       f'            <site name="{side}_side_b" pos="-0.03 0 -0.115" size="0.002"/>\n            ')
        assert s.count(a) == 1
        s = s.replace(a, thigh_sites + a)
        # 小腿：膝部滑轮 + 肌腱止点
        b = f'<joint name="{side}_knee" axis="0 1 0" range="0 2.5" class="big"/>'
        shin = (b + f'\n              <geom name="{side}_pulley" type="cylinder" size="{PULLEY_R} 0.008" euler="1.5708 0 0"'
                f' contype="0" conaffinity="0" density="0" rgba="0.82 0.6 0.15 1"/>'
                f'\n              <site name="{side}_ins_f" pos="0.011 0 -0.03" size="0.003"/>'
                f'\n              <site name="{side}_ins_b" pos="-0.011 0 -0.03" size="0.003"/>')
        assert s.count(b) == 1
        s = s.replace(b, shin)
        # 去掉膝关节原来的位置伺服
        c = f'<position name="{side}_knee" joint="{side}_knee" kp="20" kv="0.5" forcerange="-2.8 2.8"/>'
        assert s.count(c) == 1
        s = s.replace(c, f'<motor name="{side}_quad" tendon="{side}_quad" ctrlrange="-{TMAX} 0"/>\n'
                         f'    <motor name="{side}_ham" tendon="{side}_ham" ctrlrange="-{TMAX} 0"/>')
    tendons = "  <tendon>\n"
    for side in ("l", "r"):
        for t, sp, ins, sd in (("quad", "spool_f", "ins_f", "side_f"), ("ham", "spool_b", "ins_b", "side_b")):
            tendons += (f'    <spatial name="{side}_{t}" width="0.0015" rgba="0.2 0.5 0.9 1">\n'
                        f'      <site site="{side}_{sp}"/><geom geom="{side}_pulley" sidesite="{side}_{sd}"/><site site="{side}_{ins}"/>\n'
                        f'    </spatial>\n')
    tendons += "  </tendon>\n\n  <actuator>"
    s = s.replace("  <actuator>", tendons, 1)
    open(DST, "w", encoding="utf-8").write(s)


class TendonSim(Sim):
    """膝关节由两根肌腱线驱动的仿真。"""

    def __init__(self, view=False):
        super().__init__(view)
        m = self.m
        self.knee = {s: m.joint(f"{s}_knee").id for s in ("l", "r")}
        self.ten = {s: (m.tendon(f"{s}_quad").id, m.tendon(f"{s}_ham").id) for s in ("l", "r")}
        self.knee_target = {"l": 0.3, "r": 0.3}
        self.log = {"tension": [], "arm": []}

    def set_targets(self, pose):
        super().set_targets(pose)
        if not hasattr(self, "knee_target"):  # 父类初始化时还没有肌腱信息
            return
        for s in ("l", "r"):
            self.knee_target[s] = pose.get(f"{s}_knee", self.knee_target[s])
        self._tendon_ctrl()

    def _tendon_ctrl(self):
        m, d = self.m, self.d
        mujoco.mj_fwdPosition(m, d)  # 更新肌腱长度和雅可比
        for s in ("l", "r"):
            j = self.knee[s]
            q, qd = d.qpos[m.jnt_qposadr[j]], d.qvel[m.jnt_dofadr[j]]
            tau = float(np.clip(KNEE_KP * (self.knee_target[s] - q) - KNEE_KV * qd, -KNEE_LIMIT, KNEE_LIMIT))
            dof = m.jnt_dofadr[j]
            tq, th = self.ten[s]
            if d.ten_J.size == m.ntendon * m.nv:
                J = d.ten_J.reshape(m.ntendon, m.nv)
            else:  # 稀疏存储时先转成稠密矩阵
                J = np.zeros((m.ntendon, m.nv))
                src = m if hasattr(m, "ten_J_rownnz") else d  # 新版本把稀疏结构放在模型里
                mujoco.mju_sparse2dense(J, d.ten_J, src.ten_J_rownnz, src.ten_J_rowadr, src.ten_J_colind)
            Jq, Jh = J[tq, dof], J[th, dof]
            # 线的力 F ≤ 0（负号表示拉），关节扭矩 = Jq*Fq + Jh*Fh
            Fq = Fh = -PRETENSION
            r = tau - (Jq * Fq + Jh * Fh)
            if r / Jq < 0:
                Fq += r / Jq
            else:
                Fh += r / Jh
            d.ctrl[self.act[f"{s}_quad"]] = max(-TMAX, Fq)
            d.ctrl[self.act[f"{s}_ham"]] = max(-TMAX, Fh)
            self.log["arm"].append(abs(Jq))

    def torques(self):
        tq = super().torques()
        for s in ("l", "r"):  # 两根线合起来作用在膝关节上的扭矩
            tq[f"{s}_knee"] = float(self.d.qfrc_actuator[self.m.jnt_dofadr[self.knee[s]]])
        return tq

    def reset(self):
        super().reset()
        if hasattr(self, "knee_target"):
            self.knee_target = {"l": 0.3, "r": 0.3}
            self._tendon_ctrl()

    def step(self, n=1):
        for _ in range(n):
            self._tendon_ctrl()
            mujoco.mj_step(self.m, self.d)
        if self.view is not None:
            self.view.sync()


def tension_peaks(pk):
    return {k: pk[f"{s}_{k}"] for s in ("l",) for k in ("quad", "ham")}


def main():
    build()
    figure_test.MODEL = DST
    sim = TendonSim()
    print(f"绳驱模型：{os.path.basename(DST)}，膝部滑轮半径 {PULLEY_R*1000:.0f} mm，预紧力 {PRETENSION:.0f} N\n")

    print("【1】站立 3 秒")
    pk, _, fell = run_phase(sim, 3.0, lambda t: stand_pose())
    print(f"  {'站稳' if fell is None else '摔倒'}；前线（股四头肌）最大拉力 {pk['l_quad']:.0f} N，后线（腘绳肌）{pk['l_ham']:.0f} N")

    print("\n【2】深蹲（膝 0.3 → 1.6 rad）")
    sim.reset()
    pk, _, fell = run_phase(sim, 9.0, lambda t: stand_pose(squat_knee(t)))
    squat_q = max(pk["l_quad"], pk["r_quad"])
    print(f"  {'完成' if fell is None else '摔倒'}；前线最大拉力 {squat_q:.0f} N，后线 {max(pk['l_ham'], pk['r_ham']):.0f} N")

    print("\n【3】从深蹲 0.4 秒快速站起")
    import figure_stress
    sim.reset()
    run_phase(sim, 1.0, lambda t: stand_pose())
    run_phase(sim, 2.0, lambda t: stand_pose(0.3 + 1.7 * figure_stress.smooth(t / 2)))
    pk, _, fell = run_phase(sim, 2.5, lambda t: stand_pose(2.0 - 1.7 * figure_stress.smooth(t / 0.4)))
    rise_q = max(pk["l_quad"], pk["r_quad"])
    print(f"  {'完成' if fell is None else '摔倒'}；前线最大拉力 {rise_q:.0f} N")

    print("\n【4】静步行 6 步")
    import walk
    bal = Balance(sim)
    fell, dx, _, pk = walk.walk(sim, bal, 6, 0.03, T=2.4)
    walk_q, walk_h = max(pk["l_quad"], pk["r_quad"]), max(pk["l_ham"], pk["r_ham"])
    print(f"  {'走完' if fell is None else '摔倒'}，前进 {dx*100:.1f} cm；前线最大拉力 {walk_q:.0f} N，后线 {walk_h:.0f} N")

    arms = np.array(sim.log["arm"])
    print(f"\n实际力臂（MuJoCo 计算）：{arms.min()*1000:.1f}–{arms.max()*1000:.1f} mm")

    peak = max(squat_q, rise_q, walk_q)
    print("\n【换算成零件要求】")
    print(f"  最大线拉力：{peak:.0f} N（约 {peak/9.8:.0f} kgf）")
    print(f"  线的断裂拉力至少：{peak*3:.0f} N（约 {peak*3/9.8:.0f} kgf，安全系数 3）")
    for r in (0.003, 0.005):
        print(f"  绳轮半径 {r*1000:.0f} mm → 绳轮需要扭矩 {peak*r:.2f} N·m")


if __name__ == "__main__":
    main()
