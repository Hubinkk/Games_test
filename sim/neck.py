"""
55 cm 人偶 · 脖子：旧的 2 关节 vs 新的 4 关节（电机放在胸腔）

用法：
    python neck.py          # 跑全部表情和"趴着抬头"，打印结果，导出 ../neck_frames.js 供 Neck.html 回放

三种脖子：
  old    现在的脖子：左右转 ±57°、点头一个关节；两个 12 g 电机装在脖子和头里
  new    4 个关节：下颈俯仰、左右转 ±80°、上颈俯仰、歪头；电机放在胸腔里，用线 / 连杆传到关节
         头和脖子因此轻了约 24 g
  spring new + 平衡弹簧：胸腔里的弹簧抵消头的重量的 80%（按"身体直立"时的重量设计）

控制：力矩电机 + PD + 重力前馈（控制器知道头有多重），扭矩上限 0.3 N·m，转速上限 10 rad/s（沿用整体模型）。
发热：和 day.py 相同的小关节热模型（R = 15 K/W），用"一直重复这个动作时的稳定温度"来比较。
眼睛（±12° 云台）和眼睑不在物理模型里，只在网页上按关键帧画出来。
"""
import json
import math
import os

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FPS = 30
DT = 0.001
LIM = 0.3                 # N·m，沿用整体模型的小关节扭矩上限
W_MAX = 10.0              # rad/s 转速上限
KP, KD = 4.0, 0.08
SPRING_FRAC = 0.8
# 发热（与 day.py 相同）
T_AMB, R_SMALL = 25.0, 15.0
RATED_FRAC, DT_RATED = 1.35 / 2.8, 45.0
K_LOSS = DT_RATED / (R_SMALL * (RATED_FRAC * LIM) ** 2)   # W /(N·m)^2

DEG = math.pi / 180

HEAD_GEOMS = """
            <geom name="head" type="sphere" pos="0 0 0.044" size="0.046" mass="{hm}" rgba="0.95 0.85 0.79 1"/>"""


def xml(kind, gravity="0 0 -9.81"):
    """kind = old / new"""
    if kind == "old":
        neck = f"""
      <body name="neck" pos="0 0 0.07">
        <joint name="yaw" axis="0 0 1" range="-1.0 1.0" class="small"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.02" size="0.011" mass="0.03"/>
        <body name="head" pos="0 0 0.022">
          <joint name="head_pitch" axis="0 1 0" range="-0.5 0.6" class="small"/>{HEAD_GEOMS.format(hm=0.30)}
        </body>
      </body>"""
        joints = ["yaw", "head_pitch"]
    else:
        # 两个 12 g 电机搬进胸腔；脖子里只剩轴承、线轮和导线管
        neck = f"""
      <body name="neck_low" pos="0 0 0.07">
        <joint name="low_pitch" axis="0 1 0" range="-0.5 0.6" class="small"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.018" size="0.010" mass="0.012"/>
        <body name="neck_up" pos="0 0 0.018">
          <joint name="yaw" axis="0 0 1" range="-1.4 1.4" class="small"/>
          <geom type="capsule" fromto="0 0 0 0 0 0.006" size="0.009" mass="0.006"/>
          <body name="head" pos="0 0 0.006">
            <joint name="head_pitch" axis="0 1 0" range="-0.7 0.6" class="small"/>
            <joint name="roll" axis="1 0 0" range="-0.45 0.45" class="small"/>{HEAD_GEOMS.format(hm=0.288)}
          </body>
        </body>
      </body>"""
        joints = ["low_pitch", "yaw", "head_pitch", "roll"]
    acts = "\n".join(f'    <motor name="{j}" joint="{j}" ctrlrange="-{LIM} {LIM}" ctrllimited="true"/>' for j in joints)
    return f"""<mujoco model="neck_{kind}">
  <compiler angle="radian"/>
  <option timestep="{DT}" gravity="{gravity}" integrator="implicitfast"/>
  <default>
    <default class="small"><joint armature="0.001" damping="0.01" limited="true"/></default>
    <geom contype="0" conaffinity="0"/>
  </default>
  <worldbody>
    <body name="chest">
      <geom type="box" pos="0 0 0.035" size="0.028 0.04 0.035" mass="0.62"/>{neck}
    </body>
  </worldbody>
  <actuator>
{acts}
  </actuator>
</mujoco>"""


class Neck:
    def __init__(self, kind, spring=False, gravity="0 0 -9.81"):
        self.kind, self.spring = kind, spring
        self.m = mujoco.MjModel.from_xml_string(xml(kind, gravity))
        self.d = mujoco.MjData(self.m)
        # 弹簧是按"身体直立"设计的：用直立时的重力矩来算它出的力
        self.mref = mujoco.MjModel.from_xml_string(xml(kind))
        self.dref = mujoco.MjData(self.mref)
        self.mg = mujoco.MjModel.from_xml_string(xml(kind, gravity))   # 控制器的重力前馈：知道真实重力方向
        self.dg = mujoco.MjData(self.mg)
        self.joints = [self.m.joint(i).name for i in range(self.m.njnt)]
        self.lo = self.m.jnt_range[:, 0].copy()
        self.hi = self.m.jnt_range[:, 1].copy()
        self.tgt = np.zeros(self.m.nu)

    def grav(self, mm, dd):
        dd.qpos[:] = self.d.qpos
        dd.qvel[:] = 0
        mujoco.mj_forward(mm, dd)
        return dd.qfrc_bias.copy()

    def step(self, want):
        want = np.clip(want, self.lo, self.hi)
        self.tgt += np.clip(want - self.tgt, -W_MAX * DT, W_MAX * DT)
        q, qd = self.d.qpos, self.d.qvel
        g = self.grav(self.mg, self.dg)
        spring = np.zeros(self.m.nv)
        if self.spring:
            gref = self.grav(self.mref, self.dref)
            for i, n in enumerate(self.joints):
                if n != "yaw":
                    spring[i] = SPRING_FRAC * gref[i]
        self.d.qfrc_applied[:] = spring
        tau = KP * (self.tgt - q) - KD * qd + g - spring
        tau = np.clip(tau, -LIM, LIM)
        self.d.ctrl[:] = tau
        mujoco.mj_step(self.m, self.d)
        return tau

    def head_pose(self):
        b = self.m.body("head").id
        return self.d.xpos[b].copy(), self.d.xmat[b].reshape(3, 3).copy()


def ideal_pose(q4):
    """新脖子按目标角度摆出来时，头的位置和朝向（"想做的样子"）。"""
    m = mujoco.MjModel.from_xml_string(xml("new"))
    d = mujoco.MjData(m)
    d.qpos[:] = q4
    mujoco.mj_kinematics(m, d)
    b = m.body("head").id
    return d.xpos[b].copy(), d.xmat[b].reshape(3, 3).copy()


# ---------------- 表情 ----------------
# 关键帧：(时间 s, 下颈俯仰, 左右转, 上颈俯仰, 歪头, 眼左右, 眼上下, 眼睑)
#  角度单位：度。俯仰正 = 低头 / 往前；左右转正 = 向它自己的左边；歪头正 = 向它的右肩
#  眼睛 ±12°（正 = 看向它的左边 / 往上）；眼睑 0 = 睁开，1 = 合上
def K(t, lp=0, yaw=0, hp=0, roll=0, ex=0, ey=0, lid=0.0):
    return (t, lp, yaw, hp, roll, ex, ey, lid)


EXPRESSIONS = [
    ("listen", "看着你，听你说话", "边听边轻轻点头。小动作：幅度只有 3°～5°。", [
        K(0), K(0.6, hp=4), K(1.0, hp=0), K(1.6, hp=5, lid=0.1), K(2.0, hp=0), K(2.3, lid=1), K(2.45),
        K(3.2, hp=3, roll=3), K(3.8)]),
    ("tilt", "歪头", "疑惑、好奇。二次元角色最有代表性的动作；旧脖子没有这个关节。", [
        K(0), K(0.5, ey=2), K(1.2, roll=22, hp=4, ex=-3, ey=3), K(2.6, roll=22, hp=4, ex=-3, ey=3),
        K(2.75, roll=22, hp=4, ex=-3, ey=3, lid=1), K(2.9, roll=22, hp=4, ex=-3, ey=3), K(3.8)]),
    ("peer", "探头（凑近看）", "下颈往前、上颈往回：脸保持朝前，整个头往前伸。", [
        K(0), K(0.4, ey=-3), K(1.4, lp=30, hp=-32, ey=-4, lid=0.15), K(2.8, lp=30, hp=-30, ex=4, ey=-4, lid=0.15),
        K(3.2, lp=30, hp=-30, ex=-4, ey=-4, lid=0.15), K(4.2)]),
    ("flinch", "缩脖子（吓一跳）", "下颈往后、下巴收回，很快；眼睛一下睁大再眨。", [
        K(0), K(0.15, lp=-24, hp=14, ey=4), K(0.6, lp=-24, hp=14, ey=4), K(0.7, lp=-24, hp=14, ey=4, lid=1),
        K(0.85, lp=-24, hp=14, ey=2), K(2.0, lp=-10, hp=6), K(3.0)]),
    ("peek", "偷看", "头转开、低一点，只用眼睛偷偷看你；被发现时眼睛先跳开，头再转得更远。", [
        K(0), K(1.0, yaw=50, hp=10, ex=-6, ey=-3, lid=0.2), K(1.6, yaw=50, hp=10, ex=-6, ey=-3, lid=0.2),
        K(2.4, yaw=50, hp=12, roll=-4, ex=-12, ey=4, lid=0.45), K(3.4, yaw=48, hp=12, roll=-4, ex=-12, ey=4, lid=0.45),
        K(3.5, yaw=48, hp=12, roll=-4, ex=10, ey=-4, lid=0.2), K(4.1, yaw=72, hp=16, roll=-6, ex=6, ey=-6, lid=0.3),
        K(5.0, yaw=72, hp=16, roll=-6, ex=6, ey=-6, lid=0.3), K(6.0)]),
    ("avert", "躲开视线", "眼睛先往下偏开，头再慢慢跟着低下、转开。人害羞时就是这个顺序。", [
        K(0), K(0.5), K(0.7, ex=8, ey=-9, lid=0.3), K(2.0, lp=8, yaw=24, hp=16, roll=6, ex=6, ey=-8, lid=0.35),
        K(3.6, lp=8, yaw=26, hp=18, roll=6, ex=6, ey=-8, lid=0.35), K(4.0, lp=8, yaw=26, hp=18, roll=6, ex=-10, ey=-2, lid=0.25),
        K(4.3, lp=8, yaw=26, hp=18, roll=6, ex=6, ey=-8, lid=0.35), K(5.5)]),
    ("sound", "听到声音，转过去看", "眼睛先到，头跟上，转 75°。旧脖子最多转 57°。", [
        K(0), K(0.1, ex=12), K(0.45, yaw=75, ex=2), K(1.6, yaw=75), K(1.75, yaw=75, lid=1), K(1.9, yaw=75),
        K(2.6, yaw=75, roll=8), K(3.6)]),
    ("idle", "发呆时的小动作", "呼吸带着头轻轻起伏、眼睛偶尔扫一下、慢慢眨眼。幅度 1°～2°。", None),
]


def idle_keys():
    """发呆：用正弦和小跳动直接生成"""
    out = []
    for i in range(0, 61):
        t = i * 0.1
        br = math.sin(2 * math.pi * t / 3.2)
        lp = 1.2 * br
        hp = -0.8 * br + (0.6 * math.sin(2 * math.pi * t / 5.1))
        roll = 1.5 * math.sin(2 * math.pi * t / 7.3)
        yaw = 2.0 * math.sin(2 * math.pi * t / 6.1)
        ex = 0 if t < 1.5 else (5 if t < 2.6 else (-3 if t < 4.4 else 1))
        ey = 0 if t < 2.6 else (2 if t < 4.4 else 0)
        lid = 1.0 if 3.4 <= t < 3.6 else 0.12
        out.append(K(t, lp, yaw, hp, roll, ex, ey, lid))
    return out


def sample(keys, t):
    """关键帧之间用最小加加速度（平滑起止）插值"""
    if t <= keys[0][0]:
        return np.array(keys[0][1:], float)
    for a, b in zip(keys, keys[1:]):
        if t <= b[0]:
            s = (t - a[0]) / max(b[0] - a[0], 1e-9)
            s = s * s * s * (10 - 15 * s + 6 * s * s)
            return np.array(a[1:], float) * (1 - s) + np.array(b[1:], float) * s
    return np.array(keys[-1][1:], float)


def to_q(kind, v):
    lp, yaw, hp, roll = v[0] * DEG, v[1] * DEG, v[2] * DEG, v[3] * DEG
    if kind == "old":
        return np.array([yaw, lp + hp])   # 旧脖子：没有下颈和歪头，只能把俯仰合在一起
    return np.array([lp, yaw, hp, roll])


def rot_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def run_expr(keys, kind, spring=False, gravity="0 0 -9.81"):
    nk = Neck(kind, spring, gravity)
    T = keys[-1][0] + 0.4
    n = int(T / DT)
    q0 = to_q(kind, sample(keys, 0))
    nk.d.qpos[:] = np.clip(q0, nk.lo, nk.hi)
    nk.tgt[:] = nk.d.qpos
    mujoco.mj_forward(nk.m, nk.d)
    frames, loss, peak_tau, peak_w = [], 0.0, 0.0, 0.0
    err_ang, err_pos, track, settled = [], [], [], []
    per = int(1 / (FPS * DT))
    for i in range(n):
        t = i * DT
        v = sample(keys, t)
        want = to_q(kind, v)
        tau = nk.step(want)
        loss += K_LOSS * float(np.sum(tau ** 2)) * DT
        peak_tau = max(peak_tau, float(np.max(np.abs(tau))))
        peak_w = max(peak_w, float(np.max(np.abs(nk.d.qvel))))
        if i % per == 0:
            p_id, R_id = ideal_pose(to_q("new", v))
            p, R = nk.head_pose()
            ea, ep = rot_deg(R_id, R), float(np.linalg.norm(p - p_id)) * 1000
            err_ang.append(ea)
            err_pos.append(ep)
            # 目标已经停下 0.15 s 以上的时刻：看"摆不摆得出这个姿势"，不算转得慢的延迟
            moving = any(np.max(np.abs(sample(keys, t - dt_) - v)[:4]) > 0.5 for dt_ in (0.05, 0.1, 0.15))
            if not moving:
                settled.append((ea, ep))
            track.append(float(np.max(np.abs(np.clip(want, nk.lo, nk.hi) - nk.d.qpos))) / DEG)
            frames.append([round(float(x), 4) for x in nk.d.qpos] + [round(float(v[4]), 2), round(float(v[5]), 2), round(float(v[6]), 3)])
    p_avg = loss / T   # 平均发热功率（所有关节合计）
    # 一直重复这个动作时，最热的那个关节的稳定温度（按每个关节各自的发热算）
    return dict(frames=frames, T=T, p_avg=p_avg, peak_tau=peak_tau, peak_w=peak_w,
                pose_err_deg=max(err_ang), pose_err_mm=max(err_pos),
                hold_err_deg=max(a for a, _ in settled), hold_err_mm=max(b for _, b in settled), track_err_deg=float(np.percentile(track, 95)),
                track_err_rms=float(np.sqrt(np.mean(np.square(track)))))


def hottest(keys, kind, spring=False, gravity="0 0 -9.81", hold=None):
    """每个关节一直重复（或保持）这个动作时的稳定温度，取最高的"""
    nk = Neck(kind, spring, gravity)
    T = (keys[-1][0] + 0.4) if hold is None else hold
    n = int(T / DT)
    q0 = to_q(kind, sample(keys, 0))
    nk.d.qpos[:] = np.clip(q0, nk.lo, nk.hi)
    nk.tgt[:] = nk.d.qpos
    mujoco.mj_forward(nk.m, nk.d)
    e = np.zeros(nk.m.nu)
    for i in range(n):
        tau = nk.step(to_q(kind, sample(keys, i * DT)))
        e += K_LOSS * tau ** 2 * DT
    p = e / T
    j = int(np.argmax(p))
    return T_AMB + R_SMALL * float(p[j]), nk.joints[j], [round(float(x), 3) for x in p]


def prone_test():
    """趴在地上抬头看前面：胸口朝下，头顶指向前方，要把头往后仰 90° 才能看向前面。
    新脖子：下颈 −29° + 上颈 −40° = −69°（再往上看靠眼睛 +12° 和腰）；旧脖子只能 −29°。"""
    g = "9.81 0 0"   # 胸口朝下：重力指向胸的正前方（+x）
    res = {}
    keys_new = [K(0, lp=0, hp=0), K(1.2, lp=-29, hp=-40), K(21.2, lp=-29, hp=-40)]
    keys_old = [K(0), K(1.2, lp=0, hp=-29), K(21.2, lp=0, hp=-29)]
    for name, kind, spring, keys in (("old", "old", False, keys_old), ("new", "new", False, keys_new),
                                     ("spring", "new", True, keys_new)):
        nk = Neck(kind, spring, g)
        mujoco.mj_forward(nk.m, nk.d)
        taus = []
        e = np.zeros(nk.m.nu)
        n = int(21.2 / DT)
        for i in range(n):
            tau = nk.step(to_q(kind, sample(keys, i * DT)))
            e += K_LOSS * tau ** 2 * DT
            if i * DT > 2:
                taus.append(np.abs(tau))
        pk = np.max(np.array(taus), axis=0)
        p_hold = K_LOSS * np.array(taus)[-1] ** 2
        # 头抬起来的角度（相对水平看向地面算 0，看向正前方算 90）
        _, R = nk.head_pose()
        face = R[:, 0]   # 脸朝向（头的 +x）
        # 世界里"往前看"= 胸的 +z 方向；脸朝向在 +z 上的分量越大越接近看向前方
        look = math.degrees(math.asin(max(-1, min(1, face[2]))))
        res[name] = dict(peak_tau=[round(float(x), 3) for x in pk], joints=nk.joints,
                         T_hold=round(T_AMB + R_SMALL * float(np.max(p_hold)), 1),
                         look_deg=round(look, 1))
    # 弹簧在趴着时会不会帮倒忙：弹簧在头"没抬起"时的出力
    return res


def main():
    out = {"expr": [], "meta": {}}
    rows = []
    for key, title, desc, keys in EXPRESSIONS:
        if keys is None:
            keys = idle_keys()
        r_old = run_expr(keys, "old")
        r_new = run_expr(keys, "new")
        r_spr = run_expr(keys, "new", spring=True)
        h_old = hottest(keys, "old")
        h_new = hottest(keys, "new")
        h_spr = hottest(keys, "new", spring=True)
        rows.append((title, r_old, r_new, r_spr, h_old, h_new, h_spr))
        out["expr"].append(dict(
            key=key, title=title, desc=desc, T=round(r_new["T"], 2),
            old=r_old["frames"], new=r_spr["frames"],
            stats={k: dict(pose_err_deg=round(r["pose_err_deg"], 1), pose_err_mm=round(r["pose_err_mm"], 1), hold_err_deg=round(r["hold_err_deg"], 1), hold_err_mm=round(r["hold_err_mm"], 1),
                           track_rms=round(r["track_err_rms"], 2), peak_tau=round(r["peak_tau"], 3),
                           peak_w=round(r["peak_w"], 1), T_ss=round(h[0], 1), hot=h[1])
                   for k, r, h in (("old", r_old, h_old), ("new", r_new, h_new), ("spring", r_spr, h_spr))}))
        print(f"\n【{title}】")
        for k, r, h in (("旧 2 关节", r_old, h_old), ("新 4 关节", r_new, h_new), ("新 + 弹簧", r_spr, h_spr)):
            print(f"  {k}: 和想做的样子最多差 {r['pose_err_deg']:.1f}° / {r['pose_err_mm']:.1f} mm（停下来时 {r['hold_err_deg']:.1f}° / {r['hold_err_mm']:.1f} mm）；"
                  f"跟踪误差 RMS {r['track_err_rms']:.2f}°；峰值扭矩 {r['peak_tau']:.3f} N·m；最快 {r['peak_w']:.1f} rad/s；"
                  f"一直做这个动作最热的关节 {h[1]} 约 {h[0]:.0f} °C")
    pr = prone_test()
    out["prone"] = pr
    print("\n【趴着抬头看前面，保持 20 秒】")
    for k, name in (("old", "旧 2 关节"), ("new", "新 4 关节"), ("spring", "新 + 弹簧")):
        r = pr[k]
        print(f"  {name}: 脸抬到 {r['look_deg']:.0f}°（90° = 看向正前方）；峰值扭矩 {r['peak_tau']} ({r['joints']})；"
              f"一直保持的稳定温度约 {r['T_hold']:.0f} °C")
    out["meta"] = dict(fps=FPS, lim=LIM, spring=SPRING_FRAC, w_max=W_MAX,
                       old_joints=["yaw", "head_pitch"], new_joints=["low_pitch", "yaw", "head_pitch", "roll"])
    with open(os.path.join(HERE, "..", "neck_frames.js"), "w", encoding="utf-8") as f:
        f.write("window.NECK=" + json.dumps(out, ensure_ascii=False, separators=(",", ":")) + ";\n")
    with open(os.path.join(HERE, "neck_result.json"), "w", encoding="utf-8") as f:
        json.dump({"expr": [{k: v for k, v in e.items() if k not in ("old", "new")} for e in out["expr"]],
                   "prone": pr}, f, ensure_ascii=False, indent=1)
    print("\n已导出 ../neck_frames.js 和 neck_result.json")


if __name__ == "__main__":
    main()
