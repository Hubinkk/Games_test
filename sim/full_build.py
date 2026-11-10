"""
55 cm 人偶 · 整体模型生成器（身体 + 绳驱膝 + 双手 + 体积预算后的重量分布）

用法：
    python full_build.py       # 生成 figure55_full.xml

整合内容：
  - 身体骨架、24 个关节、4.4 cm 宽的脚：figure55.xml
  - 膝关节：绳驱（股四头肌 / 腘绳肌两根线，膝部滑轮半径 10 mm）：tendon_knee.py
  - 双手：6 电机手（A+B：差动双线 + 共享电机分时驱动 + 锁）：hands_ab_build.py
    手部电机放在胸腔里，经线传到手上，所以手本身很轻
  - 重量分布：docs/volume-budget.md 的推荐方案
    计算单元放进头里，30 Wh 电池放进背包，总重约 3.2 kg
碰撞：身体各部分之间不碰撞；手与手、手与地面会碰撞；手不与身体碰撞。
"""
import os

import hands_ab_build as hab
import hands_build as hsb
import tendon_knee as tk

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figure55_full.xml")
REF = os.path.join(HERE, "figure55_full_ref.xml")   # 第一版身体：动作是照着它设计的，自动配平拿它做参考

# ---------- 第二版身体（2026-11-10，docs/blueprint.md 第 7 节）----------
# 头：大脑留在头里，加一块冷板；背后模块：60 Wh 电池 + 散热；4 关节脖子，电机在胸腔；
# 控制板搬到腹部；每只手的 2 个手腕电机搬进前臂
V2_MASS = [
    ('<geom name="head" type="sphere" pos="0 0 0.044" size="0.046" mass="0.30"', '<geom name="head" type="sphere" pos="0 0 0.044" size="0.046" mass="0.32"', 1),
    ('<geom name="backpack" type="box" pos="-0.041 0 0.032" size="0.012 0.030 0.026" mass="0.20"',
     '<geom name="backpack" type="box" pos="-0.051 0 0.032" size="0.022 0.030 0.026" mass="0.31"', 1),
    ('<geom name="chest" type="box" pos="0 0 0.035" size="0.028 0.04 0.035" mass="0.62"/>',
     '<geom name="chest" type="box" pos="0 0 0.035" size="0.028 0.04 0.035" mass="0.57"/>', 1),   # +4 个脖子电机 48 g，−控制板 50 g，−手腕电机 44 g
    ('<geom type="capsule" fromto="0 0 0 0 0 0.045" size="0.03" mass="0.21"/>', '<geom type="capsule" fromto="0 0 0 0 0 0.045" size="0.03" mass="0.26"/>', 1),
    ('<geom type="capsule" fromto="0 0 0 0 0 -0.075" size="0.009" mass="0.05"/>', '<geom type="capsule" fromto="0 0 0 0 0 -0.075" size="0.009" mass="0.072"/>', 2),
]
NECK_OLD = """<body name="neck" pos="0 0 0.07">
            <joint name="neck_yaw" axis="0 0 1" range="-1.0 1.0" class="small"/>
            <geom type="capsule" fromto="0 0 0 0 0 0.02" size="0.011" mass="0.03"/>
            <body name="head" pos="0 0 0.022">
              <joint name="neck_pitch" axis="0 1 0" range="-0.5 0.6" class="small"/>"""
NECK_NEW = """<body name="neck" pos="0 0 0.07">
            <joint name="neck_low" axis="0 1 0" range="-0.5 0.6" class="small"/>
            <geom type="capsule" fromto="0 0 0 0 0 0.018" size="0.010" mass="0.012"/>
            <body name="neck_up" pos="0 0 0.018">
            <joint name="neck_yaw" axis="0 0 1" range="-1.4 1.4" class="small"/>
            <geom type="capsule" fromto="0 0 0 0 0 0.006" size="0.009" mass="0.006"/>
            <body name="head" pos="0 0 0.006">
              <joint name="neck_pitch" axis="0 1 0" range="-0.7 0.6" class="small"/>
              <joint name="neck_roll" axis="1 0 0" range="-0.45 0.45" class="small"/>"""
NECK_ACT_OLD = '<position name="neck_pitch" joint="neck_pitch" kp="2" kv="0.05" forcerange="-0.3 0.3"/>'
NECK_ACT_NEW = NECK_ACT_OLD + """
    <position name="neck_low" joint="neck_low" kp="2" kv="0.05" forcerange="-0.3 0.3"/>
    <position name="neck_roll" joint="neck_roll" kp="2" kv="0.05" forcerange="-0.3 0.3"/>"""


def v2(s):
    for a, b, n in V2_MASS:
        assert s.count(a) == n, a
        s = s.replace(a, b)
    assert s.count(NECK_OLD) == 1 and s.count(NECK_ACT_OLD) == 1
    i = s.index(NECK_OLD)
    j = s.index("</body>", i) + len("</body>")          # 头的结束标签；后面多补一个，关上 neck_up
    s = s[:i] + NECK_NEW + s[i + len(NECK_OLD):j] + "\n            </body>" + s[j:]
    return s.replace(NECK_ACT_OLD, NECK_ACT_NEW)

# 体积预算推荐方案下的重量（kg）
MASS = [
    ('<geom name="pelvis" type="box" size="0.022 0.042 0.022" mass="0.75"', '<geom name="pelvis" type="box" size="0.022 0.042 0.022" mass="0.50"'),
    ('<geom type="capsule" fromto="0 0 0 0 0 0.045" size="0.03" mass="0.15"/>', '<geom type="capsule" fromto="0 0 0 0 0 0.045" size="0.03" mass="0.21"/>'),
    ('<geom name="chest" type="box" pos="0 0 0.035" size="0.028 0.04 0.035" mass="0.55"/>',
     '<geom name="chest" type="box" pos="0 0 0.035" size="0.028 0.04 0.035" mass="0.62"/>\n'
     '          <!-- 背包：30 Wh 电池，做成角色的背包 -->\n'
     '          <geom name="backpack" type="box" pos="-0.041 0 0.032" size="0.012 0.030 0.026" mass="0.20" rgba="0.62 0.52 0.78 1"/>'),
    ('<geom name="head" type="sphere" pos="0 0 0.044" size="0.046" mass="0.25"', '<geom name="head" type="sphere" pos="0 0 0.044" size="0.046" mass="0.30"'),
]
THIGH = ('size="0.02" mass="0.14"/>', 'size="0.02" mass="0.17"/>')
KNEE_MOTOR = ('size="0.019" mass="0.07"', 'size="0.019" mass="0.09"')
SHIN = ('size="0.013" mass="0.10"/>', 'size="0.013" mass="0.11"/>')
UPPER_ARM = ('fromto="0 0 0 0 0 -0.085" size="0.011" mass="0.09"/>', 'fromto="0 0 0 0 0 -0.085" size="0.011" mass="0.10"/>')
# 身体变重、重心变高后，站立相当于一个倒立摆：踝关节需要的最小刚度约 m·g·h ≈ 3.2 × 9.8 × 0.28 ≈ 8.7 N·m/rad，
# 原来的 10 几乎没有余量，所以把踝关节伺服刚度提高（扭矩上限不变）
ANKLE = [(f'<position name="{s}_ankle_{j}" joint="{s}_ankle_{j}" kp="{kp}" kv="{kv}"', f'<position name="{s}_ankle_{j}" joint="{s}_ankle_{j}" kp="{nkp}" kv="{nkv}"')
         for s in ("l", "r") for j, kp, kv, nkp, nkv in (("pitch", 10, 0.25, 25, 0.6), ("roll", 8, 0.2, 20, 0.5))]

# 手装在前臂末端：手指朝下，掌心朝向大腿，拇指朝前
HAND_FRAME = {"L": "0 0 -1 -1 0 0", "R": "0 0 -1 1 0 0"}


def hand_body(k):
    """取 hands_build 的手（去掉手臂滑动关节和前臂），装到前臂末端。"""
    p = f"{k}_"
    xml = hsb.hand_xml(p, hsb.SIDES[k])
    start = xml.index(f'<body name="{p}palm">')
    end = xml.rindex("</body>")  # 最后一个 </body> 属于手臂，去掉
    palm = xml[start:end].rstrip()
    return (f'<body name="{p}hand" pos="0 0 -0.081" xyaxes="{HAND_FRAME[k]}" childclass="hand">\n'
            f'{palm}\n</body>')


def build():
    tk.build()
    s = open(tk.DST, encoding="utf-8").read()
    for a, b in MASS:
        assert s.count(a) == 1, a
        s = s.replace(a, b)
    for a, b in (THIGH, KNEE_MOTOR, SHIN, UPPER_ARM):
        assert s.count(a) == 2, a
        s = s.replace(a, b)
    for a, b in ANKLE:
        assert s.count(a) == 1, a
        s = s.replace(a, b)
    # 换上双手
    for side, k in (("l", "L"), ("r", "R")):
        old_start = s.index(f'<body name="{side}_hand" pos="0 0 -0.075">')
        old_end = s.index("</body>", old_start) + len("</body>")
        s = s[:old_start] + hand_body(k) + s[old_end:]
    # 手部的默认设置、等式约束、肌腱、执行器
    hand_default = """    <default class="hand">
      <joint damping="0.0004" armature="2e-7" frictionloss="0"/>
      <geom contype="4" conaffinity="5" density="1100" friction="1.0 0.01 0.001" solref="0.004 1" solimp="0.95 0.99 0.0005" rgba="0.95 0.85 0.79 1"/>
      <site size="0.0005"/>
    </default>
    <equality solref="0.001 1"/>
  </default>"""
    s = s.replace("  </default>", hand_default, 1)
    s = s.replace('<option timestep="0.002" integrator="implicitfast"/>',
                  '<option timestep="0.0005" integrator="implicitfast" cone="elliptic"/>')
    excl = "".join(f'<exclude body1="{k}_palm" body2="{k}_{b}1"/>' for k in ("L", "R") for b in ("index", "middle", "ring", "little", "thumb"))
    eq, ten, act = "", "", ""
    for k in ("L", "R"):
        p = f"{k}_"
        for n in ("index", "middle", "ring", "little"):
            eq += f'    <joint joint1="{p}{n}_dip" joint2="{p}{n}_pip" polycoef="0 0.75 0 0 0"/>\n'
        for n in ("index", "middle"):
            eq += f'    <joint joint1="{p}{n}_pip" joint2="{p}{n}_mcp" polycoef="0 1.0 0 0 0"/>\n'
        eq += f'    <joint joint1="{p}thumb_ip" joint2="{p}thumb_mcp" polycoef="0 1.0 0 0 0"/>\n'
        eq += f'    <joint joint1="{p}ring_abd" joint2="{p}little_abd" polycoef="0 0.4 0 0 0"/>\n'
        eq += f'    <joint joint1="{p}thumb_cmc" polycoef="0 0 0 0 0"/>\n'
        eq += f'    <joint joint1="{p}wrist_pitch" polycoef="0 0 0 0 0"/>\n'
        for tn, lst in hab.TENDONS.items():
            ten += f'    <fixed name="{p}{tn}">' + "".join(f'<joint joint="{p}{j}" coef="{c}"/>' for j, c in lst) + "</fixed>\n"
            fr = hsb.hb.T_WRIST if tn.startswith("wrist") else hsb.hb.T_FINGER
            kv = 12 if tn.startswith("wrist") else 8
            act += f'    <position name="{p}{tn}" tendon="{p}{tn}" kp="6000" kv="{kv}" forcerange="-{fr} {fr}"/>\n'
    s = s.replace("  <tendon>\n", "  <contact>" + excl + "</contact>\n  <equality>\n" + eq + "  </equality>\n  <tendon>\n" + ten, 1)
    s = s.replace("  </actuator>", act + "  </actuator>", 1)
    open(REF, "w", encoding="utf-8").write(s)
    open(OUT, "w", encoding="utf-8").write(v2(s))


if __name__ == "__main__":
    build()
    import mujoco
    m = mujoco.MjModel.from_xml_path(OUT)
    print(f"已生成 {os.path.basename(OUT)}：总质量 {sum(m.body_mass):.2f} kg，关节 {m.njnt}，执行器 {m.nu}，几何体 {m.ngeom}")
