"""
55 cm 人偶 · 双手十指打字 · MuJoCo 模型生成器

用法：
    python hands_build.py       # 生成 hands.xml

  - 左右两只手（右手是左手的镜像），每只手 14 个电机，结构与 hand2_build.py 相同
  - 每只手装在自己的"手臂"（三个滑动关节，代表肩和肘）上
  - 键盘为标准方向：Q 在左边（+y 一侧），空格在最下一行
坐标：手指朝 +x，掌心朝 -z，屏幕表面 z = 0；+y 为左。单位：米。
"""
import os

import hand2_build as hb

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "hands.xml")
HAND_Y = 0.026  # 两只手中心离中线的距离（太近时两个拇指会交叉）
SIDES = {"L": 1, "R": -1}


def key_layout():
    """标准 QWERTY：Q 在 +y（左）。返回 {字符: (x, y)}。"""
    P = hb.KEY_PITCH
    keys = {}
    for r, row in enumerate(hb.ROWS):
        x = hb.KB_ORIGIN[0] + (1 - r) * hb.ROW_PITCH
        offset = (len(hb.ROWS[0]) - len(row)) * P / 2
        for c, ch in enumerate(row):
            keys[ch] = (x, 9 * P / 2 - offset - c * P)
    keys[" "] = (hb.KB_ORIGIN[0] - 2 * hb.ROW_PITCH, 0.0)
    return keys


def rng(lo, hi, s, mirror):
    """镜像时，绕 x 轴和 z 轴转动的关节方向相反，范围要取反。"""
    return (lo, hi) if (s > 0 or not mirror) else (-hi, -lo)


def hand_xml(p, s):
    def cap(name, L, rad):
        return f'<geom name="{p}{name}" type="capsule" fromto="0 0 0 {L} 0 0" size="{rad}" rgba="0.95 0.85 0.79 1"/>'

    def finger(name, f):
        L1, L2, L3 = f["len"]
        r = f["rad"]
        a = rng(-0.35, 0.35, s, True)
        return f"""
        <body name="{p}{name}1" pos="0.019 {s * f['y']} 0">
          <joint name="{p}{name}_abd" axis="0 0 1" range="{a[0]} {a[1]}" stiffness="0.002"/>
          <joint name="{p}{name}_mcp" axis="0 1 0" range="-0.25 1.6" stiffness="0.002"/>
          {cap(name + '_g1', L1, r)}
          <body name="{p}{name}2" pos="{L1} 0 0">
            <joint name="{p}{name}_pip" axis="0 1 0" range="0 1.8" stiffness="0.0016"/>
            {cap(name + '_g2', L2, r)}
            <body name="{p}{name}3" pos="{L2} 0 0">
              <joint name="{p}{name}_dip" axis="0 1 0" range="0 1.4" stiffness="0.0012"/>
              {cap(name + '_g3', L3, r * 0.95)}
              <site name="{p}{name}_tip" pos="{L3} 0 0"/>
            </body>
          </body>
        </body>"""

    t = hb.THUMB
    opp = rng(-0.2, 1.6, s, True)
    fingers = "".join(finger(n, f) for n, f in hb.FINGERS.items())
    return f"""
    <body name="{p}arm" pos="0 {s * HAND_Y} 0.020">
      <joint name="{p}arm_x" type="slide" axis="1 0 0" range="-0.03 0.06"/>
      <joint name="{p}arm_y" type="slide" axis="0 1 0" range="-0.05 0.05"/>
      <joint name="{p}arm_z" type="slide" axis="0 0 1" range="-0.015 0.03"/>
      <geom name="{p}forearm" type="capsule" fromto="-0.075 0 0 -0.014 0 0" size="0.0075" rgba="0.92 0.9 0.95 1" contype="0" conaffinity="0" mass="0.06"/>
      <body name="{p}palm">
        <joint name="{p}wrist_pitch" axis="0 1 0" range="-0.9 0.9"/>
        <joint name="{p}wrist_yaw" axis="0 0 1" range="-0.5 0.5"/>
        <geom name="{p}palm" type="box" pos="0.0055 0 0" size="0.0115 0.011 0.0035" rgba="0.95 0.85 0.79 1"/>
        <geom name="{p}wrist" type="sphere" pos="-0.012 0 0" size="0.0065" rgba="0.95 0.85 0.79 1" contype="0" conaffinity="0"/>
{fingers}
        <body name="{p}thumb1" pos="0.001 {s * -0.0105} -0.0025" euler="0 0 {s * -0.6}">
          <joint name="{p}thumb_opp" axis="1 0 0" range="{opp[0]} {opp[1]}" stiffness="0.002"/>
          <joint name="{p}thumb_cmc" axis="0 1 0" range="-0.4 1.0" stiffness="0.002"/>
          {cap('thumb_g1', t['len'][0], t['rad'])}
          <body name="{p}thumb2" pos="{t['len'][0]} 0 0">
            <joint name="{p}thumb_mcp" axis="0 1 0" range="0 1.2" stiffness="0.0015"/>
            {cap('thumb_g2', t['len'][1], t['rad'])}
            <body name="{p}thumb3" pos="{t['len'][1]} 0 0">
              <joint name="{p}thumb_ip" axis="0 1 0" range="0 1.4" stiffness="0.0012"/>
              {cap('thumb_g3', t['len'][2], t['rad'] * 0.95)}
              <site name="{p}thumb_tip" pos="{t['len'][2]} 0 0"/>
            </body>
          </body>
        </body>
      </body>
    </body>"""


TENDONS = {  # 电机名: [(关节, 力臂)]
    "index_mcp": [("index_mcp", hb.ARM_MCP)], "index_pip": [("index_pip", hb.ARM_PIP)], "index_abd": [("index_abd", hb.ARM_ABD)],
    "middle_mcp": [("middle_mcp", hb.ARM_MCP)], "middle_pip": [("middle_pip", hb.ARM_PIP)], "middle_abd": [("middle_abd", hb.ARM_ABD)],
    "ring_flex": [("ring_mcp", hb.ARM_MCP), ("ring_pip", hb.ARM_PIP)], "little_flex": [("little_mcp", hb.ARM_MCP), ("little_pip", hb.ARM_PIP)],
    "spread": [("little_abd", hb.ARM_ABD)], "thumb_opp": [("thumb_opp", hb.ARM_THUMB)], "thumb_cmc": [("thumb_cmc", hb.ARM_THUMB)],
    "thumb_flex": [("thumb_mcp", hb.ARM_THUMB)], "wrist_pitch": [("wrist_pitch", hb.ARM_WRIST)], "wrist_yaw": [("wrist_yaw", hb.ARM_WRIST)]}


def build():
    keys = key_layout()
    P = hb.KEY_PITCH
    key_geoms = "".join(
        f'      <geom name="key_{hb.key_name(ch)}" type="box" pos="{x} {y} 0.00025" '
        f'size="{hb.KEY_D/2} {(5 * P - (P - hb.KEY_W)) / 2 if ch == " " else hb.KEY_W/2} 0.00025" rgba="0.82 0.86 0.92 1"/>\n'
        for ch, (x, y) in keys.items())
    hands = "".join(hand_xml(f"{k}_", s) for k, s in SIDES.items())
    excl = "".join(f'<exclude body1="{k}_palm" body2="{k}_{b}1"/>' for k in SIDES for b in ("index", "middle", "ring", "little", "thumb"))
    eq = ""
    for k in SIDES:
        p = f"{k}_"
        for n in ("index", "middle", "ring", "little"):
            eq += f'    <joint joint1="{p}{n}_dip" joint2="{p}{n}_pip" polycoef="0 0.75 0 0 0"/>\n'
        eq += f'    <joint joint1="{p}thumb_ip" joint2="{p}thumb_mcp" polycoef="0 1.0 0 0 0"/>\n'
        eq += f'    <joint joint1="{p}ring_abd" joint2="{p}little_abd" polycoef="0 0.4 0 0 0"/>\n'
    ten, act = "", ""
    for k in SIDES:
        p = f"{k}_"
        for tn, lst in TENDONS.items():
            ten += f'    <fixed name="{p}{tn}">' + "".join(f'<joint joint="{p}{j}" coef="{c}"/>' for j, c in lst) + "</fixed>\n"
            fr = hb.T_WRIST if tn.startswith("wrist") else hb.T_FINGER
            kv = 12 if tn.startswith("wrist") else 8
            act += f'    <position name="{p}{tn}" tendon="{p}{tn}" kp="6000" kv="{kv}" forcerange="-{fr} {fr}"/>\n'
        for j in ("arm_x", "arm_y", "arm_z"):
            act += f'    <position name="{p}{j}" joint="{p}{j}" kp="400" kv="25" forcerange="-20 20"/>\n'
    xml = f"""<mujoco model="hands">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.0005" integrator="implicitfast" cone="elliptic" noslip_iterations="3"/>
  <default>
    <joint damping="0.0004" armature="2e-7"/>
    <geom contype="1" conaffinity="1" density="1100" friction="1.0 0.01 0.001" solref="0.004 1" solimp="0.95 0.99 0.0005"/>
    <site size="0.0005" rgba="0.2 0.5 0.9 1"/>
  </default>
  <worldbody>
    <light pos="0.05 0 0.3" dir="0 0 -1"/>
    <body name="phone">
      <geom name="phone" type="box" pos="0.040 0 -0.002" size="0.022 0.036 0.002" rgba="0.15 0.17 0.22 1"/>
{key_geoms}    </body>
{hands}
  </worldbody>
  <contact>{excl}</contact>
  <equality>
{eq}  </equality>
  <tendon>
{ten}  </tendon>
  <actuator>
{act}  </actuator>
</mujoco>
"""
    open(OUT, "w", encoding="utf-8").write(xml)


if __name__ == "__main__":
    build()
    print(f"已生成 {os.path.basename(OUT)}：两只手，每只 14 个电机 + 3 个手臂滑动关节")
