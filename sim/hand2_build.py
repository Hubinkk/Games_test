"""
55 cm 人偶 · 绳驱手 v2（14 个电机）· MuJoCo 模型生成器

用法：
    python hand2_build.py      # 生成 hand2.xml

和 v1 的区别：
  - 食指、中指：根部弯曲 + 根部左右张开 + 中间节弯曲（指尖节和中间节按 0.75 联动，和真人一样）
  - 无名指、小指：各 1 个整指弯曲电机，另有 1 个"张开"电机带动小指外展
  - 拇指：对掌旋转 + 根部弯曲 + 中间＋指尖弯曲
  - 手腕：上下、左右
  - 每个电机用"一个绳轮绕两根线"的闭环绳驱，双向可控；
    在模型里用"固定肌腱"表示，系数 = 力臂，所以执行器的力 = 线上的拉力
  - 所有指节、手掌、手机屏幕之间都有完整的碰撞
  - 手装在三个滑动关节上，代表人偶的肩和肘把手送到目标附近（"手臂"）

坐标：手指朝 +x，掌心朝 -z；手机屏幕表面在 z = 0。单位：米。
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "hand2.xml")

R_FINGER = 0.0020
FINGERS = {
    "index":  {"y": -0.0075, "len": [0.0070, 0.0050, 0.0040], "rad": R_FINGER},
    "middle": {"y": -0.0025, "len": [0.0075, 0.0055, 0.0042], "rad": R_FINGER},
    "ring":   {"y":  0.0025, "len": [0.0070, 0.0050, 0.0040], "rad": R_FINGER},
    "little": {"y":  0.0075, "len": [0.0056, 0.0042, 0.0035], "rad": 0.0018},
}
THUMB = {"len": [0.0090, 0.0065, 0.0050], "rad": 0.0021}
ARM = {"pos": [0.0, 0.0, 0.020]}  # 手腕初始位置（相对手机屏幕）

# 力臂（m）：决定线上拉力与关节扭矩的换算
ARM_MCP, ARM_PIP, ARM_ABD, ARM_THUMB, ARM_WRIST = 0.0022, 0.0018, 0.0015, 0.0020, 0.0040
T_FINGER, T_WRIST = 15.0, 30.0  # 每个电机的最大线拉力（N）

# 键盘：手机屏幕尺度，3 行 QWERTY
KEY_PITCH, ROW_PITCH = 0.0065, 0.0090
KEY_W, KEY_D = 0.0055, 0.0075
ROWS = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
KB_ORIGIN = (0.040, 0.0)  # 键盘中心（x 为离手腕的距离方向）


def key_layout():
    """返回 {字母: (x, y)}。行沿 x 方向排列（离手越远越靠上一行），列沿 y 方向。"""
    keys = {}
    for r, row in enumerate(ROWS):
        x = KB_ORIGIN[0] + (1 - r) * ROW_PITCH
        offset = (len(ROWS[0]) - len(row)) * KEY_PITCH / 2
        for c, ch in enumerate(row):
            y = KB_ORIGIN[1] - (len(ROWS[0]) - 1) * KEY_PITCH / 2 + offset + c * KEY_PITCH
            keys[ch] = (x, y)
    keys[" "] = (KB_ORIGIN[0] - 2 * ROW_PITCH, KB_ORIGIN[1])  # 空格键在最下一行中间
    return keys


def key_name(ch):
    return "space" if ch == " " else ch


def cap(name, L, rad, rgba="0.95 0.85 0.79 1"):
    return f'<geom name="{name}" type="capsule" fromto="0 0 0 {L} 0 0" size="{rad}" rgba="{rgba}"/>'


def finger(name, f):
    L1, L2, L3 = f["len"]
    r = f["rad"]
    return f"""
        <body name="{name}1" pos="0.019 {f['y']} 0">
          <joint name="{name}_abd" axis="0 0 1" range="-0.35 0.35" stiffness="0.002"/>
          <joint name="{name}_mcp" axis="0 1 0" range="-0.25 1.6" stiffness="0.002"/>
          {cap(name + '_g1', L1, r)}
          <body name="{name}2" pos="{L1} 0 0">
            <joint name="{name}_pip" axis="0 1 0" range="0 1.8" stiffness="0.0016"/>
            {cap(name + '_g2', L2, r)}
            <body name="{name}3" pos="{L2} 0 0">
              <joint name="{name}_dip" axis="0 1 0" range="0 1.4" stiffness="0.0012"/>
              {cap(name + '_g3', L3, r * 0.95)}
              <site name="{name}_tip" pos="{L3} 0 0"/>
              <site name="{name}_pad" pos="{L3 * 0.6} 0 {-r * 0.95}"/>
              <site name="{name}_tap" pos="{L3 + r * 0.85} 0 {-r * 0.3}"/>
            </body>
          </body>
        </body>"""


def build():
    t = THUMB
    fingers = "".join(finger(n, f) for n, f in FINGERS.items())
    keys = key_layout()
    key_geoms = "".join(
        f'      <geom name="key_{key_name(ch)}" type="box" pos="{x} {y} 0.00025" size="{KEY_D/2} {(5 * KEY_PITCH - (KEY_PITCH - KEY_W)) / 2 if ch == " " else KEY_W/2} 0.00025" rgba="0.82 0.86 0.92 1"/>\n'
        f'      <site name="key_{key_name(ch)}" pos="{x} {y} 0.0005" size="0.0006"/>\n'
        for ch, (x, y) in keys.items())
    xml = f"""<mujoco model="hand2">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.0005" integrator="implicitfast" cone="elliptic" noslip_iterations="3"/>
  <default>
    <joint damping="0.0004" armature="2e-7"/>
    <geom contype="1" conaffinity="1" density="1100" friction="1.0 0.01 0.001" solref="0.004 1" solimp="0.95 0.99 0.0005"/>
    <site size="0.0005" rgba="0.2 0.5 0.9 1"/>
  </default>
  <worldbody>
    <light pos="0.05 0 0.3" dir="0 0 -1"/>
    <!-- 手机：屏幕表面 z = 0 -->
    <body name="phone" pos="0 0 0">
      <geom name="phone" type="box" pos="0.040 0 -0.002" size="0.022 0.036 0.002" rgba="0.15 0.17 0.22 1"/>
{key_geoms}    </body>
    <!-- 手臂（代表肩和肘）：三个滑动关节把手送到目标附近 -->
    <body name="arm" pos="{ARM['pos'][0]} {ARM['pos'][1]} {ARM['pos'][2]}">
      <joint name="arm_x" type="slide" axis="1 0 0" range="-0.03 0.06"/>
      <joint name="arm_y" type="slide" axis="0 1 0" range="-0.05 0.05"/>
      <joint name="arm_z" type="slide" axis="0 0 1" range="-0.015 0.03"/>
      <geom name="forearm" type="capsule" fromto="-0.075 0 0 -0.014 0 0" size="0.0075" rgba="0.92 0.9 0.95 1" contype="0" conaffinity="0" mass="0.06"/>
      <body name="palm">
        <joint name="wrist_pitch" axis="0 1 0" range="-0.9 0.9"/>
        <joint name="wrist_yaw" axis="0 0 1" range="-0.5 0.5"/>
        <geom name="palm" type="box" pos="0.0055 0 0" size="0.0115 0.011 0.0035" rgba="0.95 0.85 0.79 1"/>
        <geom name="wrist" type="sphere" pos="-0.012 0 0" size="0.0065" rgba="0.95 0.85 0.79 1" contype="0" conaffinity="0"/>
{fingers}
        <body name="thumb1" pos="0.001 -0.0105 -0.0025" euler="0 0 -0.6">
          <joint name="thumb_opp" axis="1 0 0" range="-0.2 1.6" stiffness="0.002"/>
          <joint name="thumb_cmc" axis="0 1 0" range="-0.4 1.0" stiffness="0.002"/>
          {cap('thumb_g1', t['len'][0], t['rad'])}
          <body name="thumb2" pos="{t['len'][0]} 0 0">
            <joint name="thumb_mcp" axis="0 1 0" range="0 1.2" stiffness="0.0015"/>
            {cap('thumb_g2', t['len'][1], t['rad'])}
            <body name="thumb3" pos="{t['len'][1]} 0 0">
              <joint name="thumb_ip" axis="0 1 0" range="0 1.4" stiffness="0.0012"/>
              {cap('thumb_g3', t['len'][2], t['rad'] * 0.95)}
              <site name="thumb_tip" pos="{t['len'][2]} 0 0"/>
              <site name="thumb_pad" pos="{t['len'][2] * 0.6} 0 {-t['rad'] * 0.95}"/>
            </body>
          </body>
        </body>
      </body>
    </body>
    <body name="pen" pos="0.026 -0.0025 0.008">
      <geom name="pen" type="cylinder" size="0.003 0.012" euler="1.5708 0 0" rgba="0.55 0.7 0.92 1" contype="2" conaffinity="2"/>
    </body>
  </worldbody>

  <contact>
    <exclude body1="palm" body2="index1"/><exclude body1="palm" body2="middle1"/>
    <exclude body1="palm" body2="ring1"/><exclude body1="palm" body2="little1"/>
    <exclude body1="palm" body2="thumb1"/>
  </contact>

  <equality>
    <!-- 指尖节与中间节联动（和真人一样，DIP ≈ 0.75 × PIP） -->
    <joint joint1="index_dip" joint2="index_pip" polycoef="0 0.75 0 0 0"/>
    <joint joint1="middle_dip" joint2="middle_pip" polycoef="0 0.75 0 0 0"/>
    <joint joint1="ring_dip" joint2="ring_pip" polycoef="0 0.75 0 0 0"/>
    <joint joint1="little_dip" joint2="little_pip" polycoef="0 0.75 0 0 0"/>
    <joint joint1="thumb_ip" joint2="thumb_mcp" polycoef="0 1.0 0 0 0"/>
    <joint joint1="ring_abd" joint2="little_abd" polycoef="0 0.4 0 0 0"/>
  </equality>

  <tendon>
    <fixed name="index_mcp"><joint joint="index_mcp" coef="{ARM_MCP}"/></fixed>
    <fixed name="index_pip"><joint joint="index_pip" coef="{ARM_PIP}"/></fixed>
    <fixed name="index_abd"><joint joint="index_abd" coef="{ARM_ABD}"/></fixed>
    <fixed name="middle_mcp"><joint joint="middle_mcp" coef="{ARM_MCP}"/></fixed>
    <fixed name="middle_pip"><joint joint="middle_pip" coef="{ARM_PIP}"/></fixed>
    <fixed name="middle_abd"><joint joint="middle_abd" coef="{ARM_ABD}"/></fixed>
    <fixed name="ring_flex"><joint joint="ring_mcp" coef="{ARM_MCP}"/><joint joint="ring_pip" coef="{ARM_PIP}"/></fixed>
    <fixed name="little_flex"><joint joint="little_mcp" coef="{ARM_MCP}"/><joint joint="little_pip" coef="{ARM_PIP}"/></fixed>
    <fixed name="spread"><joint joint="little_abd" coef="{ARM_ABD}"/></fixed>
    <fixed name="thumb_opp"><joint joint="thumb_opp" coef="{ARM_THUMB}"/></fixed>
    <fixed name="thumb_cmc"><joint joint="thumb_cmc" coef="{ARM_THUMB}"/></fixed>
    <fixed name="thumb_flex"><joint joint="thumb_mcp" coef="{ARM_THUMB}"/></fixed>
    <fixed name="wrist_pitch"><joint joint="wrist_pitch" coef="{ARM_WRIST}"/></fixed>
    <fixed name="wrist_yaw"><joint joint="wrist_yaw" coef="{ARM_WRIST}"/></fixed>
  </tendon>

  <actuator>
"""
    for tn in ["index_mcp", "index_pip", "index_abd", "middle_mcp", "middle_pip", "middle_abd",
               "ring_flex", "little_flex", "spread", "thumb_opp", "thumb_cmc", "thumb_flex"]:
        xml += f'    <position name="{tn}" tendon="{tn}" kp="6000" kv="8" forcerange="-{T_FINGER} {T_FINGER}"/>\n'
    for tn in ["wrist_pitch", "wrist_yaw"]:
        xml += f'    <position name="{tn}" tendon="{tn}" kp="6000" kv="12" forcerange="-{T_WRIST} {T_WRIST}"/>\n'
    for j in ["arm_x", "arm_y", "arm_z"]:
        xml += f'    <position name="{j}" joint="{j}" kp="400" kv="25" forcerange="-20 20"/>\n'
    xml += "  </actuator>\n</mujoco>\n"
    open(OUT, "w", encoding="utf-8").write(xml)


HAND_ACTUATORS = ["index_mcp", "index_pip", "index_abd", "middle_mcp", "middle_pip", "middle_abd",
                  "ring_flex", "little_flex", "spread", "thumb_opp", "thumb_cmc", "thumb_flex",
                  "wrist_pitch", "wrist_yaw"]

if __name__ == "__main__":
    build()
    print(f"已生成 {os.path.basename(OUT)}：14 个手部电机 + 3 个手臂滑动关节")
