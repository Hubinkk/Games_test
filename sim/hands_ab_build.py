"""
55 cm 人偶 · 双手十指打字 · A+B 混合方案（每只手 6 个电机 + 6 个锁）

用法：
    python hands_ab_build.py      # 生成 hands_ab.xml

和 14 电机版（hands_build.py）的区别：
  A（耦合）
    - 食指、中指：两根线从指根左右两侧穿过（差动），同时拉 = 弯曲，单拉一侧 = 偏向那一侧；
      中间节与根部 1:1 连杆联动，末节再按 0.75 联动 → 每根手指 2 个电机
    - 拇指根部弯曲固定（cmc 不动），拇指只保留"对掌摆动"和"弯曲"两个动作
  B（复用）
    - 一个共享电机，经离合器分时驱动：无名指弯曲、小指弯曲、张开、拇指对掌、拇指弯曲
    - 没被选中的输出由锁保持不动；手腕上下平时锁住
  每只手：食指 2 + 中指 2 + 手腕左右 1 + 共享 1 = 6 个电机；锁 6 个
几何、键盘、手臂与 hands_build.py 完全相同。
"""
import os

import hand2_build as hb
import hands_build as base
from hands_build import SIDES, key_layout  # noqa: F401  （供 hands_demo 使用）

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "hands_ab.xml")

TENDONS = {  # 执行器: [(关节, 系数)]
    "index_a": [("index_mcp", hb.ARM_MCP), ("index_abd", hb.ARM_ABD)],
    "index_b": [("index_mcp", hb.ARM_MCP), ("index_abd", -hb.ARM_ABD)],
    "middle_a": [("middle_mcp", hb.ARM_MCP), ("middle_abd", hb.ARM_ABD)],
    "middle_b": [("middle_mcp", hb.ARM_MCP), ("middle_abd", -hb.ARM_ABD)],
    "ring_flex": [("ring_mcp", hb.ARM_MCP), ("ring_pip", hb.ARM_PIP)],
    "little_flex": [("little_mcp", hb.ARM_MCP), ("little_pip", hb.ARM_PIP)],
    "spread": [("little_abd", hb.ARM_ABD)],
    "thumb_opp": [("thumb_opp", hb.ARM_THUMB)],
    "thumb_flex": [("thumb_mcp", hb.ARM_THUMB)],
    "wrist_yaw": [("wrist_yaw", hb.ARM_WRIST)],
    "wrist_pitch": [("wrist_pitch", hb.ARM_WRIST)],
}
DEDICATED = ["index_a", "index_b", "middle_a", "middle_b", "wrist_yaw"]       # 各自有电机
MUXED = ["ring_flex", "little_flex", "spread", "thumb_opp", "thumb_flex"]      # 共享一个电机，分时驱动
LOCKED = ["wrist_pitch"]                                                       # 只有锁


def build():
    keys = key_layout()
    P = hb.KEY_PITCH
    key_geoms = "".join(
        f'      <geom name="key_{hb.key_name(ch)}" type="box" pos="{x} {y} 0.00025" '
        f'size="{hb.KEY_D/2} {(5 * P - (P - hb.KEY_W)) / 2 if ch == " " else hb.KEY_W/2} 0.00025" rgba="0.82 0.86 0.92 1"/>\n'
        for ch, (x, y) in keys.items())
    hands = "".join(base.hand_xml(f"{k}_", s) for k, s in SIDES.items())
    excl = "".join(f'<exclude body1="{k}_palm" body2="{k}_{b}1"/>' for k in SIDES for b in ("index", "middle", "ring", "little", "thumb"))
    eq, ten, act = "", "", ""
    for k in SIDES:
        p = f"{k}_"
        for n in ("index", "middle", "ring", "little"):
            eq += f'    <joint joint1="{p}{n}_dip" joint2="{p}{n}_pip" polycoef="0 0.75 0 0 0"/>\n'
        for n in ("index", "middle"):  # A：中间节与根部 1:1 连杆联动
            eq += f'    <joint joint1="{p}{n}_pip" joint2="{p}{n}_mcp" polycoef="0 1.0 0 0 0"/>\n'
        eq += f'    <joint joint1="{p}thumb_ip" joint2="{p}thumb_mcp" polycoef="0 1.0 0 0 0"/>\n'
        eq += f'    <joint joint1="{p}ring_abd" joint2="{p}little_abd" polycoef="0 0.4 0 0 0"/>\n'
        eq += f'    <joint joint1="{p}thumb_cmc" polycoef="0 0 0 0 0"/>\n'  # A：拇指根部弯曲固定
        eq += f'    <joint joint1="{p}wrist_pitch" polycoef="0 0 0 0 0"/>\n'  # B：手腕上下由锁固定（刚性，不会被撞开）
        for tn, lst in TENDONS.items():
            ten += f'    <fixed name="{p}{tn}">' + "".join(f'<joint joint="{p}{j}" coef="{c}"/>' for j, c in lst) + "</fixed>\n"
            fr = hb.T_WRIST if tn.startswith("wrist") else hb.T_FINGER
            kv = 12 if tn.startswith("wrist") else 8
            act += f'    <position name="{p}{tn}" tendon="{p}{tn}" kp="6000" kv="{kv}" forcerange="-{fr} {fr}"/>\n'
        for j in ("arm_x", "arm_y", "arm_z"):
            act += f'    <position name="{p}{j}" joint="{p}{j}" kp="400" kv="25" forcerange="-20 20"/>\n'
    xml = f"""<mujoco model="hands_ab">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.0005" integrator="implicitfast" cone="elliptic" noslip_iterations="3"/>
  <default>
    <joint damping="0.0004" armature="2e-7"/>
    <geom contype="1" conaffinity="1" density="1100" friction="1.0 0.01 0.001" solref="0.004 1" solimp="0.95 0.99 0.0005"/>
    <site size="0.0005" rgba="0.2 0.5 0.9 1"/>
    <!-- 联动是刚性连杆：约束要足够硬，否则很轻的指节会被关节回弹弹簧拉偏 -->
    <equality solref="0.001 1"/>
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
    print(f"已生成 {os.path.basename(OUT)}：每只手 {len(DEDICATED) + 1} 个电机（含 1 个共享电机）+ {len(MUXED) + len(LOCKED)} 个锁")
