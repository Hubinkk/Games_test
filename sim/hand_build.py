"""
55 cm 人偶 · 绳驱手（MuJoCo 模型生成器）

生成 hand55.xml，并把同一套参数写进 hand_params.json，网页演示直接用这份参数。

结构：
  - 手掌约 20 × 22 mm，四指各 3 节，拇指 2 节（掌根处固定为对掌姿态）
  - 每根手指：掌侧一根肌腱线（屈指），背侧弹性带（伸指，用关节弹簧表示）
  - 4 个微型电机：拇指 / 食指 / 中指 / 无名指＋小指
  - 电机 = 绳轮，用"肌腱长度位置伺服"表示，只能拉不能推，拉力上限 30 N
  - 坐标：手指朝 +x，掌心朝 -z，手指弯曲时指尖向 -z 转动
"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))

FINGERS = {
    # name: base y (m), segment lengths (m), moment arms (m)
    "index":  {"y": -0.0075, "len": [0.0070, 0.0050, 0.0040], "r": [0.0022, 0.0018, 0.0014]},
    "middle": {"y": -0.0025, "len": [0.0075, 0.0055, 0.0042], "r": [0.0022, 0.0018, 0.0014]},
    "ring":   {"y":  0.0025, "len": [0.0070, 0.0050, 0.0040], "r": [0.0022, 0.0018, 0.0014]},
    "little": {"y":  0.0075, "len": [0.0056, 0.0042, 0.0035], "r": [0.0020, 0.0016, 0.0012]},
    "thumb":  {"y": -0.0110, "len": [0.0080, 0.0060], "r": [0.0020, 0.0016]},
}
MOTORS = {"thumb": ["thumb"], "index": ["index"], "middle": ["middle"], "ringlittle": ["ring", "little"]}
MOTOR_NAMES = {"thumb": "拇指", "index": "食指", "middle": "中指", "ringlittle": "无名指＋小指"}
MOTOR_PLACE = {"thumb": "前臂", "index": "前臂", "middle": "上臂（线穿过肘）", "ringlittle": "上臂（线穿过肘）"}
FINGER_R = 0.0022          # 手指半径
T_REF, THETA_REF = 10.0, 1.3  # 拉力 10 N 时各关节约弯 1.3 rad → 由此定弹簧刚度
JOINT_MAX = [1.6, 1.8, 1.4]
T_MAX = 30.0               # 电机经绳轮能提供的最大拉力（N）
SPOOL_R = 0.003            # 绳轮半径（m）
THUMB_BASE = {"pos": [0.012, -0.013, -0.0068], "euler": [0.0, 0.4, 0.4]}  # 对掌姿态（掌骨固定，由搜索确定朝向）


def stiffness(r):
    return T_REF * r / THETA_REF


def finger_xml(name, f):
    n = len(f["len"])
    # 不动的"指根"：放第一个关节的滑轮，下面才是带关节的第一节
    if name == "thumb":
        head = f'<body name="thumb_base" pos="{" ".join(map(str, THUMB_BASE["pos"]))}" euler="{" ".join(map(str, THUMB_BASE["euler"]))}">\n'
    else:
        head = f'<body name="{name}_base" pos="0.020 {f["y"]} 0">\n'
    r0 = f["r"][0]
    head += (f'  <geom name="{name}_pulley0" type="cylinder" size="{r0} 0.0015" euler="1.5708 0 0"'
             f' contype="0" conaffinity="0" density="0" rgba="0.82 0.6 0.15 1"/>\n'
             f'  <site name="{name}_side0" pos="0 0 {-3*r0}"/>\n'
             f'<body name="{name}0">\n')
    x = head
    close = "</body>\n</body>\n"
    for i in range(n):
        L, r = f["len"][i], f["r"][i]
        k = stiffness(r)
        jx = f["len"][i-1] if i > 0 else 0.0
        pulley = (f'  <geom name="{name}_pulley{i}" type="cylinder" size="{r} 0.0015" pos="{jx} 0 0" euler="1.5708 0 0"'
                  f' contype="0" conaffinity="0" density="0" rgba="0.82 0.6 0.15 1"/>\n'
                  f'  <site name="{name}_side{i}" pos="{jx} 0 {-3*r}"/>\n')
        if i > 0:
            x += pulley
            x += f'<body name="{name}{i}" pos="{f["len"][i-1]} 0 0">\n'
            close += "</body>\n"
        x += (f'  <joint name="{name}_j{i}" axis="0 1 0" range="0 {JOINT_MAX[i]}" stiffness="{k:.5g}" springref="0"/>\n'

              f'  <geom type="capsule" fromto="0 0 0 {L} 0 0" size="{FINGER_R if name != "little" else FINGER_R*0.9}"/>\n'
              f'  <site name="{name}_s{i}a" pos="{L*0.25} 0 {-r}"/>\n'
              f'  <site name="{name}_s{i}b" pos="{L*0.75} 0 {-r}"/>\n')
        if i == n - 1:
            x += f'  <site name="{name}_tip" pos="{L} 0 0"/>\n'
    return x + close


def build():
    bodies = "".join(finger_xml(n, f) for n, f in FINGERS.items())
    tendons, acts = "", ""
    # 理想腱鞘滑车：每个关节的力臂恒定为 r，用"固定肌腱"表示
    # 肌腱长度 = Σ(-r_i × 关节角_i)，弯曲时变短，拉线就会让手指弯曲
    for n, f in FINGERS.items():
        joints = "".join(f'<joint joint="{n}_j{i}" coef="{-r:.5g}"/>' for i, r in enumerate(f["r"]))
        tendons += f'    <fixed name="{n}">{joints}</fixed>\n'
    for m, fs in MOTORS.items():
        for n in fs:
            acts += f'    <position name="{n}_motor" tendon="{n}" kp="20000" kv="20" forcerange="-{T_MAX} 0" ctrlrange="-0.03 0"/>\n'
    palm_sites = ""
    for n, f in FINGERS.items():
        y = f["y"] if n != "thumb" else -0.008
        palm_sites += f'      <site name="wrist_{n}" pos="-0.012 {y*0.6} -0.0035"/>\n'
        if n != "thumb":
            palm_sites += f'      <site name="palm_{n}" pos="0.017 {y} -0.0035"/>\n'
    xml = f"""<mujoco model="hand55">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.0005" integrator="implicitfast"/>
  <default>
    <joint damping="0.0004" armature="2e-7"/>
    <geom contype="1" conaffinity="1" density="1100" rgba="0.95 0.85 0.79 1"/>
    <site size="0.0006" rgba="0.2 0.5 0.9 1"/>
  </default>
  <worldbody>
    <light pos="0 0 0.3" dir="0 0 -1"/>
    <body name="palm" pos="0 0 0.1">
      <geom type="box" pos="0.0055 0 0" size="0.0115 0.011 0.004"/>
{palm_sites}{bodies}    </body>
    <body name="target" pos="0.020 -0.0025 0.0915">
      <geom name="target" type="cylinder" size="0.005 0.012" euler="1.5708 0 0" rgba="0.5 0.7 0.9 0.4" contype="2" conaffinity="2"/>
    </body>
  </worldbody>
  <tendon>
{tendons}  </tendon>
  <actuator>
{acts}  </actuator>
</mujoco>
"""
    open(os.path.join(HERE, "hand55.xml"), "w", encoding="utf-8").write(xml)
    params = {
        "fingers": {n: {**f, "k": [stiffness(r) for r in f["r"]], "max": JOINT_MAX[:len(f["len"])]} for n, f in FINGERS.items()},
        "motors": {m: {"fingers": fs, "name": MOTOR_NAMES[m], "place": MOTOR_PLACE[m]} for m, fs in MOTORS.items()},
        "finger_r": FINGER_R, "t_max": T_MAX, "spool_r": SPOOL_R, "thumb_base": THUMB_BASE,
    }
    return params


def main():
    p = build()
    json.dump(p, open(os.path.join(HERE, "hand_params.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("已生成 hand55.xml 和 hand_params.json")


if __name__ == "__main__":
    main()
