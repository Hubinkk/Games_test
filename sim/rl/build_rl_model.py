"""
训练用的简化身体：和第三版整体模型同样的尺寸、重量、关节范围和力气上限，去掉算起来很慢的部分。

用法：
    python build_rl_model.py        # 生成 huaran_rl.xml，并用参考动作检查它和整体模型像不像

和整体模型（../figure55_full.xml）的差别：
  - 手指去掉：每只手换成一块手掌（同样的重量，6.8 g）。显卡上要同时算几千个身体，手指的 40 多个小关节太慢。
  - 膝盖：整体模型用两根线拉；这里换成同样力气上限（2.8 N·m）的普通关节电机。
  - 仿真步长 0.5 ms → 2 ms，接触用简单的摩擦模型。
其余不变：关节范围、扭矩上限、腰两段联动、4 关节脖子、重量分布。
"""
import os
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "figure55_full.xml")
OUT = os.path.join(HERE, "huaran_rl.xml")
HAND_MASS = 0.0068


def build():
    tree = ET.parse(SRC)
    root = tree.getroot()
    root.set("model", "huaran_rl")
    opt = root.find("option")
    opt.attrib.clear()
    opt.attrib.update(timestep="0.002", integrator="implicitfast", iterations="6", ls_iterations="10")
    # 手：去掉手指，换成手掌
    for parent in root.iter("body"):
        for child in list(parent):
            if child.tag == "body" and child.get("name") in ("L_hand", "R_hand"):
                side = child.get("name")[0].lower()
                pos = [float(v) for v in child.get("pos").split()]
                parent.remove(child)
                # 和整体模型里手掌那块硬的部分同样大小、同样位置（手指是软的，会弯，这里不算进去）
                ET.SubElement(parent, "geom", name=f"{side}_palm", type="box", pos=f"0 0 {pos[2] - 0.006:.4f}",
                              size="0.011 0.0035 0.0115", mass=str(HAND_MASS), rgba="0.95 0.85 0.79 1")
    # 肌腱、手的联动和碰撞排除都去掉；腰的联动留下
    for tag in ("tendon", "contact"):
        for e in root.findall(tag):
            root.remove(e)
    eq = root.find("equality")
    for e in list(eq):
        if e.get("joint1") != "waist_pitch2":
            eq.remove(e)
    act = root.find("actuator")
    for e in list(act):
        n = e.get("name", "")
        if n[:2] in ("L_", "R_"):
            act.remove(e)
        elif n.endswith("_ham"):
            act.remove(e)
        elif n.endswith("_quad"):          # 换成膝关节电机（和 tendon_knee.py 里的控制同样的刚度和上限）
            side = n[0]
            e.tag = "position"
            e.attrib.clear()
            e.attrib.update(name=f"{side}_knee", joint=f"{side}_knee", kp="20", kv="0.5", forcerange="-2.8 2.8")
    # 没有用到的"手"的默认设置去掉（里面的接触参数是给手指用的）
    for d in root.iter("default"):
        for c in list(d):
            if c.tag == "default" and c.get("class") == "hand":
                d.remove(c)
    for b in root.iter("body"):
        if b.get("childclass") == "hand":
            del b.attrib["childclass"]
    ET.indent(tree, space="  ")
    tree.write(OUT, encoding="utf-8", xml_declaration=False)
    return OUT


if __name__ == "__main__":
    import mujoco
    p = build()
    m = mujoco.MjModel.from_xml_path(p)
    print(f"已生成 {os.path.basename(p)}：总质量 {sum(m.body_mass):.2f} kg，关节 {m.njnt}，电机 {m.nu}，几何体 {m.ngeom}，步长 {m.opt.timestep*1000:.0f} ms")
    print("电机：", [m.actuator(i).name for i in range(m.nu)])
