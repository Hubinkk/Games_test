"""
走路训练用的身体：在起身用的简化身体（huaran_rl.xml）上再减一步，只留两只脚和地面的碰撞。

用法：
    python build_walk_model.py        # 读 huaran_rl.xml，生成 huaran_walk.xml

为什么：显卡上每个身体每一步都要把"可能碰到地面的每一块"算一遍，不管它实际碰没碰到。
走路时只有脚碰地；别的部位碰地就是摔了，训练里直接算这一局结束（见 walk_env.py），用不着算碰撞。
其余（尺寸、重量、关节范围、力气上限、摩擦锥 elliptic、步长 2 ms）和 huaran_rl.xml 完全一样。
检验一律放回完整身体做（eval_walk.py），那里所有部位都会碰撞。
"""
import os
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "huaran_rl.xml")
OUT = os.path.join(HERE, "huaran_walk.xml")
KEEP = ("floor", "l_foot", "r_foot")


def build():
    tree = ET.parse(SRC)
    root = tree.getroot()
    root.set("model", "huaran_walk")
    n = 0
    for g in root.find("worldbody").iter("geom"):
        if g.get("name") not in KEEP:
            g.set("contype", "0")
            g.set("conaffinity", "0")
            n += 1
    ET.indent(tree, space="  ")
    tree.write(OUT, encoding="utf-8", xml_declaration=False)
    return OUT, n


if __name__ == "__main__":
    import mujoco
    p, n = build()
    m = mujoco.MjModel.from_xml_path(p)
    hit = [m.geom(g).name for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]
    print(f"已生成 {os.path.basename(p)}：总质量 {sum(m.body_mass):.2f} kg，电机 {m.nu}，关掉碰撞的几何体 {n} 个，还会碰撞的：{hit}")
