"""
把 figure55.xml（MuJoCo）转换成 URDF，供 Isaac Sim / Isaac Lab 的 URDF 导入器使用。

用法：
    python mjcf_to_urdf.py                 # 生成 figure55.urdf
    python mjcf_to_urdf.py --check         # 生成后用 MuJoCo 重新读入，核对质量和关节

说明：
  - 质量、质心、惯量直接取自 MuJoCo 编译后的模型，所以和仿真完全一致
  - URDF 每个连杆只能有一个关节，所以多轴关节（髋 3 轴、肩 3 轴、腰 2 轴）
    会拆成几个几乎没有质量的中间连杆
  - 几何体统一导出为盒子、圆柱或球（URDF 没有胶囊体，胶囊体近似成圆柱）
  - 扭矩上限写在 <limit effort>，Isaac Lab 里也要在执行器配置中设置同样的值
"""
import argparse
import math
import os

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "figure55.xml")
DST = os.path.join(HERE, "figure55.urdf")
TINY = 1e-4  # 中间连杆的质量


def quat_to_rpy(q):
    w, x, y, z = q
    r = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    p = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return r, p, yw


def f(v):
    return " ".join(f"{x:.6g}" for x in v)


def inertial_xml(m, b):
    mass = float(m.body_mass[b])
    ipos = m.body_ipos[b]
    rpy = quat_to_rpy(m.body_iquat[b])
    I = m.body_inertia[b]
    return (f'    <inertial><origin xyz="{f(ipos)}" rpy="{f(rpy)}"/><mass value="{mass:.6g}"/>'
            f'<inertia ixx="{I[0]:.6g}" iyy="{I[1]:.6g}" izz="{I[2]:.6g}" ixy="0" ixz="0" iyz="0"/></inertial>\n')


def geom_xml(m, g, tag):
    t = m.geom_type[g]
    size = m.geom_size[g]
    pos = m.geom_pos[g]
    rpy = quat_to_rpy(m.geom_quat[g])
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        shape = f'<box size="{f(2 * size)}"/>'
    elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
        shape = f'<sphere radius="{size[0]:.6g}"/>'
    elif t in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
        L = 2 * size[1] + (2 * size[0] if t == mujoco.mjtGeom.mjGEOM_CAPSULE else 0)
        shape = f'<cylinder radius="{size[0]:.6g}" length="{L:.6g}"/>'
    elif t == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        shape = f'<sphere radius="{max(size):.6g}"/>'
    else:
        return ""
    return f'    <{tag}><origin xyz="{f(pos)}" rpy="{f(rpy)}"/><geometry>{shape}</geometry></{tag}>\n'


def convert():
    m = mujoco.MjModel.from_xml_path(SRC)
    act_limit = {}
    for i in range(m.nu):
        act_limit[int(m.actuator_trnid[i][0])] = float(m.actuator_forcerange[i][1])

    out = ['<?xml version="1.0"?>\n<robot name="figure55">\n']
    link_of_body = {}

    for b in range(1, m.nbody):
        name = m.body(b).name
        parent = int(m.body_parentid[b])
        jnts = [j for j in range(m.njnt) if m.jnt_bodyid[j] == b and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
        # 主连杆：带质量和几何体
        body_xml = [f'  <link name="{name}">\n', inertial_xml(m, b)]
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != b:
                continue
            body_xml.append(geom_xml(m, g, "visual"))
            if m.geom_contype[g] or m.geom_conaffinity[g]:
                body_xml.append(geom_xml(m, g, "collision"))
        body_xml.append("  </link>\n")

        if parent == 0:  # 根（骨盆），自由关节由 Isaac 处理
            out += body_xml
            link_of_body[b] = name
            continue

        prev = link_of_body[parent]
        origin = f'<origin xyz="{f(m.body_pos[b])}" rpy="{f(quat_to_rpy(m.body_quat[b]))}"/>'
        if not jnts:
            out += body_xml
            out.append(f'  <joint name="{name}_fixed" type="fixed"><parent link="{prev}"/><child link="{name}"/>{origin}</joint>\n')
            link_of_body[b] = name
            continue

        for k, j in enumerate(jnts):
            jname = m.joint(j).name
            last = k == len(jnts) - 1
            child = name if last else f"{name}__{jname}"
            if not last:
                out.append(f'  <link name="{child}"><inertial><mass value="{TINY}"/>'
                           f'<inertia ixx="1e-8" iyy="1e-8" izz="1e-8" ixy="0" ixz="0" iyz="0"/></inertial></link>\n')
            else:
                out += body_xml
            o = origin if k == 0 else '<origin xyz="0 0 0" rpy="0 0 0"/>'
            lo, hi = m.jnt_range[j]
            eff = act_limit.get(j, 1.0)
            out.append(
                f'  <joint name="{jname}" type="revolute"><parent link="{prev}"/><child link="{child}"/>{o}'
                f'<axis xyz="{f(m.jnt_axis[j])}"/><limit lower="{lo:.6g}" upper="{hi:.6g}" effort="{eff:.6g}" velocity="12"/>'
                f'<dynamics damping="{m.dof_damping[m.jnt_dofadr[j]]:.6g}" friction="{m.dof_frictionloss[m.jnt_dofadr[j]]:.6g}"/></joint>\n')
            prev = child
        link_of_body[b] = name

    out.append("</robot>\n")
    with open(DST, "w", encoding="utf-8") as fh:
        fh.write("".join(out))
    return m


def check(src):
    u = mujoco.MjModel.from_xml_path(DST)
    hinge = lambda mm: sorted(mm.joint(j).name for j in range(mm.njnt) if mm.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE)
    print(f"MuJoCo 原模型：质量 {sum(src.body_mass):.3f} kg，转动关节 {len(hinge(src))} 个")
    import re
    with open(DST, encoding="utf-8") as fh:
        urdf_mass = sum(float(x) for x in re.findall(r'<mass value="([^"]+)"', fh.read()))
    # 注意：MuJoCo 读 URDF 时会把根连杆并入世界，所以这里直接从文件里统计质量
    print(f"URDF 文件：    质量 {urdf_mass:.3f} kg，转动关节 {len(hinge(u))} 个")
    print("关节名一致" if hinge(src) == hinge(u) else "关节名不一致！")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    src = convert()
    print(f"已生成 {os.path.basename(DST)}")
    if a.check:
        check(src)
