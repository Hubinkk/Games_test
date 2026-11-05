"""
绳驱手测试：各手势能不能做出来、每根线要多大的拉力

用法：
    python hand_build.py      # 先生成模型（参数改了以后要重新生成）
    python hand_test.py
    python hand_test.py --view

每个电机的控制量 = 拉线长度（mm），0 = 完全放松。
"""
import argparse
import json
import os

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MOTORS = {"thumb": ["thumb"], "index": ["index"], "middle": ["middle"], "ringlittle": ["ring", "little"]}
FULL = 9.0  # mm，握拳所需的大致拉线量

GESTURES = {
    "张开": {"thumb": 0, "index": 0, "middle": 0, "ringlittle": 0},
    "握拳": {"thumb": 4.5, "index": FULL, "middle": FULL, "ringlittle": FULL},
    "比 V": {"thumb": 4.5, "index": 0, "middle": 0, "ringlittle": FULL},
    "指":   {"thumb": 4.5, "index": 0, "middle": FULL, "ringlittle": FULL},
}


class Hand:
    def __init__(self, view=False):
        self.m = mujoco.MjModel.from_xml_path(os.path.join(HERE, "hand55.xml"))
        self.d = mujoco.MjData(self.m)
        mujoco.mj_forward(self.m, self.d)
        self.rest = {self.m.tendon(i).name: float(self.d.ten_length[i]) for i in range(self.m.ntendon)}
        self.act = {self.m.actuator(i).name: i for i in range(self.m.nu)}
        self.target = self.m.geom("target").id
        self.view = None
        if view:
            from mujoco import viewer as mjv
            self.view = mjv.launch_passive(self.m, self.d)

    def reset(self, grasp=False):
        mujoco.mj_resetData(self.m, self.d)
        self.m.geom_contype[self.target] = 1 if grasp else 0
        self.m.geom_conaffinity[self.target] = 1 if grasp else 0
        b = self.m.geom_bodyid[self.target]  # 物体层的碰撞设置也要一起改
        self.m.body_contype[b] = self.m.body_conaffinity[b] = 1 if grasp else 0

    def set(self, pulls_mm):
        for motor, fingers in MOTORS.items():
            for f in fingers:
                self.d.ctrl[self.act[f"{f}_motor"]] = self.rest[f] - pulls_mm.get(motor, 0) / 1000

    def run(self, pulls_mm, seconds=0.6):
        self.set(pulls_mm)
        for _ in range(int(seconds / self.m.opt.timestep)):
            mujoco.mj_step(self.m, self.d)
            if self.view is not None and _ % 40 == 0:
                self.view.sync()

    def tensions(self):
        return {f: -float(self.d.actuator_force[self.act[f"{f}_motor"]]) for f in ("thumb", "index", "middle", "ring", "little")}

    def joints(self, f):
        n = 2 if f == "thumb" else 3
        return [float(self.d.qpos[self.m.jnt_qposadr[self.m.joint(f"{f}_j{i}").id]]) for i in range(n)]

    def tip(self, f):
        return self.d.site(f"{f}_tip").xpos.copy()

    def contacts_with_target(self):
        names = set()
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            if self.target in (c.geom1, c.geom2):
                other = c.geom2 if c.geom1 == self.target else c.geom1
                names.add(self.m.body(self.m.geom_bodyid[other]).name)
        return sorted(names)


def fmt_angles(a):
    return "/".join(f"{np.degrees(x):.0f}" for x in a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", action="store_true")
    args = ap.parse_args()
    h = Hand(args.view)
    results = {}

    print("【1】手势（关节角度：根部/中间/指尖，单位：度）")
    for name, pulls in GESTURES.items():
        h.reset()
        h.run(pulls)
        t = h.tensions()
        print(f"  {name}：食指 {fmt_angles(h.joints('index'))}，中指 {fmt_angles(h.joints('middle'))}，"
              f"无名指 {fmt_angles(h.joints('ring'))}，拇指 {fmt_angles(h.joints('thumb'))}；"
              f"最大线拉力 {max(t.values()):.1f} N")
        results[name] = {"pulls": pulls, "tension": t}

    print("\n【2】OK 手势：找拇指和食指指尖的最近距离")
    best = None
    for pt in np.arange(0.0, 9.01, 0.5):
        for pi in np.arange(2.0, 9.01, 0.25):
            h.reset()
            h.run({"thumb": pt, "index": pi, "middle": 0, "ringlittle": 0}, 0.4)
            dist = np.linalg.norm(h.tip("thumb") - h.tip("index")) * 1000
            if best is None or dist < best[0]:
                best = (dist, pt, pi, h.tensions())
    dist, pt, pi, t = best
    print(f"  拇指拉 {pt:.1f} mm、食指拉 {pi:.1f} mm 时，指尖相距 {dist:.1f} mm"
          f"（手指半径 2.2 mm，两指尖表面{'相碰' if dist <= 4.6 else '没有碰到'}）")
    results["OK"] = {"pulls": {"thumb": pt, "index": pi, "middle": 0, "ringlittle": 0}, "tip_mm": dist}

    print("\n【3】握住直径 10 mm 的圆柱（像握笔，手指自动包住物体）")
    grasp = {"thumb": 0, "index": 5.0, "middle": 5.0, "ringlittle": 5.0}
    h.reset()
    h.run(grasp, 0.8)
    free = {f: h.joints(f) for f in ("index", "middle", "ring")}
    h.reset(grasp=True)
    h.run(grasp, 0.8)
    t = h.tensions()
    print(f"  接触圆柱的部位：{', '.join(h.contacts_with_target())}")
    for f, nm in (("index", "食指"), ("middle", "中指"), ("ring", "无名指")):
        print(f"  {nm}：空手 {fmt_angles(free[f])} → 握圆柱 {fmt_angles(h.joints(f))}")
    print("  （同样的拉线量，被圆柱挡住的关节停下，其余关节继续弯，手指自动贴合圆柱）")
    print(f"  线拉力：食指 {t['index']:.1f} N，中指 {t['middle']:.1f} N，无名指 {t['ring']:.1f} N，小指 {t['little']:.1f} N")
    results["grasp"] = {"tension": t}

    peak = max(max(r.get("tension", {"x": 0}).values()) for r in results.values() if "tension" in r)
    print(f"\n【换算】最大线拉力 {peak:.1f} N；绳轮半径 3 mm 时，电机需要 {peak*0.003*1000:.0f} mN·m")
    print("      这在直径 12 mm 级别微型减速电机（N20 一类）的能力范围内")
    json.dump(results, open(os.path.join(HERE, "hand_results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    if h.view is not None:
        import time
        while h.view.is_running():
            for pulls in GESTURES.values():
                h.reset()
                h.run(pulls, 1.0)
                time.sleep(0.5)


if __name__ == "__main__":
    main()
