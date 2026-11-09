"""
55 cm 人偶 · "灵活"测试：在整体模型上测活动范围、速度、平衡、走路和摔倒后爬起。

用法：
    python full_build.py
    python getup.py            # 先搜索起身动作（约 10 分钟），结果存在 getup_best.json
    python agility.py          # 跑全部测试，打印结果，导出 ../agility_frames.js 供网页回放

所有测试用同一个整体模型（figure55_full.xml）、同样的扭矩上限和同样的控制器
（重心反馈平衡、静步行），和 full_demo.py 一致。
模型本身只限制扭矩；这里另外给每个关节加上电机转速上限（10 rad/s，常见微型舵机和膝关节 v1 的目标），
并记录每个测试里最快的关节。
"""
import json
import math
import os

import mujoco
import numpy as np

import figure_stress
import figure_test
import full_build as fb
import full_demo as FD
import getup
import walk
from balance import Balance
from figure_test import run_phase, stand_pose

HERE = os.path.dirname(os.path.abspath(__file__))
BIG = ("hip", "knee", "ankle", "waist")
SPEED_LIMIT = {"big": 10.0, "small": 10.0}   # rad/s：膝关节 v1 的目标、常见微型舵机约 8–10 rad/s


class VelTracker:
    def __init__(self, sim, rec=None):
        self.sim, self.rec, self.vmax = sim, rec, {}
        m = sim.m
        self.body = [(m.joint(j).name, m.jnt_dofadr[j]) for j in range(m.njnt)
                     if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and m.joint(j).name[:2] not in ("L_", "R_")]

    def __call__(self):
        d = self.sim.d
        for n, a in self.body:
            v = abs(float(d.qvel[a]))
            if v > self.vmax.get(n, 0.0):
                self.vmax[n] = v
        if self.rec:
            self.rec()

    def worst(self):
        n = max(self.vmax, key=self.vmax.get)
        return n, self.vmax[n]


def add_speed_limit(sim, w=SPEED_LIMIT["big"]):
    """给每个关节加上电机的转速上限：目标角度每一步最多变化 w·dt（像舵机那样以最高转速追目标）。"""
    orig_set, orig_reset = sim.set_targets, sim.reset
    dt = sim.m.opt.timestep
    last = {}

    def cur(name):
        try:
            return joint_q(sim, name)
        except Exception:
            return None

    def set_targets(pose):
        out = {}
        for k, v in pose.items():
            if k.startswith("hand_"):
                out[k] = v
                continue
            p = last.get(k)
            if p is None:
                p = cur(k)
                if p is None:
                    out[k] = v
                    continue
            nv = p + max(-w * dt, min(w * dt, v - p))
            last[k] = nv
            out[k] = nv
        orig_set(out)

    def reset():
        last.clear()
        orig_reset()

    sim.set_targets, sim.reset = set_targets, reset


def attach(sim, record=False):
    rec = FD.Recorder(sim) if record else None
    vt = VelTracker(sim, rec)
    sim.recorder = vt
    return vt, rec


def joint_q(sim, name):
    return float(sim.d.qpos[sim.m.jnt_qposadr[sim.m.joint(name).id]])


# ---------------- 1. 活动范围 ----------------
NEEDS = [  # (动作, 关节, 方向, 需要的角度°, 说明)
    ("举手过头", "l_shoulder_roll", +1, 170, "手臂从身体侧面举到头顶"),
    ("左右转头", "neck_yaw", +1, 80, "人约 ±80°"),
    ("低头", "neck_pitch", +1, 50, "看自己的手和脚边"),
    ("抬头", "neck_pitch", -1, 45, "仰头看高处"),
    ("弯腰", "waist_pitch", +1, 70, "弯腰捡东西、鞠躬（人约 90°）"),
    ("扭腰", "waist_yaw", +1, 40, "回头、转身"),
    ("深蹲 / 跪坐", "l_knee", +1, 145, "跪坐、蹲到底"),
    ("抬腿", "l_hip_pitch", -1, 120, "抱膝、爬起来时收腿"),
    ("勾脚", "l_ankle_pitch", -1, 30, "深蹲时脚跟不离地"),
    ("屈肘", "l_elbow", -1, 140, "手碰到肩膀"),
]


def rom(m):
    out = []
    for act, j, sign, need, note in NEEDS:
        lo, hi = m.jnt_range[m.joint(j).id]
        have = math.degrees(hi if sign > 0 else -lo)
        out.append(dict(act=act, joint=j, need=need, have=round(have), ok=have >= need - 2, note=note))
    return out


# ---------------- 2. 速度 ----------------
def stand(sim, bal, seconds=1.0, extra=None):
    fn = (lambda t: bal.adjust(dict(stand_pose(), **(extra or {}))))
    return run_phase(sim, seconds, fn)


def time_to(sim, bal, joint, start, goal, extra=None, seconds=2.0):
    sim.reset(); bal.prev = None
    stand(sim, bal, 1.0, {joint: start, **(extra or {})})
    dt = sim.m.opt.timestep
    t90 = None
    for k in range(int(seconds / dt)):
        sim.set_targets(bal.adjust(dict(stand_pose(), **(extra or {}), **{joint: goal})))
        sim.step()
        q = joint_q(sim, joint)
        if t90 is None and abs(q - start) >= 0.9 * abs(goal - start):
            t90 = (k + 1) * dt
    return t90, sim.fallen()


def wave(sim, bal, f, amp=0.35, seconds=3.0):
    sim.reset(); bal.prev = None
    arm = dict(r_shoulder_roll=-2.3, r_elbow=-0.6)
    stand(sim, bal, 1.2, arm)
    dt = sim.m.opt.timestep
    qs = []
    for k in range(int(seconds / dt)):
        t = k * dt
        sim.set_targets(bal.adjust(dict(stand_pose(), **arm, r_shoulder_yaw=amp * math.sin(2 * math.pi * f * t))))
        sim.step()
        if t > seconds - 1.5:
            qs.append(joint_q(sim, "r_shoulder_yaw"))
    return (max(qs) - min(qs)) / 2 / amp, sim.fallen()


def walk_speed(sim, stride, T, steps=4):
    bal = Balance(sim)
    fell, dx, dy, pk = walk.walk(sim, bal, steps, stride, T=T)
    return dx / (steps * T), fell is not None, dx


def main():
    fb.build()
    figure_test.MODEL = fb.OUT
    sim = FD.FullSim()
    getup._SIM = sim          # 起身动作也用这个模型（录制回放需要）
    add_speed_limit(sim)
    bal = Balance(sim)
    res, scenes = {}, {}
    res["rom"] = rom(sim.m)
    print("1) 活动范围")
    for r in res["rom"]:
        print(f"  {r['act']:<8} 需要 {r['need']:>3}°  有 {r['have']:>3}°  {'✓' if r['ok'] else '✕'}")

    print("2) 速度")
    vt, _ = attach(sim)
    t, fell = time_to(sim, bal, "neck_yaw", 0.0, 0.9)
    res["head"] = dict(t90=t, deg=round(math.degrees(0.9 * 0.9)), fell=fell)
    print(f"  转头 52°：{t:.2f} s" if t else "  转头：到不了")
    t, fell = time_to(sim, bal, "r_shoulder_roll", -0.15, -2.3, dict(r_elbow=-0.2))
    res["arm"] = dict(t90=t, fell=fell)
    print(f"  举手过头：{t:.2f} s，摔倒 {fell}" if t else "  举手：到不了")
    waves = []
    for f in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
        ratio, fell = wave(sim, bal, f)
        waves.append(dict(f=f, ratio=round(ratio, 2), fell=fell))
        print(f"  挥手 {f:.0f} Hz：幅度保持 {ratio*100:.0f}%，摔倒 {fell}", flush=True)
    res["wave"] = waves
    ups = []
    for T in (1.0, 0.6, 0.4, 0.3, 0.25, 0.2):
        knee, fell = figure_stress.fast_rise(sim, T)
        ups.append(dict(T=T, knee_Nm=round(knee, 2), fell=fell is not None))
        print(f"  从深蹲站起 {T:.2f} s：膝 {knee:.2f} N·m，摔倒 {fell is not None}", flush=True)
    res["rise"] = ups
    res["vmax_speed_tests"] = {k: round(v, 1) for k, v in vt.vmax.items()}

    print("3) 抗推（0.1 秒的水平推力，推胸口）")
    pushes = []
    for F in (2, 3, 4, 5, 6, 8, 10):
        ok = figure_stress.push(sim, F, bal)
        pushes.append(dict(F=F, ok=ok))
        print(f"  {F} N：{'站住' if ok else '摔倒'}", flush=True)
    res["push"] = pushes

    print("4) 走路（静步行，换不同的步长和节奏）")
    walks = []
    for stride, T in ((0.03, 2.4), (0.03, 1.6), (0.04, 1.6), (0.03, 1.2), (0.04, 1.2), (0.05, 1.2), (0.03, 0.9), (0.04, 0.9), (0.03, 0.7)):
        v, fell, dx = walk_speed(sim, stride, T)
        walks.append(dict(stride=stride, T=T, v_cm_s=round(v * 100, 2), fell=fell, dx_cm=round(dx * 100, 1)))
        print(f"  步长 {stride*100:.0f} cm、每步 {T:.1f} s：{v*100:.2f} cm/s，{'摔倒' if fell else '走完'}", flush=True)
    res["walk"] = walks

    print("5) 摔倒后爬起（getup.py 搜索到的最好动作）")
    gb = json.load(open(os.path.join(HERE, "getup_best.json"), encoding="utf-8"))
    res["getup"] = gb["result"]
    print(f"  站起来了：{gb['result']['ok']}，骨盆高 {gb['result']['z_cm']:.1f} cm，倾斜 {gb['result']['tilt']:.0f}°")

    # ---------- 录制回放 ----------
    print("录制回放……", flush=True)
    def record(name, title, fn):
        vt, rec = attach(sim, True)
        out = fn()
        sim.recorder = None
        scenes[name] = {"title": title, "frames": rec.frames}
        res.setdefault("vmax", {})[name] = vt.worst()
        return out

    fmax = max([w["f"] for w in waves if w["ratio"] >= 0.7 and not w["fell"]] or [1.0])
    record("wave", f"挥手 {fmax:.0f} Hz", lambda: wave(sim, bal, fmax))
    good = [p["F"] for p in pushes if p["ok"]]
    fpush = max(good) if good else 2
    record("push", f"被推 {fpush} N", lambda: figure_stress.push(sim, fpush, bal))
    okw = [w for w in walks if not w["fell"]]
    bw = max(okw, key=lambda w: w["v_cm_s"]) if okw else walks[0]
    record("walk", f"走路 {bw['v_cm_s']:.1f} cm/s", lambda: walk_speed(sim, bw["stride"], bw["T"]))
    record("getup", "趴着爬起", lambda: getup.run(gb["x"]))
    for k, (n, v) in res["vmax"].items():
        print(f"  {k}：最快的关节 {n} {v:.1f} rad/s")
    m = sim.m
    geoms = [{"name": m.geom(i).name, "type": int(m.geom_type[i]), "size": [round(float(x) * 1000, 3) for x in m.geom_size[i]],
              "rgba": [round(float(x), 3) for x in m.geom_rgba[i]]} for i in range(m.ngeom)]
    with open(os.path.join(HERE, "..", "agility_frames.js"), "w", encoding="utf-8") as f:
        f.write("window.AGI = " + json.dumps({"geoms": geoms, "scenes": scenes}, ensure_ascii=False, separators=(",", ":")) + ";\n")
    json.dump(res, open(os.path.join(HERE, "agility_result.json"), "w"), ensure_ascii=False, indent=1,
              default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("已导出 agility_frames.js 和 agility_result.json")


if __name__ == "__main__":
    main()
