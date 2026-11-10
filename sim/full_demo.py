"""
55 cm 人偶 · 整体测试与回放导出

用法：
    python full_build.py
    python full_demo.py          # 跑全部场景，打印结果，导出 ../full_frames.js 供网页回放

场景：
  1. 站立与深蹲（绳驱膝）
  2. 单腿站立（重心反馈平衡）
  3. 走路（静步行，6 步）
  4. 挥手与手势：右臂举起挥手，右手 张开 → 比 V → OK，左手握拳；全程保持平衡
所有身体控制沿用之前的代码（figure_test / tendon_knee / balance / walk / figure_stress），
手部控制沿用 6 电机手的做法（目标线长 + 弹簧前馈 + 重力补偿）。
"""
import json
import math
import os

import mujoco
import numpy as np

import autotrim
import figure_stress
import figure_test
import full_build as fb
import hands_ab_build as hab
import hands_demo as hd
import tendon_knee as tk
import walk
from balance import Balance

HERE = os.path.dirname(os.path.abspath(__file__))
FPS = 30
COUPLE = dict(hd.COUPLE)
COUPLE.update({"index_mcp": [("index_pip", 1.0), ("index_dip", 0.75)],
               "middle_mcp": [("middle_pip", 1.0), ("middle_dip", 0.75)],
               "ring_mcp": [("ring_pip", 1.0), ("ring_dip", 0.75)], "little_mcp": [("little_pip", 1.0), ("little_dip", 0.75)]})
HAND_JOINTS = ["index_abd", "index_mcp", "middle_abd", "middle_mcp", "ring_mcp", "little_mcp", "little_abd",
               "thumb_opp", "thumb_mcp", "wrist_yaw", "wrist_pitch"]
SIGN = {"L": 1, "R": -1}


def hand_pose(k, **kw):
    q = {j: 0.0 for j in HAND_JOINTS}
    for j, v in kw.items():
        q[j] = v
    return q


def relaxed(k):
    s = SIGN[k]
    return hand_pose(k, index_mcp=0.25, middle_mcp=0.3, ring_mcp=0.35, little_mcp=0.4, thumb_opp=s * 0.4, thumb_mcp=0.3)


def fist(k):
    s = SIGN[k]
    return hand_pose(k, index_mcp=1.45, middle_mcp=1.45, ring_mcp=1.45, little_mcp=1.45, thumb_opp=s * 1.0, thumb_mcp=0.9)


def vsign(k):
    s = SIGN[k]
    return hand_pose(k, index_abd=-s * 0.25, middle_abd=s * 0.2, ring_mcp=1.45, little_mcp=1.45, thumb_opp=s * 1.0, thumb_mcp=0.9)


class FullSim(tk.TendonSim):
    def __init__(self, view=False):
        self.hand_q = {k: relaxed(k) for k in "LR"}
        self.recorder = None
        super().__init__(view)
        m = self.m
        self.hand_dofs = [m.jnt_dofadr[j] for j in range(m.njnt) if m.joint(j).name[:2] in ("L_", "R_")]
        self.hand_act = {k: [i for n, i in self.act.items() if n.startswith(k + "_")] for k in "LR"}
        # 第二版身体：自动配平（L1）。只在用正式的整体模型时装上；测试用的临时模型自己决定装不装
        self.trim = None
        if os.path.abspath(figure_test.MODEL) == os.path.abspath(fb.OUT) and os.path.exists(fb.REF):
            self.trim = autotrim.Trim(self, fb.REF)

    # ---------- 手部控制 ----------
    def full_q(self, q):
        out = dict(q)
        for j in list(q):
            for c, f in COUPLE.get(j, []):
                out[c] = f * q[j]
        return out

    def k_eff(self, k, j):
        kk = float(self.m.jnt_stiffness[self.m.joint(f"{k}_{j}").id])
        for c, f in COUPLE.get(j, []):
            kk += f * f * float(self.m.jnt_stiffness[self.m.joint(f"{k}_{c}").id])
        return kk

    def hand_ctrl(self, k, q):
        q = self.full_q(q)
        for tn, lst in hab.TENDONS.items():
            length = sum(c * q.get(j, 0.0) for j, c in lst)
            force = np.mean([self.k_eff(k, j) * q.get(j, 0.0) / c for j, c in lst])
            self.d.ctrl[self.act[f"{k}_{tn}"]] = length + force / 6000.0

    def set_targets(self, pose):
        pose = dict(pose)
        hands = {k: pose.pop(f"hand_{k}", None) for k in "LR"}
        super().set_targets(pose)
        if not hasattr(self, "hand_act"):
            return
        for k in "LR":
            if hands[k] is not None:
                self.hand_q[k] = hands[k]
            self.hand_ctrl(k, self.hand_q[k])

    def reset(self):
        self.hand_q = {k: relaxed(k) for k in "LR"}
        super().reset()
        if hasattr(self, "hand_dofs"):
            for k in "LR":  # 手指直接放到放松姿势
                for j, v in self.full_q(self.hand_q[k]).items():
                    self.d.qpos[self.m.jnt_qposadr[self.m.joint(f"{k}_{j}").id]] = v
            mujoco.mj_forward(self.m, self.d)

    def step(self, n=1):
        for _ in range(n):
            self._tendon_ctrl()
            self.d.qfrc_applied[:] = 0
            self.d.qfrc_applied[self.hand_dofs] = self.d.qfrc_bias[self.hand_dofs]  # 手部电机带重力补偿
            mujoco.mj_step(self.m, self.d)
            if self.recorder:
                self.recorder()
        if self.view is not None:
            self.view.sync()

    # ---------- OK 手势：在当前姿态下，让拇指尖去找食指尖 ----------
    def ok_pose(self, k):
        m = self.m
        best = None
        sid_t, sid_i = m.site(f"{k}_thumb_tip").id, m.site(f"{k}_index_tip").id
        jac = np.zeros((3, m.nv))
        for c in np.linspace(0.4, 1.3, 7):
            q = relaxed(k)
            q.update(index_mcp=c, middle_mcp=0.2, ring_mcp=0.3, little_mcp=0.4)
            d = mujoco.MjData(m)
            d.qpos[:] = self.d.qpos
            for it in range(200):
                for j, v in self.full_q(q).items():
                    d.qpos[m.jnt_qposadr[m.joint(f"{k}_{j}").id]] = v
                mujoco.mj_kinematics(m, d)
                mujoco.mj_comPos(m, d)
                err = d.site_xpos[sid_i] - d.site_xpos[sid_t]
                if np.linalg.norm(err) < 1e-5:
                    break
                mujoco.mj_jacSite(m, d, jac, None, sid_t)
                js = ["thumb_opp", "thumb_mcp"]
                J = np.zeros((3, 2))
                for n, j in enumerate(js):
                    J[:, n] = jac[:, m.jnt_dofadr[m.joint(f"{k}_{j}").id]]
                    for cc, f in COUPLE.get(j, []):
                        J[:, n] += f * jac[:, m.jnt_dofadr[m.joint(f"{k}_{cc}").id]]
                dq = J.T @ np.linalg.solve(J @ J.T + 1e-9 * np.eye(3), err)
                for n, j in enumerate(js):
                    lo, hi = m.jnt_range[m.joint(f"{k}_{j}").id]
                    q[j] = float(np.clip(q[j] + 0.5 * dq[n], lo, hi))
            e = float(np.linalg.norm(err))
            if best is None or e < best[0]:
                best = (e, dict(q))
        return best[1], best[0]


# ---------- 录制 ----------
class Recorder:
    def __init__(self, sim):
        self.sim, self.frames, self.k, self.label = sim, [], 0, ""
        self.every = int(1 / FPS / sim.m.opt.timestep)

    def __call__(self):
        self.k += 1
        if self.k % self.every:
            return
        m, d = self.sim.m, self.sim.d
        g = []
        for i in range(m.ngeom):
            q = np.zeros(4)
            mujoco.mju_mat2Quat(q, d.geom_xmat[i])
            g.append([round(float(x) * 1000, 2) for x in d.geom_xpos[i]] + [round(float(x), 3) for x in q])
        tq = self.sim.torques()
        hand = {k: round(max(abs(float(d.actuator_force[i])) for i in self.sim.hand_act[k]), 2) for k in "LR"}
        self.frames.append({"t": round(len(self.frames) / FPS, 3), "g": g, "label": self.label,
                            "v": {"quad_l": round(abs(tq["l_quad"]), 1), "quad_r": round(abs(tq["r_quad"]), 1),
                                  "hip_l": round(abs(tq["l_hip_pitch"]), 2), "hip_r": round(abs(tq["r_hip_pitch"]), 2),
                                  "ankle_l": round(abs(tq["l_ankle_pitch"]), 2), "ankle_r": round(abs(tq["r_ankle_pitch"]), 2),
                                  "hand_l": hand["L"], "hand_r": hand["R"], "tilt": round(self.sim.tilt_deg(), 1)}})


def scene_squat(sim, rec):
    sim.reset(); rec.label = "站立"
    pk1, _, f1 = figure_test.run_phase(sim, 1.0, lambda t: figure_test.stand_pose())
    rec.label = "深蹲"
    pk2, _, f2 = figure_test.run_phase(sim, 9.0, lambda t: figure_test.stand_pose(figure_test.squat_knee(t)))
    return {"fell": f1 is not None or f2 is not None, "quad_peak_N": max(pk2["l_quad"], pk2["r_quad"]),
            "knee_peak_Nm": max(pk2["l_knee"], pk2["r_knee"])}


def scene_single_leg(sim, rec):
    rec.label = "单腿站立"
    bal = Balance(sim)
    fell, pk = figure_stress.single_leg(sim, bal, 0.7)
    return {"fell": fell is not None, "hip_roll_peak_Nm": pk["l_hip_roll"], "ankle_roll_peak_Nm": pk["l_ankle_roll"]}


def scene_walk(sim, rec):
    rec.label = "走路"
    bal = Balance(sim)
    fell, dx, dy, pk = walk.walk(sim, bal, 6, 0.03, T=2.4)
    return {"fell": fell is not None, "dx_cm": dx * 100, "dy_cm": dy * 100,
            "quad_peak_N": max(pk["l_quad"], pk["r_quad"]), "hip_peak_Nm": max(pk["l_hip_pitch"], pk["r_hip_pitch"])}


def scene_gesture(sim, rec):
    sim.reset()
    bal = Balance(sim)
    okq, okerr = None, None
    stages = [  # (开始时间, 标签, 右臂, 右手, 左手)
        (0.0, "站立", (0.0, -0.15, -0.2), "relaxed", "relaxed"),
        (0.8, "举起右手", (-2.3, 0.0, -0.6), "relaxed", "fist"),
        (1.8, "挥手（张开）", (-2.3, 0.0, -0.6), "open", "fist"),
        (3.6, "比 V", (-2.3, 0.0, -0.6), "v", "fist"),
        (5.2, "OK", (-2.3, 0.0, -0.6), "ok", "fist"),
        (7.0, "放下", (0.0, -0.15, -0.2), "relaxed", "relaxed")]
    T_END = 8.2
    state = {"ok": None}

    def hand_of(name, k):
        if name == "relaxed":
            return relaxed(k)
        if name == "fist":
            return fist(k)
        if name == "open":
            return hand_pose(k)
        if name == "v":
            return vsign(k)
        if state["ok"] is None:
            state["ok"], state["err"] = sim.ok_pose(k)
        return state["ok"]

    def pose(t):
        i = max(n for n, s in enumerate(stages) if s[0] <= t)
        t0, label, arm, rh, lh = stages[i]
        prev = stages[max(0, i - 1)][2]
        a = min(1.0, (t - t0) / 0.6) if i > 0 else 1.0
        a = 0.5 - 0.5 * math.cos(math.pi * a)
        roll, yaw, elbow = [p + (c - p) * a for p, c in zip(prev, arm)]
        if "挥手" in label or label in ("比 V", "OK"):
            yaw += 0.35 * math.sin(2 * math.pi * 1.6 * (t - 1.8)) if label == "挥手（张开）" else 0.0
        rec.label = label
        p = figure_test.stand_pose()
        p.update(r_shoulder_roll=roll, r_shoulder_yaw=yaw, r_elbow=elbow)
        p["hand_R"] = hand_of(rh, "R")
        p["hand_L"] = hand_of(lh, "L")
        return bal.adjust(p)

    pk, _, fell = figure_test.run_phase(sim, T_END, pose)
    return {"fell": fell is not None, "ok_ik_err_mm": (state.get("err") or 0) * 1000,
            "hand_peak_N": max(abs(float(sim.d.actuator_force[i])) for i in sim.hand_act["R"])}


def main():
    fb.build()
    figure_test.MODEL = fb.OUT
    sim = FullSim()
    total = float(sum(sim.m.body_mass))
    print(f"整体模型：{os.path.basename(fb.OUT)}，总质量 {total:.2f} kg\n")
    scenes, results = {}, {}
    for name, title, fn in (("squat", "站立与深蹲", scene_squat), ("single", "单腿站立", scene_single_leg),
                            ("walk", "走路", scene_walk), ("gesture", "挥手与手势", scene_gesture)):
        rec = Recorder(sim)
        sim.recorder = rec
        r = fn(sim, rec)
        sim.recorder = None
        scenes[name] = {"title": title, "frames": rec.frames}
        results[name] = r
        print(f"【{title}】" + "，".join(f"{k} = {v:.2f}" if isinstance(v, float) else f"{k} = {v}" for k, v in r.items()))
    m = sim.m
    geoms = [{"name": m.geom(i).name, "type": int(m.geom_type[i]), "size": [round(float(x) * 1000, 3) for x in m.geom_size[i]],
              "rgba": [round(float(x), 3) for x in m.geom_rgba[i]]} for i in range(m.ngeom)]
    out = {"geoms": geoms, "scenes": scenes, "results": results, "mass_kg": round(total, 2)}
    path = os.path.join(HERE, "..", "full_frames.js")
    with open(path, "w", encoding="utf-8") as f:
        f.write("window.FULL = " + json.dumps(out, ensure_ascii=False, separators=(",", ":")) + ";\n")
    print(f"\n已导出 {os.path.relpath(path, HERE)}（{os.path.getsize(path)/1e6:.1f} MB，{sum(len(s['frames']) for s in scenes.values())} 帧）")


if __name__ == "__main__":
    main()
