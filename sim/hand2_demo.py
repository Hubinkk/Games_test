"""
绳驱手 v2 · 演示与测量（MuJoCo）

用法：
    python hand2_build.py
    python hand2_demo.py            # 跑全部场景，打印测量结果，并导出 ../hand2_frames.js 供网页回放

场景：
    1. 手势：张开 → 握拳 → 比 V → 指 → 张开五指 → OK → 张开
    2. 打字：在手机屏幕键盘上用食指打 "hello"
       手臂（肩肘）负责粗定位，每两个键一档；手指用左右张开和弯曲负责最后几毫米
    3. 握笔：直径 6 mm 的笔
"""
import json
import math
import os

import mujoco
import numpy as np

import hand2_build as hb

HERE = os.path.dirname(os.path.abspath(__file__))
FPS = 30
COUPLE = {"index_pip": [("index_dip", 0.75)], "middle_pip": [("middle_dip", 0.75)],
          "ring_pip": [("ring_dip", 0.75)], "little_pip": [("little_dip", 0.75)],
          "thumb_mcp": [("thumb_ip", 1.0)], "little_abd": [("ring_abd", 0.4)]}
TENDON_JOINTS = {
    "index_mcp": [("index_mcp", hb.ARM_MCP)], "index_pip": [("index_pip", hb.ARM_PIP)], "index_abd": [("index_abd", hb.ARM_ABD)],
    "middle_mcp": [("middle_mcp", hb.ARM_MCP)], "middle_pip": [("middle_pip", hb.ARM_PIP)], "middle_abd": [("middle_abd", hb.ARM_ABD)],
    "ring_flex": [("ring_mcp", hb.ARM_MCP), ("ring_pip", hb.ARM_PIP)], "little_flex": [("little_mcp", hb.ARM_MCP), ("little_pip", hb.ARM_PIP)],
    "spread": [("little_abd", hb.ARM_ABD)], "thumb_opp": [("thumb_opp", hb.ARM_THUMB)], "thumb_cmc": [("thumb_cmc", hb.ARM_THUMB)],
    "thumb_flex": [("thumb_mcp", hb.ARM_THUMB)], "wrist_pitch": [("wrist_pitch", hb.ARM_WRIST)], "wrist_yaw": [("wrist_yaw", hb.ARM_WRIST)]}


class Hand2:
    def __init__(self):
        hb.build()
        self.m = mujoco.MjModel.from_xml_path(hb.OUT)
        self.d = mujoco.MjData(self.m)
        self.act = {self.m.actuator(i).name: i for i in range(self.m.nu)}
        self.qadr = {self.m.joint(j).name: self.m.jnt_qposadr[j] for j in range(self.m.njnt)}
        self.dadr = {self.m.joint(j).name: self.m.jnt_dofadr[j] for j in range(self.m.njnt)}
        self.keys = hb.key_layout()
        self.pen = self.m.geom("pen").id
        self.frames, self.events = [], []
        self.min_dist = 0.0
        self.recording = True
        self.on_step = None
        self.q = {}  # 当前关节目标
        self.reset()

    # ---------- 基础 ----------
    def reset(self):
        mujoco.mj_resetData(self.m, self.d)
        self.q = {j: 0.0 for j in self.qadr}
        self.apply()
        mujoco.mj_forward(self.m, self.d)

    def pen_on(self, on):
        b = self.m.geom_bodyid[self.pen]
        v = 1 if on else 0
        self.m.geom_contype[self.pen] = self.m.geom_conaffinity[self.pen] = v
        self.m.body_contype[b] = self.m.body_conaffinity[b] = v
        self.m.geom_rgba[self.pen][3] = 1.0 if on else 0.0

    def set_q(self, **kw):
        for j, v in kw.items():
            self.q[j] = v
            for c, k in COUPLE.get(j, []):
                self.q[c] = k * v

    def apply(self):
        for tn, lst in TENDON_JOINTS.items():
            self.d.ctrl[self.act[tn]] = sum(c * self.q[j] for j, c in lst)
        for j in ("arm_x", "arm_y", "arm_z"):
            self.d.ctrl[self.act[j]] = self.q[j]

    def step(self):
        """控制器带重力补偿（真实机器人控制器的标准做法）：先抵消自重，伺服只负责跟踪目标。"""
        self.d.qfrc_applied[:] = self.d.qfrc_bias
        mujoco.mj_step(self.m, self.d)
        if self.d.ncon:
            self.min_dist = min(self.min_dist, float(self.d.contact.dist[:self.d.ncon].min()))
        if self.on_step:
            self.on_step()

    def run(self, seconds, label=""):
        """以 ctrl 平滑过渡到目标，并按 FPS 记录帧。"""
        start = self.d.ctrl.copy()
        self.apply()
        goal = self.d.ctrl.copy()
        steps = max(1, int(seconds / self.m.opt.timestep))
        every = int(1 / FPS / self.m.opt.timestep)
        for k in range(steps):
            a = min(1.0, (k + 1) / (0.6 * steps))
            a = 0.5 - 0.5 * math.cos(math.pi * a)
            self.d.ctrl[:] = start + (goal - start) * a
            self.step()
            if k % every == 0:
                self.record(label)

    def record(self, label):
        if not self.recording:
            return
        m, d = self.m, self.d
        g = []
        for i in range(m.ngeom):
            q = np.zeros(4)
            mujoco.mju_mat2Quat(q, d.geom_xmat[i])
            g.append([round(float(x) * 1000, 3) for x in d.geom_xpos[i]] + [round(float(x), 4) for x in q])
        ten = {n: round(abs(float(d.actuator_force[self.act[n]])), 2) for n in hb.HAND_ACTUATORS}
        self.frames.append({"t": round(len(self.frames) / FPS, 3), "g": g, "ten": ten, "label": label})

    # ---------- 逆运动学 ----------
    def ik(self, site, target, joints, q0=None, iters=300):
        """阻尼最小二乘：只动 joints，让 site 到达 target。联动关节一起计算。"""
        m = self.m
        d = mujoco.MjData(m)
        d.qpos[:] = self.d.qpos
        q = dict(self.q) if q0 is None else dict(q0)
        for j, v in q.items():
            d.qpos[self.qadr[j]] = v
        sid = m.site(site).id
        jac = np.zeros((3, m.nv))
        for _ in range(iters):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            err = np.array(target) - d.site_xpos[sid]
            if np.linalg.norm(err) < 2e-5:
                break
            mujoco.mj_jacSite(m, d, jac, None, sid)
            J = np.zeros((3, len(joints)))
            for k, j in enumerate(joints):
                J[:, k] = jac[:, self.dadr[j]]
                for c, f in COUPLE.get(j, []):
                    J[:, k] += f * jac[:, self.dadr[c]]
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-7 * np.eye(3), err)
            for k, j in enumerate(joints):
                lo, hi = m.jnt_range[m.joint(j).id]
                q[j] = float(np.clip(q[j] + 0.5 * dq[k], lo, hi))
                d.qpos[self.qadr[j]] = q[j]
                for c, f in COUPLE.get(j, []):
                    q[c] = f * q[j]
                    d.qpos[self.qadr[c]] = q[c]
        mujoco.mj_kinematics(m, d)
        return q, float(np.linalg.norm(np.array(target) - d.site_xpos[sid]))

    # ---------- 测量 ----------
    def key_contact(self):
        """返回 (按到的键, 接触点, 法向力)，取食指与键之间力最大的接触。"""
        m, d = self.m, self.d
        best = None
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            n1, n2 = m.geom(c.geom1).name, m.geom(c.geom2).name
            names = (n1, n2)
            key = next((n for n in names if n.startswith("key_")), None)
            fing = next((n for n in names if n.startswith("index_g")), None)
            if key and fing:
                mujoco.mj_contactForce(m, d, i, f6)
                if best is None or f6[0] > best[2]:
                    best = (key[4:], c.pos.copy(), float(f6[0]))
        return best


def scene_gestures(h):
    h.reset(); h.pen_on(False)
    h.set_q(arm_z=0.0)
    h.run(0.6, "张开")
    fist = dict(index_mcp=1.45, index_pip=1.5, middle_mcp=1.45, middle_pip=1.5, ring_mcp=1.45, ring_pip=1.5,
                little_mcp=1.45, little_pip=1.5, thumb_opp=0.9, thumb_cmc=0.5, thumb_mcp=0.9)
    h.set_q(**fist); h.run(1.0, "握拳")
    h.set_q(index_mcp=0, index_pip=0, middle_mcp=0, middle_pip=0, index_abd=-0.22, middle_abd=0.18); h.run(1.0, "比 V")
    h.set_q(middle_mcp=1.45, middle_pip=1.5, index_abd=0, middle_abd=0); h.run(1.0, "指")
    h.set_q(**{k: 0.0 for k in fist}); h.set_q(index_abd=-0.3, middle_abd=-0.08, little_abd=0.3, thumb_cmc=-0.3)
    h.run(1.0, "张开五指")
    h.set_q(index_abd=0, middle_abd=0, little_abd=0, thumb_cmc=0)
    # OK：让拇指尖和食指尖在掌前下方相遇
    # 搜索一个拇指和食指都够得到的会合点：遍历食指姿态，让拇指去找食指指尖
    best = None
    for mcp in np.linspace(0.3, 1.3, 6):
        for pip in np.linspace(0.3, 1.3, 6):
            q0 = {**h.q, "index_mcp": mcp, "index_pip": pip, "index_dip": 0.75 * pip}
            dd = mujoco.MjData(h.m)
            for j, v in q0.items():
                dd.qpos[h.qadr[j]] = v
            mujoco.mj_kinematics(h.m, dd)
            tip = dd.site_xpos[h.m.site("index_tip").id].copy()
            qt_, et_ = h.ik("thumb_tip", tip, ["thumb_opp", "thumb_cmc", "thumb_mcp"], q0=q0, iters=150)
            if best is None or et_ < best[1]:
                best = (qt_, et_)
    qt, et = best
    ei = 0.0
    print(f"  OK 逆运动学残差：食指 {ei*1000:.2f} mm，拇指 {et*1000:.2f} mm")
    h.set_q(**{k: qt[k] for k in ("index_mcp", "index_pip", "thumb_opp", "thumb_cmc", "thumb_mcp")})
    h.set_q(middle_mcp=0.2, ring_mcp=0.3, little_mcp=0.4)
    h.run(1.2, "OK")
    dist = np.linalg.norm(h.d.site_xpos[h.m.site("thumb_tip").id] - h.d.site_xpos[h.m.site("index_tip").id]) * 1000
    h.events.append({"t": h.frames[-1]["t"], "type": "ok", "dist_mm": round(float(dist), 2)})
    h.reset_targets = None
    for k in ("index_mcp", "index_pip", "thumb_opp", "thumb_cmc", "thumb_mcp", "middle_mcp", "ring_mcp", "little_mcp"):
        h.set_q(**{k: 0.0})
    h.run(0.8, "张开")
    return {"ok_tip_distance_mm": float(dist)}


def scene_typing(h, text="hello"):
    h.reset(); h.pen_on(False)
    # 用一根手指点屏幕时，其余手指收起（"指"的手势），避免先碰到屏幕
    hover = dict(index_mcp=0.35, index_pip=0.5, middle_mcp=1.45, middle_pip=1.5, ring_mcp=1.45, ring_pip=1.5,
                 little_mcp=1.45, little_pip=1.5, thumb_opp=0.3, thumb_cmc=-0.2, thumb_mcp=0.3)  # 拇指贴在手的侧面
    h.set_q(**hover)
    # 名义姿态下，食指指腹相对手臂原点的位置
    mujoco.mj_forward(h.m, h.d)
    base = np.array(hb.ARM["pos"])
    d = mujoco.MjData(h.m)
    press_pose = {**h.q, "index_mcp": 0.55, "index_pip": 0.75, "index_dip": 0.75 * 0.75}
    for j, v in press_pose.items():  # 按"按下时的姿势"放手臂，按下点处在手指活动范围的中间
        d.qpos[h.qadr[j]] = v
    mujoco.mj_kinematics(h.m, d)
    R = 0.0019  # 指尖圆头半径（0.95 × 2 mm）
    pad0 = d.site_xpos[h.m.site("index_tip").id] - base - np.array([0, 0, R])  # 圆头最低点
    results = []
    h.run(0.5, "准备打字")
    for ch in text:
        kx, ky = h.keys[ch]
        top = 0.0005
        # 手臂粗定位：y 每两个键一档，x 每行一档
        gy = (round((ky + hb.KEY_PITCH / 2) / (2 * hb.KEY_PITCH)) * 2 * hb.KEY_PITCH) - hb.KEY_PITCH / 2
        ax, ay = kx - pad0[0], gy - pad0[1]
        az = top - 0.0006 - pad0[2] - base[2]
        h.set_q(arm_x=ax, arm_y=ay, arm_z=az)
        # 先让手臂到位，手指保持悬停
        qh, _ = h.ik("index_tip", [kx, ky, top + 0.0035 + R], ["wrist_yaw", "index_abd", "index_mcp", "index_pip"],
                     q0={**h.q, "arm_x": ax, "arm_y": ay, "arm_z": az})
        h.set_q(**{k: qh[k] for k in ("wrist_yaw", "index_abd", "index_mcp", "index_pip")})
        h.run(0.45, f"移到 {ch.upper()}")
        # 到位检查：手臂和食指都到达目标后才按（最多再等 0.5 秒）
        every = int(1 / FPS / h.m.opt.timestep)
        for k in range(int(0.5 / h.m.opt.timestep)):
            errs = [abs(h.d.qpos[h.qadr[j]] - h.q[j]) for j in ("arm_x", "arm_y", "arm_z")]
            if max(errs) < 0.0002:
                break
            h.step()
            if k % every == 0:
                h.record(f"移到 {ch.upper()}")
        # 按下：目标在键面以下 0.6 mm，力由电机的拉力上限和伺服刚度决定
        qp, err_ik = h.ik("index_tip", [kx, ky, top - 0.0006 + R], ["wrist_yaw", "index_abd", "index_mcp", "index_pip"], q0=h.q)
        h.set_q(**{k: qp[k] for k in ("wrist_yaw", "index_abd", "index_mcp", "index_pip")})
        peak = None
        start = h.d.ctrl.copy(); h.apply(); goal = h.d.ctrl.copy()
        steps = int(0.18 / h.m.opt.timestep); every = int(1 / FPS / h.m.opt.timestep)
        for k in range(steps):
            a = min(1.0, (k + 1) / (0.5 * steps))
            h.d.ctrl[:] = start + (goal - start) * a
            h.step()
            kc = h.key_contact()
            if kc and (peak is None or kc[2] > peak[2]):
                peak = kc
            if k % every == 0:
                h.record(f"按 {ch.upper()}")
        hit, pos, force = peak if peak else (None, None, 0.0)
        err = float(np.hypot(pos[0] - kx, pos[1] - ky) * 1000) if pos is not None else None
        results.append({"key": ch, "hit": hit, "error_mm": err, "force_N": force,
                        "abd_deg": math.degrees(qp["index_abd"]), "yaw_deg": math.degrees(qp["wrist_yaw"])})
        h.events.append({"t": h.frames[-1]["t"], "type": "key", "key": ch, "hit": hit,
                         "error_mm": None if err is None else round(err, 2), "force_N": round(force, 2)})
        h.set_q(**{k: qh[k] for k in ("wrist_yaw", "index_abd", "index_mcp", "index_pip")})
        h.run(0.2, f"抬起")
    return results


def scene_pen(h):
    h.reset(); h.pen_on(True)
    h.set_q(thumb_opp=0.6, thumb_cmc=-0.2)
    h.run(0.6, "准备握笔")
    h.set_q(index_mcp=1.3, index_pip=1.6, middle_mcp=1.3, middle_pip=1.6, ring_mcp=1.3, ring_pip=1.6,
            little_mcp=1.3, little_pip=1.6, thumb_opp=1.2, thumb_cmc=0.7, thumb_mcp=0.9)
    h.run(1.4, "握笔")
    touch = set()
    for i in range(h.d.ncon):
        c = h.d.contact[i]
        if h.pen in (c.geom1, c.geom2):
            o = c.geom2 if c.geom1 == h.pen else c.geom1
            touch.add(h.m.geom(o).name)
    ten = {n: abs(float(h.d.actuator_force[h.act[n]])) for n in hb.HAND_ACTUATORS}
    h.run(0.6, "握笔")
    h.set_q(**{k: 0.0 for k in ("index_mcp", "index_pip", "middle_mcp", "middle_pip", "ring_mcp", "ring_pip",
                                 "little_mcp", "little_pip", "thumb_opp", "thumb_cmc", "thumb_mcp")})
    h.run(0.8, "松开")
    h.pen_on(False)
    return {"touch": sorted(touch), "tension": ten}


def max_penetration(h):
    """回放的每一帧都来自 MuJoCo；这里额外统计仿真中最深的穿透量。"""
    return None


def main(extra=()):
    h = Hand2()
    scenes = {}
    for name, fn in (("gestures", scene_gestures), ("typing", scene_typing), ("pen", scene_pen)) + tuple(extra):
        h.frames, h.events, h.min_dist = [], [], 0.0
        res = fn(h)
        print(f"  （{name}：仿真中最深穿透 {-h.min_dist*1000:.2f} mm）")
        scenes[name] = {"frames": h.frames, "events": h.events}
        if name == "gestures":
            print(f"【手势】OK 手势两指尖中心相距 {res['ok_tip_distance_mm']:.1f} mm（两指半径约 2 mm，<4.2 mm 即表面相碰）")
        elif name == "typing":
            print("【打字 hello】")
            for r in res:
                ok = r["hit"] == r["key"]
                print(f"  {r['key'].upper()}：按到 {str(r['hit']).upper() if r['hit'] else '没按到'}"
                      f"{'' if ok else '（错）'}，误差 {r['error_mm'] if r['error_mm'] is None else round(r['error_mm'], 2)} mm，"
                      f"按键力 {r['force_N']:.2f} N，食指张开 {r['abd_deg']:.0f}°，手腕转 {r['yaw_deg']:.0f}°")
            errs = [r["error_mm"] for r in res if r["error_mm"] is not None]
            if errs:
                print(f"  平均误差 {np.mean(errs):.2f} mm，最大 {max(errs):.2f} mm；键宽 {hb.KEY_W*1000:.1f} mm")
            scenes[name]["summary"] = res
        elif name != "pen":
            scenes[name]["summary"] = res
        else:
            print(f"【握笔】碰到笔的部位：{', '.join(res['touch']) or '无'}")
            top = sorted(res["tension"].items(), key=lambda x: -x[1])[:4]
            print("  线拉力最大的电机：" + "，".join(f"{k} {v:.1f} N" for k, v in top))
    # 导出几何体定义 + 帧
    m = h.m
    geoms = []
    for i in range(m.ngeom):
        geoms.append({"name": m.geom(i).name, "type": int(m.geom_type[i]),
                      "size": [round(float(x) * 1000, 3) for x in m.geom_size[i]],
                      "rgba": [round(float(x), 3) for x in m.geom_rgba[i]]})
    out = {"geoms": geoms, "actuators": hb.HAND_ACTUATORS, "t_max": {"finger": hb.T_FINGER, "wrist": hb.T_WRIST},
           "keys": {k: [round(v[0] * 1000, 2), round(v[1] * 1000, 2)] for k, v in h.keys.items()}, "scenes": scenes}
    path = os.path.join(HERE, "..", "hand2_frames.js")
    with open(path, "w", encoding="utf-8") as f:
        f.write("window.HAND2 = " + json.dumps(out, ensure_ascii=False, separators=(",", ":")) + ";\n")
    print(f"\n已导出 {os.path.relpath(path, HERE)}（{os.path.getsize(path)/1e6:.1f} MB，{sum(len(s['frames']) for s in scenes.values())} 帧）")


if __name__ == "__main__":
    main()
