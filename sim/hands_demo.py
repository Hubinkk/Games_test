"""
55 cm 人偶 · 双手十指打字（MuJoCo）

用法：
    python hands_build.py
    python hands_demo.py          # 逐档测速度，并把不出错的最快一档导出到 ../hands_frames.js

做法：
  - 标准指法：左手小指 Q/A/Z，无名指 W/S/X，中指 E/D/C，食指 R/F/V/T/G/B；
    右手食指 Y/H/N/U/J/M，中指 I/K，无名指 O/L，小指 P；右手拇指按空格
  - 每个键先算"按下时"的完整姿势（手臂、手腕、手指一起，逆运动学），
    悬停姿势 = 同样的手臂和手腕 + 手指回到打字姿势，所以落点准确
  - 两只手并行：一只手按键时，另一只手已经移向它的下一个键；
    同一只手按完立刻移向下一个键。相邻两次按下之间至少间隔 t_gap，保证输入顺序
  - 实际"打出来的字"按指尖接触键面的先后顺序记录（触摸屏在接触瞬间输入）
"""
import json
import math
import os

import mujoco
import numpy as np

import hands_build as hsb

HERE = os.path.dirname(os.path.abspath(__file__))
FPS = 30
TEXT = "hello how are you"
TOP = 0.0005          # 键面高度
LIFT = 0.003          # 打字姿势下指尖离键面的高度
PRESS = 0.0006        # 按下时目标在键面以下的深度
CURL = 0.5            # 打字姿势的手指弯曲

FINGER_OF = {}
for hand, fingers in (("L", {"little": "qaz", "ring": "wsx", "middle": "edc", "index": "rfvtgb"}),
                      ("R", {"index": "yhnujm", "middle": "ik", "ring": "ol", "little": "p"})):
    for f, chars in fingers.items():
        for ch in chars:
            FINGER_OF[ch] = (hand, f)
FINGER_OF[" "] = ("R", "thumb")
FINGER_NAMES = {"index": "食指", "middle": "中指", "ring": "无名指", "little": "小指", "thumb": "拇指"}
JOINTS_OF = {"index": ["index_abd", "index_mcp", "index_pip"], "middle": ["middle_abd", "middle_mcp", "middle_pip"],
             "ring": ["ring_mcp"], "little": ["little_mcp"], "thumb": ["thumb_opp", "thumb_cmc", "thumb_mcp"]}
TIP_R = {"index": 0.0019, "middle": 0.0019, "ring": 0.0019, "little": 0.00171, "thumb": 0.0020}
COUPLE = {"index_pip": [("index_dip", 0.75)], "middle_pip": [("middle_dip", 0.75)],
          "ring_mcp": [("ring_pip", 1.0), ("ring_dip", 0.75)], "little_mcp": [("little_pip", 1.0), ("little_dip", 0.75)],
          "thumb_mcp": [("thumb_ip", 1.0)], "little_abd": [("ring_abd", 0.4)]}
W_ARM, W_YAW = 0.015, 0.5   # 逆运动学权重：让手臂、手腕、手指分担移动


class Hands:
    def __init__(self):
        hsb.build()
        self.m = mujoco.MjModel.from_xml_path(hsb.OUT)
        self.d = mujoco.MjData(self.m)
        self.act = {self.m.actuator(i).name: i for i in range(self.m.nu)}
        self.qadr = {self.m.joint(j).name: self.m.jnt_qposadr[j] for j in range(self.m.njnt)}
        self.dadr = {self.m.joint(j).name: self.m.jnt_dofadr[j] for j in range(self.m.njnt)}
        self.keys = hsb.key_layout()
        self.ids = {h: [i for n, i in self.act.items() if n.startswith(h + "_")] for h in hsb.SIDES}
        self.key_geom = {self.m.geom(i).name[4:]: i for i in range(self.m.ngeom) if self.m.geom(i).name.startswith("key_")}
        self.recording = True

    # ---------- 姿势 ----------
    def typing_posture(self, hand):
        s = hsb.SIDES[hand]
        q = {j: 0.0 for j in ("arm_x", "arm_y", "arm_z", "wrist_pitch", "wrist_yaw", "spread")}
        q.update({j: 0.0 for j in ("index_abd", "middle_abd", "ring_abd", "little_abd")})
        for f in ("index", "middle"):
            self.set(q, f + "_mcp", CURL); self.set(q, f + "_pip", CURL)
        for f in ("ring", "little"):
            self.set(q, f + "_mcp", CURL)
        # 拇指收在自己手掌下面：键盘只有 6.5 cm 宽，拇指伸在外面时，两只手的拇指会互相碰撞
        # 姿势由搜索得到：不碰手掌和手指（间隙 ≥ 0.3 mm），整根拇指的最低点只比手掌中心低 4.6 mm
        q["thumb_opp"], q["thumb_cmc"] = s * 1.6, 0.0
        self.set(q, "thumb_mcp", 0.6)
        return q

    @staticmethod
    def set(q, j, v):
        q[j] = v
        for c, k in COUPLE.get(j, []):
            q[c] = k * v

    def kin(self, qs):
        """qs: {hand: q}。返回已做好运动学计算的 MjData。"""
        d = mujoco.MjData(self.m)
        for h, q in qs.items():
            for j, v in q.items():
                if f"{h}_{j}" in self.qadr:
                    d.qpos[self.qadr[f"{h}_{j}"]] = v
        mujoco.mj_kinematics(self.m, d)
        mujoco.mj_comPos(self.m, d)
        return d

    def lowest(self, d, hand, finger):
        return d.site_xpos[self.m.site(f"{hand}_{finger}_tip").id] - np.array([0, 0, TIP_R[finger]])

    def ik(self, hand, finger, target, q0, joints, weights, iters=400):
        q = dict(q0)
        sid = self.m.site(f"{hand}_{finger}_tip").id
        jac = np.zeros((3, self.m.nv))
        target = np.array(target) + np.array([0, 0, TIP_R[finger]])
        for _ in range(iters):
            d = self.kin({hand: q})
            err = target - d.site_xpos[sid]
            if np.linalg.norm(err) < 1e-5:
                break
            mujoco.mj_jacSite(self.m, d, jac, None, sid)
            J = np.zeros((3, len(joints)))
            for k, j in enumerate(joints):
                J[:, k] = jac[:, self.dadr[f"{hand}_{j}"]]
                for c, f in COUPLE.get(j, []):
                    J[:, k] += f * jac[:, self.dadr[f"{hand}_{c}"]]
                J[:, k] *= weights[k]
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-9 * np.eye(3), err)
            for k, j in enumerate(joints):
                lo, hi = self.m.jnt_range[self.m.joint(f"{hand}_{j}").id]
                self.set(q, j, float(np.clip(q[j] + 0.5 * weights[k] * dq[k], lo, hi)))
        d = self.kin({hand: q})
        return q, float(np.linalg.norm(target - d.site_xpos[sid]))

    # ---------- 规划 ----------
    def plan(self, text):
        base = {h: self.typing_posture(h) for h in hsb.SIDES}
        # 每根手指弯曲量 c 与指尖最低点高度的关系（手臂高度为 0 时）
        grid = np.linspace(0, 1.6, 81)
        zc = {}
        for f in ("index", "middle", "ring", "little"):
            zs = []
            for c in grid:
                q = dict(base["L"]); self.set(q, f + "_mcp", c)
                if f in ("index", "middle"):
                    self.set(q, f + "_pip", c)
                zs.append(self.lowest(self.kin({"L": q}), "L", f)[2])
            zc[f] = np.array(zs)
        # 手的高度：以最难够到屏幕的手指为准，让它也能比按键深度再低 0.8 mm
        arm_z = (TOP - PRESS - 0.0008) - max(z.min() for z in zc.values())
        self.zc, self.grid = zc, grid
        self.hover_curl = {}
        for f, z in zc.items():  # 每根手指的悬停弯曲量：在下降段上，指尖刚好在键面以上 LIFT
            k = int(np.argmax(z + arm_z <= TOP + LIFT))
            self.hover_curl[f] = float(grid[k])
        for h in hsb.SIDES:
            base[h]["arm_z"] = arm_z
            for f, c in self.hover_curl.items():
                self.set(base[h], f + "_mcp", c)
                if f in ("index", "middle"):
                    self.set(base[h], f + "_pip", c)
            d = self.kin(base)
            base[h]["arm_x"] = self.keys["f"][0] - self.lowest(d, h, "index")[0]  # 手指大致在中间一行上方
        self.base = base
        strokes = []
        for ch in text:
            hand, finger = FINGER_OF[ch]
            kx, ky = self.keys[ch]
            if finger in ("ring", "little"):
                # 只有一个弯曲电机的手指：先找弯曲量让指尖降到按键深度，再平移手臂对准键心（精确解）
                lo = self.hover_curl[finger]
                hi = float(self.grid[int(np.argmin(self.zc[finger]))])  # 只在下降段上找
                for _ in range(40):
                    c = (lo + hi) / 2
                    q = dict(base[hand]); self.set(q, f"{finger}_mcp", c)
                    z = self.lowest(self.kin({hand: q}), hand, finger)[2]
                    lo, hi = (c, hi) if z > TOP - PRESS else (lo, c)
                qp = dict(base[hand]); self.set(qp, f"{finger}_mcp", (lo + hi) / 2)
                tip = self.lowest(self.kin({hand: qp}), hand, finger)
                qp["arm_x"] += kx - tip[0]
                qp["arm_y"] += ky - tip[1]
                err = float(np.linalg.norm(self.lowest(self.kin({hand: qp}), hand, finger) - [kx, ky, TOP - PRESS]))
            else:
                joints = ["arm_x", "arm_y", "wrist_yaw"] + JOINTS_OF[finger]
                w = [W_ARM, W_ARM, W_YAW] + [1.0] * len(JOINTS_OF[finger])
                qp, err = self.ik(hand, finger, [kx, ky, TOP - PRESS], base[hand], joints, w)
            hover = dict(base[hand])
            for j in ("arm_x", "arm_y", "wrist_yaw"):
                hover[j] = qp[j]
            if finger not in ("ring", "little"):  # 悬停时指尖正好在目标键上方，垂直落下，不擦邻键
                lift = 0.005 if finger == "thumb" else LIFT  # 拇指从更高处落下，避免路过最下一行时擦到键
                qh, _ = self.ik(hand, finger, [kx, ky, TOP + lift], {**hover, **{j: qp[j] for j in qp if j.startswith(finger)}},
                                JOINTS_OF[finger], [1.0] * len(JOINTS_OF[finger]))
                for j, v in qh.items():
                    if j.startswith(finger):
                        hover[j] = v
            lift_q = hover
            if finger == "thumb":  # 拇指按完先抬到键面以上 8 mm，再收回掌下，收回途中不会擦到最下一行
                ql, _ = self.ik(hand, finger, [kx, ky, TOP + 0.008], hover, JOINTS_OF[finger], [1.0] * 3)
                lift_q = dict(hover); lift_q.update({j: v for j, v in ql.items() if j.startswith(finger)})
            strokes.append({"ch": ch, "key": hsb.hb.key_name(ch), "hand": hand, "finger": finger,
                            "press": qp, "hover": hover, "lift": lift_q, "ik_err": err})
        return strokes

    # ---------- 控制 ----------
    def k_eff(self, hand, j):
        """关节的等效回弹刚度（加上被它带动的联动关节）。"""
        k = float(self.m.jnt_stiffness[self.m.joint(f"{hand}_{j}").id])
        for c, f in COUPLE.get(j, []):
            k += f * f * float(self.m.jnt_stiffness[self.m.joint(f"{hand}_{c}").id])
        return k

    def ctrl_of(self, hand, q):
        """电机指令 = 目标线长 + 弹簧前馈（提前加上抵消关节回弹所需的拉力，伺服就不会"差一点到不了"）。"""
        out = {}
        for tn, lst in hsb.TENDONS.items():
            length = sum(c * q[j] for j, c in lst)
            force = np.mean([self.k_eff(hand, j) * q[j] / c for j, c in lst])
            out[self.act[f"{hand}_{tn}"]] = length + force / 6000.0
        for j in ("arm_x", "arm_y", "arm_z"):
            out[self.act[f"{hand}_{j}"]] = q[j]
        return out

    def command(self, hand, q, dur):
        for i, v in self.ctrl_of(hand, q).items():
            self.c0[i], self.c1[i] = self.d.ctrl[i], v
            self.ts[i], self.te[i] = self.d.time, self.d.time + max(dur, 1e-3)

    def type_text(self, text, t_move, t_press, t_gap, label="十指打字"):
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        strokes = self.plan(text)
        nu = m.nu
        self.c0, self.c1 = np.zeros(nu), np.zeros(nu)
        self.ts, self.te = np.zeros(nu), np.full(nu, 1e-3)
        self.frames, self.touches = [], []
        # 起手：两只手在键盘上方 6 mm 处到位，再垂直落下
        for h in hsb.SIDES:
            q = dict(self.base[h]); q["arm_z"] += 0.006
            for j, v in q.items():
                if f"{h}_{j}" in self.qadr:
                    d.qpos[self.qadr[f"{h}_{j}"]] = v
            for i, v in self.ctrl_of(h, q).items():
                d.ctrl[i] = self.c0[i] = self.c1[i] = v
        mujoco.mj_forward(m, d)
        for h in hsb.SIDES:
            self.command(h, self.base[h], 0.3)
        # 时间表
        T0 = 0.45
        t_lift = max(0.06, 0.4 * t_move)
        events, ready, last_press = [], {h: T0 for h in hsb.SIDES}, -1.0
        for i, s in enumerate(strokes):
            pr = max(ready[s["hand"]] + t_move, last_press + t_gap)
            mv = pr - t_move  # 到该按的时候才移过去，等待时手指不悬在键上
            s["t_move"], s["t_press"] = mv, pr
            ready[s["hand"]] = pr + t_press + (t_lift if s["finger"] == "thumb" else 0.0)
            last_press = pr
            events += [(mv, 0, i), (pr, 1, i)]
        for i, s in enumerate(strokes):
            nxt = next((t for t in strokes[i + 1:] if t["hand"] == s["hand"]), None)
            end = s["t_press"] + t_press
            idle = (nxt["t_move"] - end) if nxt else 1.0
            # 拇指按完一定先垂直抬起（否则收回掌下时会擦过最下一行）；
            # 其他手指如果马上要按下一个键，就把"抬起"合并进移动；如果要等，先抬起再回到放松姿势
            if s["finger"] == "thumb" or idle > t_lift:
                events.append((end, 3, i))
                if idle > t_lift + 0.02:
                    rest = dict(self.base[s["hand"]])
                    for j in ("arm_x", "arm_y", "wrist_yaw"):
                        rest[j] = s["hover"][j]
                    s["rest"] = rest
                    events.append((end + t_lift, 4, i))
        for h in hsb.SIDES:
            events.append((ready[h], 2, h))
        events.sort(key=lambda e: (e[0], {0: 2, 1: 3, 2: 0, 3: 0, 4: 1}[e[1]]))  # 同一时刻：先抬起/归位，再移动，最后按下
        t_end = max(ready.values()) + 0.35
        # 仿真
        every = int(1 / FPS / m.opt.timestep)
        active = set()
        last_contact = {}  # 每个键最近一次被接触的时间（防抖：断开不足 40 ms 又接触，不算新的一次按键）
        ei, step, self.min_dist = 0, 0, 0.0
        geom_owner = {}
        for g in range(m.ngeom):
            n = m.geom(g).name
            for h in hsb.SIDES:
                for f in ("index", "middle", "ring", "little", "thumb"):
                    if n.startswith(f"{h}_{f}_g"):
                        geom_owner[g] = (h, f)
        key_ids = set(self.key_geom.values())
        while d.time < t_end:
            while ei < len(events) and events[ei][0] <= d.time:
                t, kind, i = events[ei]
                if kind == 0:
                    s = strokes[i]; self.command(s["hand"], s["hover"], t_move * 0.9)
                elif kind == 1:
                    s = strokes[i]; self.command(s["hand"], s["press"], t_press * 0.5)
                elif kind == 3:
                    s = strokes[i]; self.command(s["hand"], s["lift"], t_lift * 0.9)
                elif kind == 4:
                    s = strokes[i]; self.command(s["hand"], s["rest"], 0.15)
                else:
                    self.command(i, self.base[i], 0.2)
                ei += 1
            a = np.clip((d.time - self.ts) / (self.te - self.ts), 0, 1)
            a = 0.5 - 0.5 * np.cos(np.pi * a)
            d.ctrl[:] = self.c0 + (self.c1 - self.c0) * a
            d.qfrc_applied[:] = d.qfrc_bias  # 重力补偿
            mujoco.mj_step(m, d)
            now = set()
            for k in range(d.ncon):
                c = d.contact[k]
                self.min_dist = min(self.min_dist, float(c.dist))
                g1, g2 = c.geom1, c.geom2
                kg = g1 if g1 in key_ids else (g2 if g2 in key_ids else None)
                og = g2 if kg == g1 else g1
                if kg is not None and og in geom_owner:
                    now.add((m.geom(kg).name[4:], og))
            touched_keys = {k for k, _ in active}
            for k, g in now - active:
                recent = d.time - last_contact.get(k, -1.0) < 0.04
                if k not in touched_keys and not recent:
                    h, f = geom_owner[g]
                    pos = None
                    for ci in range(d.ncon):
                        c = d.contact[ci]
                        if {c.geom1, c.geom2} == {self.key_geom[k], g}:
                            pos = c.pos.copy()
                    self.touches.append({"t": d.time, "key": k, "hand": h, "finger": f, "pos": pos})
                    touched_keys.add(k)
            for k, _ in now:
                last_contact[k] = d.time
            active = now
            if self.recording and step % every == 0:
                self.record(label)
            step += 1
        return strokes, self.evaluate(strokes, t_move, t_press)

    def evaluate(self, strokes, t_move, t_press):
        typed = "".join(" " if t["key"] == "space" else t["key"] for t in self.touches)
        used = set()
        res = []
        for s in strokes:
            hit = None
            for k, t in enumerate(self.touches):
                if k in used or t["hand"] != s["hand"] or t["finger"] != s["finger"]:
                    continue
                if s["t_press"] - 0.5 * t_move <= t["t"] <= s["t_press"] + t_press:
                    hit = k
                    break
            if hit is not None:
                used.add(hit)
                t = self.touches[hit]
                kx, ky = self.keys[s["ch"]]
                err = float(np.hypot(t["pos"][0] - kx, t["pos"][1] - ky) * 1000)
                res.append({"key": s["key"], "hit": t["key"], "error_mm": err, "hand": s["hand"], "finger": s["finger"]})
            else:
                res.append({"key": s["key"], "hit": None, "error_mm": None, "hand": s["hand"], "finger": s["finger"]})
        mistouch = [t for k, t in enumerate(self.touches) if k not in used]
        first, last = strokes[0]["t_press"], strokes[-1]["t_press"] + t_press
        errs = [r["error_mm"] for r in res if r["error_mm"] is not None and r["key"] != "space"]  # 空格键很宽，不计入定位误差
        return {"typed": typed, "correct": typed == TEXT, "hits": sum(r["hit"] == r["key"] for r in res), "n": len(res),
                "mistouch": len(mistouch), "mistouch_keys": [t["key"] for t in mistouch],
                "cps": len(strokes) / (last - first), "wpm": len(strokes) / 5 / (last - first) * 60,
                "err_mean": float(np.mean(errs)) if errs else None, "err_max": float(max(errs)) if errs else None,
                "results": res}

    def record(self, label):
        m, d = self.m, self.d
        g = []
        for i in range(m.ngeom):
            q = np.zeros(4)
            mujoco.mju_mat2Quat(q, d.geom_xmat[i])
            g.append([round(float(x) * 1000, 2) for x in d.geom_xpos[i]] + [round(float(x), 3) for x in q])
        ten = {}
        for h in hsb.SIDES:
            for grp, names in (("index", ["index_mcp", "index_pip", "index_abd"]), ("middle", ["middle_mcp", "middle_pip", "middle_abd"]),
                               ("ring", ["ring_flex"]), ("little", ["little_flex", "spread"]),
                               ("thumb", ["thumb_opp", "thumb_cmc", "thumb_flex"]), ("wrist", ["wrist_pitch", "wrist_yaw"])):
                ten[f"{h}_{grp}"] = round(max(abs(float(d.actuator_force[self.act[f"{h}_{n}"]])) for n in names), 2)
        typed = "".join(" " if t["key"] == "space" else t["key"] for t in self.touches)
        self.frames.append({"t": round(d.time, 3), "g": g, "ten": ten, "typed": typed})


SPEEDS = [  # (移动, 按下, 两次按下的最小间隔) 秒
    (0.30, 0.12, 0.10), (0.22, 0.10, 0.08), (0.16, 0.08, 0.06), (0.12, 0.07, 0.05), (0.09, 0.06, 0.04), (0.07, 0.05, 0.03)]


def main():
    h = Hands()
    h.recording = False
    strokes = h.plan(TEXT)
    print(f'文字："{TEXT}"；逆运动学最大残差 {max(s["ik_err"] for s in strokes)*1000:.2f} mm')
    print("指法：" + "  ".join(f"{s['ch'] if s['ch'] != ' ' else '空格'}={'左' if s['hand']=='L' else '右'}{FINGER_NAMES[s['finger']]}" for s in strokes))
    print(f"\n  {'参数（移动/按下/间隔）':<22}{'速度':<22}{'打出的字':<22}{'按对':<7}{'误触':<6}{'平均误差':<10}{'最大误差'}")
    rows, best = [], None
    for tm, tp, tg in SPEEDS:
        _, r = h.type_text(TEXT, tm, tp, tg)
        ok = r["correct"] and r["hits"] == r["n"] and r["mistouch"] == 0
        rows.append(((tm, tp, tg), r))
        print(f"  {tm:.2f}/{tp:.2f}/{tg:.2f} s{'':<9}{r['cps']:.1f} 字符/秒（{r['wpm']:.0f} 词/分）{'':<2}"
              f"{r['typed']!r:<22}{r['hits']}/{r['n']:<5}{r['mistouch']:<6}"
              f"{(r['err_mean'] or 0):.2f} mm{'':<3}{(r['err_max'] or 0):.2f} mm{'  ✓' if ok else ''}")
        if ok:
            best = ((tm, tp, tg), r)
    if not best:
        print("\n没有完全不出错的速度档。")
        return
    (tm, tp, tg), r = best
    print(f"\n不出错的最高速度：{r['cps']:.1f} 字符/秒，约 {r['wpm']:.0f} 词/分（单指为 3.3 字符/秒）")
    h.recording = True
    strokes, r = h.type_text(TEXT, tm, tp, tg)
    print(f"重新录制：打出 {r['typed']!r}，最深穿透 {-h.min_dist*1000:.2f} mm")
    m = h.m
    geoms = [{"name": m.geom(i).name, "type": int(m.geom_type[i]), "size": [round(float(x) * 1000, 3) for x in m.geom_size[i]],
              "rgba": [round(float(x), 3) for x in m.geom_rgba[i]]} for i in range(m.ngeom)]
    out = {"geoms": geoms, "text": TEXT, "frames": h.frames, "summary": {k: v for k, v in r.items()},
           "speeds": [{"params": list(p), "cps": rr["cps"], "wpm": rr["wpm"], "typed": rr["typed"], "hits": rr["hits"], "n": rr["n"],
                       "mistouch": rr["mistouch"], "err_mean": rr["err_mean"], "err_max": rr["err_max"]} for p, rr in rows],
           "strokes": [{"ch": s["ch"], "hand": s["hand"], "finger": s["finger"], "t": round(s["t_press"], 3)} for s in strokes],
           "penetration_mm": round(-h.min_dist * 1000, 2), "t_max": {"finger": hsb.hb.T_FINGER, "wrist": hsb.hb.T_WRIST}}
    path = os.path.join(HERE, "..", "hands_frames.js")
    with open(path, "w", encoding="utf-8") as f:
        f.write("window.HANDS = " + json.dumps(out, ensure_ascii=False, separators=(",", ":")) + ";\n")
    print(f"已导出 {os.path.relpath(path, HERE)}（{os.path.getsize(path)/1e6:.1f} MB，{len(h.frames)} 帧）")


if __name__ == "__main__":
    main()
