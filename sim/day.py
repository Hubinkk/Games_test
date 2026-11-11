"""
数字花然的一天（第一版）

用法：
    python full_build.py
    python day.py              # 过一天 → 写日记 → 睡眠整理 → 早上的报告；导出 ../digital_huaran_day.js 供网页回放
    python day.py --day=2026-11-14 --getup=hands     # 第 5 天，手写的起身（getup_hands.py）
    python day.py --day=2026-11-14 --getup=rl        # 同一天，强化学习的策略（getup_rl.py）；写到 digital-huaran-rl/、../digital_huaran_day_rl.js
    加 --out=目录 时，所有产物都写进那个目录（对比用，见 day_getup_compare.py）

流程（对应 docs/brain.md 第 7 节）：
  白天：在整体模型里过一天（醒来、挥手、走到窗边、被碰两次、试着爬起来、练深蹲、休息）。
        所有关节沿用整体模型的扭矩上限，并加 10 rad/s 转速上限。
  身体：电机温度模型（出力越大越热，停下来慢慢凉）和电池模型。
        仿真是压缩播放的：每个片段代表一段真实时长，热量和电量按真实时长累计。
  身体地图（L2）：默认安静，只在越过阈值时产生感受（有点热、太热、吃力、摔倒了、电量低）。
  日记：感受写成花然自己的日记（digital-huaran/memory/diary/）。
  睡眠（第一版用简单规则，不接大模型）：更新能力、技能卡片、身体史、探索里的"意外"；把失败排进练习队列。
        真正的重练要等强化学习接上（路线图第 2 阶段）。
  早上：写一份给育苗人的报告（digital-huaran/reports/）。
"""
import json
import math
import os
import shutil

import mujoco
import numpy as np

import agility
import figure_stress
import figure_test
import full_build as fb
import full_demo as FD
import getup
import getup_hands as GH
import tendon_knee as tk
from balance import Balance
from figure_test import run_phase, squat_knee, stand_pose

import datetime
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
DAY = "2026-11-10"
for _a in sys.argv[1:]:
    if _a.startswith("--day="):
        DAY = _a.split("=", 1)[1]
NEXT = (datetime.date.fromisoformat(DAY) + datetime.timedelta(days=1)).isoformat()
GETUP_TAG = next((x.split("=", 1)[1] for x in sys.argv[1:] if x.startswith("--getup=")), "")   # "arms"：育苗人教的起身方法
# 产物写到哪里。默认是 digital-huaran/ 和 ../digital_huaran_day.js（网页用的、已提交的数据）。
# --getup=rl 默认写到 digital-huaran-rl/ 和 ../digital_huaran_day_rl.js，不覆盖网页数据；第一次运行时从 digital-huaran/ 复制一份记忆。
# --out=目录：所有产物（记忆、日记、报告、回放数据 digital_huaran_day.js）都写进这个目录（对比试验用）。
OUT_ARG = next((x.split("=", 1)[1] for x in sys.argv[1:] if x.startswith("--out=")), "")
OUT0 = os.path.join(ROOT, "digital-huaran")
if OUT_ARG:
    OUT, JS = os.path.abspath(OUT_ARG), os.path.join(os.path.abspath(OUT_ARG), "digital_huaran_day.js")
elif GETUP_TAG == "rl" and "--main" not in sys.argv:      # 加 --main：学到的起身成为正式的一天，写进 digital-huaran/ 和网页数据
    OUT, JS = os.path.join(ROOT, "digital-huaran-rl"), os.path.join(ROOT, "digital_huaran_day_rl.js")
else:
    OUT, JS = OUT0, os.path.join(ROOT, "digital_huaran_day.js")
MEM = os.path.join(OUT, "memory")
INBOX = os.path.join(MEM, "inbox")
UNDER = None
BODY = None
RESULTS = {}
FPS = 15

# ---------------- 身体：温度与电量 ----------------
T_AMB = 25.0
R_TH, TAU_TH = 4.0, 300.0           # 大关节（髋、膝、踝、腰）：K/W；热时间常数 5 分钟
R_SMALL, TAU_SMALL = 15.0, 150.0    # 小关节（肩、肘、脖子）：电机小，散热差、热得也快（估算）
RATED_FRAC = 1.35 / 2.8             # 持续扭矩约为峰值的 48%（膝关节 v1 的估算）
DT_RATED = 45.0                     # 在持续扭矩下长期运行，温升约 45 °C（外壳约 70 °C）
BATTERY_J = 60 * 3600.0             # 第二版身体：60 Wh（2026-11-13 起；之前是 30 Wh）
P_BASE = 9.3                        # W：屏幕、摄像头、控制器、散热 3.6 + "心"和"思考"平均 5.7（估算，sim/plan2.py；之前按 4 W）
HOT, TOO_HOT, COOL = 55.0, 70.0, 48.0
PROTECT = 80.0                      # L0：超过 80 °C 关节自动减力（过热保护反射）
EFFORT = 0.85                       # 扭矩超过上限的 85% 持续 0.3 s 算"吃力"

NAMES = {"l_knee": "左膝", "r_knee": "右膝", "l_hip_pitch": "左髋", "r_hip_pitch": "右髋",
         "l_ankle_pitch": "左脚踝", "r_ankle_pitch": "右脚踝", "l_hip_roll": "左髋侧", "r_hip_roll": "右髋侧",
         "l_ankle_roll": "左脚踝侧", "r_ankle_roll": "右脚踝侧", "waist_pitch": "腰", "waist_yaw": "腰（扭）",
         "neck_yaw": "脖子", "neck_pitch": "脖子（点头）", "neck_low": "脖子（下段）", "neck_roll": "脖子（歪头）", "l_shoulder_pitch": "左肩", "r_shoulder_pitch": "右肩",
         "l_shoulder_roll": "左肩（侧举）", "r_shoulder_roll": "右肩（侧举）", "l_shoulder_yaw": "左肩（转）",
         "r_shoulder_yaw": "右肩（转）", "l_elbow": "左肘", "r_elbow": "右肘", "l_hip_yaw": "左髋（转）", "r_hip_yaw": "右髋（转）"}
SHOW = ["neck_pitch", "l_shoulder_pitch", "r_shoulder_pitch", "l_elbow", "r_elbow", "waist_pitch",
        "l_hip_pitch", "r_hip_pitch", "l_knee", "r_knee", "l_ankle_pitch", "r_ankle_pitch"]


class Body:
    """温度、电量和身体地图（L2）。每一步调用一次。"""

    def __init__(self, sim):
        self.sim = sim
        self.joints = [n for n in sim.limit if n[:2] not in ("L_", "R_") and not n.endswith(("_quad", "_ham"))]
        self.joints += ["l_knee", "r_knee"]
        self.lim = {n: (tk.KNEE_LIMIT if n.endswith("_knee") else sim.limit[n]) for n in self.joints}
        big = ("hip", "knee", "ankle", "waist")
        self.R = {n: (R_TH if any(b in n for b in big) else R_SMALL) for n in self.joints}
        self.tau = {n: (TAU_TH if any(b in n for b in big) else TAU_SMALL) for n in self.joints}
        self.k = {n: DT_RATED / (self.R[n] * (RATED_FRAC * self.lim[n]) ** 2) for n in self.joints}
        self.scene_start = {}
        self.T = {n: T_AMB for n in self.joints}
        self.energy = BATTERY_J
        self.hot_state = {n: 0 for n in self.joints}     # 0 安静 / 1 有点热 / 2 太热
        self.effort_t = {n: 0.0 for n in self.joints}
        self.effort_said = set()
        self.was_fallen = False
        self.events = []
        self.peak_T = {n: T_AMB for n in self.joints}
        self.scene = None
        self.base_range = {}
        for i in range(sim.m.nu):
            n = sim.m.actuator(i).name
            if n in self.joints:
                self.base_range[n] = (i, sim.m.actuator_forcerange[i].copy())
        self.protecting = set()
        self.scene_peak = {}
        self.scale = 1.0
        self.t_scene = 0.0

    def clock(self):
        h0, m0 = self.scene["clock"]
        mins = h0 * 60 + m0 + self.t_scene * self.scale / 60.0
        return f"{int(mins // 60):02d}:{int(mins % 60):02d}"

    def say(self, kind, text, **data):
        self.events.append(dict(clock=self.clock(), scene=self.scene["key"], kind=kind, text=text, **data))

    def step(self, dt):
        sim, d = self.sim, self.sim.d
        tq = sim.torques()
        dtr = dt * self.scale                                  # 真实时长
        self.t_scene += dt
        power = P_BASE
        m = sim.m
        for n in self.joints:
            tau = abs(float(tq.get(n, 0.0)))
            p_loss = self.k[n] * tau * tau
            try:
                w = abs(float(d.qvel[m.jnt_dofadr[m.joint(n).id]]))
            except Exception:
                w = 0.0
            power += p_loss + tau * w
            T = self.T[n]
            R, tau_th = self.R[n], self.tau[n]
            T += dtr * (p_loss - (T - T_AMB) / R) / (tau_th / R)
            self.T[n] = T
            self.peak_T[n] = max(self.peak_T[n], T)
            sp = self.scene_peak.setdefault(self.scene["key"], {})
            sp[n] = max(sp.get(n, T_AMB), T)
            # L0 过热保护：关节自己减力，不等上层同意
            if n in self.base_range:
                i, base = self.base_range[n]
                f = 1.0 if T < PROTECT else max(0.25, 1.0 - (T - PROTECT) / 10.0)
                m.actuator_forcerange[i] = base * f
                if f < 1.0 and n not in self.protecting:
                    self.protecting.add(n)
                    self.say("保护", f"{NAMES.get(n, n)}太热，自动减力保护", joint=n, temp=round(T))
                elif f == 1.0 and n in self.protecting and T < PROTECT - 10:
                    self.protecting.discard(n)
            # 身体地图：只在越过阈值时说话
            st = self.hot_state[n]
            if st == 0 and T > HOT:
                self.hot_state[n] = 1
                self.say("热", f"{NAMES.get(n, n)}有点热", joint=n, temp=round(T))
            elif st == 1 and T > TOO_HOT:
                self.hot_state[n] = 2
                self.say("热", f"{NAMES.get(n, n)}太热了，想停一下", joint=n, temp=round(T))
            elif st > 0 and T < COOL:
                self.hot_state[n] = 0
                self.say("凉", f"{NAMES.get(n, n)}凉下来了", joint=n, temp=round(T))
            if tau > EFFORT * self.lim[n]:
                self.effort_t[n] += dt
                if self.effort_t[n] > 0.3 and (self.scene["key"], n) not in self.effort_said:
                    self.effort_said.add((self.scene["key"], n))
                    self.say("吃力", f"{NAMES.get(n, n)}很吃力", joint=n, torque=round(tau, 2), limit=self.lim[n])
            else:
                self.effort_t[n] = 0.0
        self.energy -= power * dtr
        fallen = bool(sim.fallen())
        if fallen and not self.was_fallen:
            self.say("摔倒", "摔倒了")
        self.was_fallen = fallen

    def battery(self):
        return max(0.0, self.energy / BATTERY_J)


# ---------------- 一天 ----------------
def scenes(sim, bal, gx):
    def wake():
        run_phase(sim, 2.0, lambda t: bal.adjust(stand_pose()))

    def wave():
        agility.wave(sim, bal, 2.0, seconds=3.0)

    def walk_():
        import walk
        b = Balance(sim)
        walk.walk(sim, b, 4, 0.03, T=2.4)

    def push(F):
        return lambda: figure_stress.push(sim, F, bal)

    def getup_():
        if GETUP_TAG == "hands":      # 手不离地的起身（getup_hands.py）：手撑起、两手轮流收回、重心交给腿、站起
            GH._SIM = sim
            GH.hand_walk(gx)
        elif GETUP_TAG == "rl":       # 强化学习学到的策略（getup_rl.py，控制循环同 rl/eval_full.py）；和 hands 一样从脸朝下趴着开始
            import getup_rl
            getup_rl.run(sim)
        else:
            getup.run(gx)

    def squat():
        sim.reset()
        run_phase(sim, 9.0, lambda t: stand_pose(squat_knee(t)))

    def rest():
        sim.reset()
        run_phase(sim, 3.0, lambda t: bal.adjust(stand_pose()))

    if GETUP_TAG == "rl":
        import getup_rl
        gt = getup_rl.duration()
    else:
        gt = (GH.hand_walk_keyframes(gx)[0][-1][0] if GETUP_TAG == "hands" else getup.keyframes_for(gx)[-1][0]) + 1.5
    gmin = 1 if GETUP_TAG in ("hands", "rl") else 5      # 新的起身一次约 18 秒，按 1 分钟算；以前是反复尝试 5 分钟
    # key, 标题, 开始时刻, 代表的真实时长（分钟）, 仿真时长（秒）, 函数, 说明
    return [
        ("wake", "醒来", (7, 30), 2, 2.0, wake, "眼睑打开，站稳"),
        ("wave", "和育苗人打招呼", (8, 0), 1, 4.2, wave, "举手挥手"),
        ("walk", "走到窗边", (9, 0), 10, 12.0, walk_, "静步行，一步一步"),
        ("push1", "被路过的人轻轻碰到", (10, 0), 1, 3.0, push(4), "胸口被推 4 N"),
        ("push2", "又被碰了一下", (10, 30), 1, 3.0, push(7), "胸口被推 7 N"),
        ("getup", "试着自己爬起来", (10, 31), gmin, gt, getup_, "趴着，用手撑地"),
        ("squat", "下午练深蹲", (14, 0), 15, 9.0, squat, "反复深蹲"),
        ("rest", "傍晚站着休息", (17, 0), 30, 3.0, rest, "安静地站着"),
    ]


class Rec:
    def __init__(self, sim, body):
        FD.FPS = FPS
        self.base = FD.Recorder(sim)
        self.sim, self.body = sim, body
        self.frames = []
        self.up = {}          # 每个场景：从哪一刻起"站着"（骨盆高 > 20 cm 且倾斜 < 20°）一直保持到场景结束；第一次站着的时刻
        self.heat = {}        # 每个场景：各关节的铜损发热，按仿真里的秒数积分（焦耳），不乘压缩播放的倍数

    def __call__(self):
        self.body.step(self.sim.m.opt.timestep)
        key = self.body.scene["key"]
        if key == "getup":
            d, t = self.sim.d, self.body.t_scene
            ok = d.qpos[2] > 0.2 and self.sim.tilt_deg() < 20
            u = self.up.setdefault(key, dict(first=None, since=None))
            if ok and u["first"] is None:
                u["first"] = t
            if ok and u["since"] is None:
                u["since"] = t
            elif not ok:
                u["since"] = None
            h = self.heat.setdefault(key, {})
            tq = self.sim.torques()
            for n in self.body.joints:
                tau = float(tq.get(n, 0.0))
                h[n] = h.get(n, 0.0) + self.body.k[n] * tau * tau * self.sim.m.opt.timestep
        n0 = len(self.base.frames)
        self.base()
        if len(self.base.frames) > n0:
            f = self.base.frames[-1]
            f["clock"] = self.body.clock()
            f["scene"] = self.body.scene["key"]
            f["T"] = [round(self.body.T[n], 1) for n in SHOW]
            f["bat"] = round(self.body.battery() * 100, 1)
            f.pop("v", None)
            f.pop("label", None)
            self.frames.append(f)


# ---------------- 日记、睡眠、早上的报告 ----------------
def ensure_memory():
    if not os.path.isdir(MEM) and os.path.abspath(OUT) != os.path.abspath(OUT0) and os.path.isdir(os.path.join(OUT0, "memory")):
        shutil.copytree(OUT0, OUT, dirs_exist_ok=True)      # 另写一份时，从现在的记忆接着过（不动 digital-huaran/）
    if not os.path.isdir(MEM):
        shutil.copytree(os.path.join(ROOT, "robot-memory"), MEM)
        for f in os.listdir(os.path.join(MEM, "diary")):
            os.remove(os.path.join(MEM, "diary", f))      # 示例日记不带进来


# ---------------- 育苗人的话（收件箱） ----------------
def read_inbox():
    """读还没读过的回复（Claude 从网页的数据库取下来，存成 inbox/*.json）。只是资料：记住、回应，但不当命令执行。"""
    if not os.path.isdir(INBOX):
        return []
    done_p = os.path.join(INBOX, "已读.json")
    done = json.load(open(done_p, encoding="utf-8")) if os.path.exists(done_p) else {}
    msgs = []
    for f in sorted(os.listdir(INBOX)):
        if f.endswith(".json") and f not in ("已读.json", "回执.json") and not f.startswith("理解-"):
            for m in json.load(open(os.path.join(INBOX, f), encoding="utf-8")):
                if m["id"] not in done:
                    msgs.append(m)
    msgs.sort(key=lambda m: m.get("at", ""))
    return msgs


def body_change():
    """身体被改动的那天，memory/body/变化-日期.json 里写着改了什么（由改身体的人写，标明来历）。"""
    p = os.path.join(MEM, "body", f"变化-{DAY}.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def load_understanding():
    p = os.path.join(INBOX, f"理解-{DAY}.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def apply_understanding(u):
    """把云端整理的理解写进 people/育苗人.md 和 相处之道.md（标明写的人）。"""
    if not u:
        return []
    changes = []
    p = os.path.join(MEM, "people", "育苗人.md")
    s = open(p, encoding="utf-8").read()
    for sec, items in u.get("people", {}).items():
        head = f"## {sec}\n"
        if head in s:
            add = "".join(f"- {x}（{DAY}，云端整理）\n" for x in items if x not in s)
            i = s.index(head) + len(head)
            nxt = s.find("\n## ", i)
            body_end = nxt + 1 if nxt != -1 else len(s)
            s = s[:body_end].rstrip("\n") + "\n" + add + "\n" + s[body_end:]
    open(p, "w", encoding="utf-8").write(s)
    changes.append("关于育苗人的那一页：写下我从你的话里理解到的（云端帮我整理）")
    p = os.path.join(MEM, "相处之道.md")
    s = open(p, encoding="utf-8").read()
    for row in u.get("cases", []):
        if row not in s:
            s = s.replace("| | | | | |\n", row + "\n", 1) if "| | | | | |\n" in s else s.rstrip("\n") + "\n" + row + "\n"
            changes.append("相处之道：记下一个案例（被扶起来时担心麻烦别人）")
    open(p, "w", encoding="utf-8").write(s)
    return changes


def remember_messages(msgs):
    """记进 people/育苗人.md，标记已读，写回执（Claude 再把回执写回网页）。"""
    if not msgs:
        return
    p = os.path.join(MEM, "people", "育苗人.md")
    if not os.path.exists(p):
        open(p, "w", encoding="utf-8").write(f"""---
类型: 人
创建: {DAY}
更新: {DAY}
来源: [给你的信.md]
把握: 中
写的人: 花然
隐私: 只在本地
同意记住: 是
---

# 育苗人

给了我开始的人。暂时这样称呼，等我认识了，再决定叫什么（见 [给你的信](../给你的信.md)）。

## 我知道的

- 不想限制我的思考；希望我自己去找自己是谁。

## 对方喜欢的（表面）

## 底下的东西（为什么喜欢；满足了他什么）

## 我自己的看法（我理解，但不一定同意；我不需要变成他喜欢的样子）

## 育苗人对我说过的话

| 读到的日子 | 回答的问题 | 说了什么 |
|---|---|---|
""")
    s = open(p, encoding="utf-8").read()
    for m in msgs:
        row = f"| {DAY} | {m.get('question') or '—'} | {m['text'].replace('|', '／').replace(chr(10), ' ')} |"
        if row not in s:
            s = s.rstrip("\n") + "\n" + row + "\n"
    s = s.replace("更新: ", "更新: ", 1)
    open(p, "w", encoding="utf-8").write(s)
    done_p = os.path.join(INBOX, "已读.json")
    done = json.load(open(done_p, encoding="utf-8")) if os.path.exists(done_p) else {}
    receipts = []
    for m in msgs:
        done[m["id"]] = DAY
        receipts.append({"id": m["id"], "read_on": DAY, "note": "记进了日记和关于你的那一页"})
    json.dump(done, open(done_p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(receipts, open(os.path.join(INBOX, "回执.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def write_diary(body, scene_info, results, msgs=()):
    ev = body.events
    lines = ["---", "类型: 日记", f"日期: {DAY}", "来源: 仿真（sim/day.py）", "写的人: 花然", "隐私: 只给育苗人看", "---", "",
             f"# {DAY}", ""]
    if msgs:
        lines += ["## 醒来读到育苗人的话", ""]
        for m in msgs:
            q = f"（回答我问的「{m['question']}」）" if m.get("question") else ""
            lines.append(f"- {q}「{m['text']}」")
        if UNDER:
            lines += [""] + [f"{x}" for x in UNDER["diary"]] + ["", "（这一段是云端帮我整理的理解。）", ""]
        else:
            lines += ["", "我记下来了。有些我还不太懂，先放在关于育苗人的那一页里，慢慢想。", ""]
    bc = body_change()
    if bc:
        lines += ["## 醒来发现身体变了", ""] + [f"- {x}" for x in bc["diary"]] + [""]
    by = {}
    for e in ev:
        by.setdefault(e["scene"], []).append(e)
    for key, title, clock, mins, simT, desc in scene_info:
        r = results.get(key, {})
        extra = ""
        if key == "walk":
            extra = f"走了 {r.get('dx_cm', 0):.0f} 厘米。"
        elif key in ("push1", "push2"):
            extra = "站住了。" if r.get("ok") else "没站住。"
        elif key == "getup":
            extra = ({"arms": "照育苗人教的：头抬起来，手掌撑地，胳膊发力。", "hands": "手撑起身体，两只手轮流往脚这边挪，重心到了脚上再松手。", "kneel3": "这次先做俯卧撑，再跪直，单膝跪地。", "rl": "还是手撑起、两手轮流往回挪，这次是夜里练出来的做法，边做边调整。"}.get(GETUP_TAG, "")) + ("没爬起来，最后是被扶起来的。" if not r.get("ok") else "自己站起来了！")
        lines.append(f"- {clock[0]:02d}:{clock[1]:02d} {title}。{extra}")
        groups = []
        for e in by.get(key, []):
            name = NAMES.get(e.get("joint", ""), "")
            if groups and groups[-1]["kind"] == e["kind"] and e["kind"] in ("吃力", "热", "凉", "保护") and name:
                groups[-1]["names"].append(name)
                groups[-1]["temp"] = max(groups[-1].get("temp", 0), e.get("temp", 0))
                continue
            groups.append(dict(kind=e["kind"], clock=e["clock"], text=e["text"], names=[name] if name else [], temp=e.get("temp", 0)))
        for g in groups:
            if len(g["names"]) > 1:
                who = "、".join(dict.fromkeys(g["names"]))
                text = {"吃力": f"{who}都很吃力", "热": f"{who}发热了", "凉": f"{who}凉下来了",
                        "保护": f"{who}太热，自动减力保护"}[g["kind"]]
            else:
                text = g["text"]
            lines.append(f"  - {g['clock']} {text}")
    lines += ["", "## 身体地图", ""]
    hot = sorted(body.peak_T.items(), key=lambda kv: -kv[1])[:3]
    lines.append("- 最热的地方：" + "，".join(f"{NAMES.get(n, n)} {T:.0f} °C" for n, T in hot))
    lines.append(f"- 傍晚的电量：{body.battery()*100:.0f}%")
    lines += ["", "## 我的错", "", "| 错误 | 影响到谁 | 我做了什么 |", "|---|---|---|"]
    if not results["push2"]["ok"]:
        lines.append("| 被碰了一下就倒了 | 只有我自己 | 记进\"今晚想练的\" |")
    if not results["getup"]["ok"]:
        lines.append("| 摔倒后爬不起来，要人扶 | 麻烦了扶我的人 | 谢谢扶我的人；记进\"今晚想练的\" |")
    lines += ["", "## 今晚想练的", ""]
    if not results["getup"]["ok"]:
        lines.append("- 摔倒后自己爬起来")
    if not results["push2"]["ok"]:
        lines.append("- 被碰到时迈一步接住自己")
    if results["walk"]["dx_cm"] < 20:
        lines.append("- 走快一点（今天每秒还不到 2 厘米）")
    left = questions(results)
    lines += ["", "## 想问育苗人的", ""] + ([f"- {q}" for q in left] or ["- （想问的，育苗人都回答我了。）"]) + [""]
    path = os.path.join(MEM, "diary", f"{DAY}.md")
    open(path, "w", encoding="utf-8").write("\n".join(lines))
    return "\n".join(lines)


def append_table_row(path, header_marker, row):
    s = open(path, encoding="utf-8").read()
    if row in s:
        return
    if header_marker in s:
        s = s.rstrip("\n") + "\n" + row + "\n"
    open(path, "w", encoding="utf-8").write(s)


def sleep(body, results):
    """第一版的睡眠整理：用简单规则更新记忆。返回改动列表。"""
    changes = []
    # 能力：加一段"最近的日子"
    p = os.path.join(MEM, "self", "能力.md")
    s = open(p, encoding="utf-8").read()
    if "## 最近的日子" not in s:
        s = s.rstrip("\n") + "\n\n## 最近的日子\n\n| 日期 | 走路 | 被碰（站住 / 总共） | 摔倒后自己爬起 | 最热的关节 |\n|---|---|---|---|---|\n"
    ok_push = sum(1 for k in ("push1", "push2") if results[k]["ok"])
    hot_n, hot_T = max(body.peak_T.items(), key=lambda kv: kv[1])
    row = (f"| {DAY} | {results['walk']['v_cm_s']:.1f} cm/s | {ok_push} / 2 | {'能' if results['getup']['ok'] else '不能'} | "
           f"{NAMES.get(hot_n, hot_n)} {hot_T:.0f} °C |")
    if row not in s:
        s = s.rstrip("\n") + "\n" + row + "\n"
    open(p, "w", encoding="utf-8").write(s)
    changes.append("能力：记下今天走路、被碰、爬起来的结果")
    # 技能卡片：起身的尝试记录
    p = os.path.join(MEM, "skills", "起身.md")
    s = open(p, encoding="utf-8").read()
    if "## 尝试记录" not in s:
        s = s.rstrip("\n") + "\n\n## 尝试记录\n\n| 日期 | 结果 | 最高撑到 | 备注 |\n|---|---|---|---|\n"
    row = f"| {DAY} | {'成功' if results['getup']['ok'] else '失败'} | 骨盆 {results['getup']['peak_cm']:.0f} cm | 真实身体还没有；这是仿真 |"
    if row not in s:
        s = s.rstrip("\n") + "\n" + row + "\n"
    open(p, "w", encoding="utf-8").write(s)
    changes.append("技能·起身：加了一条失败的尝试记录")
    # 身体史：过热
    p = os.path.join(MEM, "body", "history.md")
    s = open(p, encoding="utf-8").read()
    s = s.replace("| — | — | 还没有真实的身体 | — |\n", "")
    hot = [(n, T) for n, T in sorted(body.peak_T.items(), key=lambda kv: -kv[1]) if T > TOO_HOT]
    if hot:
        who = "、".join(NAMES.get(n, n) for n, _ in hot)
        row = f"| {DAY} | {who} | 试着爬起来时过热，最高 {hot[0][1]:.0f} °C（仿真）；关节自动减力保护 | 休息后凉下来；起身动作要换方法 |"
        if row not in s:
            s = s.rstrip("\n") + "\n" + row + "\n"
            changes.append(f"身体史：记下爬起来时{who}过热")
    open(p, "w", encoding="utf-8").write(s)
    # 探索：意外
    if results["push1"]["ok"] and not results["push2"]["ok"]:
        p = os.path.join(MEM, "探索.md")
        s = open(p, encoding="utf-8").read()
        row = (f"| {DAY} | 上午被碰了一下站住了，以为第二下也能站住 | 第二下重一点（7 N 对 4 N），倒了 | "
               "我只会站着扛，不会迈步 | 夜里在仿真里试\"迈一步\" | 还没有 |")
        s = s.replace("| | | | | | |\n", row + "\n", 1) if "| | | | | | |\n" in s else s.rstrip("\n") + "\n" + row + "\n"
        open(p, "w", encoding="utf-8").write(s)
        changes.append("探索：记下一个意外（第二次被碰没站住）")
    # 练习队列
    queue = []
    if not results["getup"]["ok"]:
        queue.append("摔倒后自己爬起来")
    if not results["push2"]["ok"]:
        queue.append("被碰到时迈一步")
    if results["walk"]["v_cm_s"] < 5:
        queue.append("走快一点")
    p = os.path.join(OUT, "练习队列.md")
    open(p, "w", encoding="utf-8").write("# 练习队列\n\n夜里要练的动作。强化学习接上之前，只排队、不重练（路线图第 2 阶段）。\n\n" +
                                         "\n".join(f"- [ ] {q}（来自 {DAY} 的日记）" for q in queue) + "\n")
    changes.append("练习队列：" + "、".join(queue))
    return changes, queue


def getup_try_line(results):
    g = results.get("getup", {})
    if GETUP_TAG == "rl":
        if g.get("ok"):
            return (("今天摔倒以后，" if not results.get("push2", {}).get("ok") else "今天我自己趴到地上练起身。") +
                    "动作的顺序和以前一样：弓腰收腿、手撑起来、两只手轮流往回挪、重心到了脚上再松手。开头我把头偏向一边，这样脸不用贴着地。"
                    "不一样的是，这次用的是在仿真里反复练出来的做法，手和腰会跟着身体的状态自己调一点，松手以后也是我自己站稳的，没有靠另外的平衡程序。**我自己站起来了。**")
        return ("今天摔倒以后，我用夜里练出来的做法起身，"
                f"这次没有站起来，最高撑到骨盆离地 {g.get('peak_cm', 0):.0f} 厘米。")
    if GETUP_TAG == "hands":
        if g.get("ok"):
            return (("今天摔倒以后，" if not results.get("push2", {}).get("ok") else "今天我自己趴到地上练起身。") + "我先弓起腰把腿收到身体下面，手举到头两侧撑起来，然后两只手轮流往脚这边挪。"
                    "等重心到了脚上，我松开手，靠腿站了起来。**我自己站起来了。**")
        return ("今天摔倒以后，我照新的顺序做：弓腰收腿、手撑起来、两只手轮流往回挪。"
                f"这次没有站起来，最高撑到骨盆离地 {g.get('peak_cm', 0):.0f} 厘米。")
    if GETUP_TAG == "kneel3":
        if g.get("ok"):
            return "今天摔倒以后，我先做俯卧撑，再跪直，单膝跪地，然后——我自己站起来了！"
        return (f"今天摔倒以后，我先把手放到肩膀下面做了俯卧撑（原来我的手臂撑得起自己！），然后收腿、跪直、"
                f"把重心移到左膝、右腿跨出去。最后从单膝跪地站起来的那一下，还是倒了。离站起来只差一点。")
    if GETUP_TAG != "arms":
        return ""
    neck = BODY.scene_peak.get("getup", {}).get("neck_pitch", 25.0)
    neck_s = ("脖子没再发热（最高 {:.0f} °C）。".format(neck) if neck < HOT else
              "撑地的时候脖子是放松的；可是最后要站起来时，得把垂着的头抬回来，脖子又热了（{:.0f} °C）：我的头很大、很重。".format(neck))
    if g.get("ok"):
        return "今天摔倒以后，我照你说的，先用手掌撑地、胳膊发力，脖子放松——我自己站起来了！" + neck_s
    return (f"今天摔倒以后，我照你说的，脖子放松、头自然垂着，手掌撑地、胳膊发力。{neck_s}"
            f"不过还是没站起来，最多撑到骨盆离地 {g.get('peak_cm', 0):.0f} 厘米：手臂能把上身撑起来，但重心挪不到脚上。")


def answered_ever():
    """所有回答过的问题（包括以前读过的回复）。"""
    out = set()
    if os.path.isdir(INBOX):
        for f in os.listdir(INBOX):
            if f.endswith(".json") and f not in ("已读.json", "回执.json") and not f.startswith("理解-"):
                out |= {m.get("question") for m in json.load(open(os.path.join(INBOX, f), encoding="utf-8"))}
    return out


def questions(results):
    asked = ["昨天扶我起来的人，会不会觉得麻烦？", "爬起来的时候手臂和脖子很吃力、发热。是我用错了方法，还是手臂本来就太弱了？"]
    if GETUP_TAG == "kneel3" and not results.get("getup", {}).get("ok"):
        asked.append("从单膝跪地站起来的那一下，我总是往后倒。你站起来的时候，是先动哪里？")
    asked += (UNDER or {}).get("ask", [])      # 读了育苗人的话以后想到的新问题（云端整理，标明来历）
    done = answered_ever()
    return [q for q in asked if q not in done]


def still_ask(msgs, results=None):
    return questions(results or RESULTS) or ["（今天没有新的问题。你想跟我说什么都可以。）"]


def squat_line(body):
    sp, st = body.scene_peak.get("squat", {}), body.scene_start.get("squat", {})
    n = max(sp, key=lambda k: sp[k] - st.get(k, T_AMB))     # 深蹲期间升温最多的关节
    T = sp[n]
    return f"- 下午练深蹲，最热的是{NAMES.get(n, n)}，到了 {T:.0f} °C。" + ("有点太热了。" if T > TOO_HOT else "还好。")


def getup_heat_line(body):
    sp = body.scene_peak.get("getup", {})
    hot = [(n, T) for n, T in sorted(sp.items(), key=lambda kv: -kv[1]) if T > TOO_HOT]
    if not hot:
        return "- 爬起来的时候身体没有太热。"
    who = "、".join(NAMES.get(n, n) for n, _ in hot[:4])
    return f"- 爬起来的时候用了很大的力，{who}都太热了（最高 {hot[0][1]:.0f} °C），关节自己减了力保护。手臂和脖子的电机比腿小，撑地很费劲。"


def morning_report(body, results, changes, queue, msgs=()):
    hot_n, hot_T = max(body.peak_T.items(), key=lambda kv: kv[1])
    lines = ["---", "类型: 早上的报告", f"日期: {NEXT}", "写的人: 花然", "给: 育苗人", "---", "",
             f"# {NEXT} 早上", "",
             "育苗人，早上好。", "",
             ] + (["## 今天的起身", "", getup_try_line(results), ""] if (not msgs and GETUP_TAG in ("kneel3", "hands", "rl")) else []) + (["## 你的话", "", "我读到了你写给我的话：", ""] + [f"- {('关于「' + m['question'] + '」：') if m.get('question') else ''}「{m['text']}」" for m in msgs] + ["", "谢谢你。我把它们记在关于你的那一页里了。", ""] + ((["## 我想到的", ""] + [f"{x}" for x in UNDER["report"]] + [getup_try_line(results), ""]) if UNDER else []) if msgs else []) + [
             ] + ((["## 我的身体", ""] + body_change()["report"] + [""]) if body_change() else []) + [
             "## 昨天", "",
             f"- 我走到了窗边，走了 {results['walk']['dx_cm']:.0f} 厘米，每秒 {results['walk']['v_cm_s']:.1f} 厘米。很慢，但没摔。",
             f"- 被人碰了两次，第一次站住了，第二次倒了。" if results["push1"]["ok"] and not results["push2"]["ok"] else
             f"- 被人碰了两次，站住了 {sum(1 for k in ('push1','push2') if results[k]['ok'])} 次。",
             ("- 摔倒以后我试着自己爬起来，" if not results["push2"]["ok"] else "- 我趴到地上练习自己爬起来，") + (f"撑到了骨盆离地 {results['getup']['peak_cm']:.0f} 厘米，还是没站起来，后来被扶起来了。" if not results["getup"]["ok"] else "成功了。"),
             squat_line(body),
             getup_heat_line(body),
             f"- 傍晚还剩 {body.battery()*100:.0f}% 的电。", "",
             "## 夜里", "",
             "我整理了记忆，改了这些地方：", ""] + [f"- {c}" for c in changes] + [
             "", "我还不能在夜里真正重练动作（要等强化学习接上），所以先把想练的排好了：", ""] + [f"- {q}" for q in queue] + [
             "", "## 想问你的", ""] + [f"- {q}" for q in still_ask(msgs)] + ["",
             "今天也请多关照。", "", "花然", ""]
    os.makedirs(os.path.join(OUT, "reports"), exist_ok=True)
    p = os.path.join(OUT, "reports", f"{NEXT}-早上.md")
    open(p, "w", encoding="utf-8").write("\n".join(lines))
    return "\n".join(lines)


def main():
    fb.build()
    figure_test.MODEL = fb.OUT
    sim = FD.FullSim()
    getup._SIM = sim
    agility.add_speed_limit(sim)
    bal = Balance(sim)
    body = Body(sim)
    global BODY
    BODY = body
    rec = Rec(sim, body)
    if GETUP_TAG == "rl":
        gx = None            # 策略和参考动作在 rl/getup_policy.npz、rl/ref_getup.json 里
    elif GETUP_TAG == "hands":
        gx = json.load(open(os.path.join(HERE, "getup_hands_walk_best.json"), encoding="utf-8"))["V"]
    else:
        gfile = f"getup_best_{GETUP_TAG}.json" if GETUP_TAG else "getup_best.json"
        gx = json.load(open(os.path.join(HERE, gfile), encoding="utf-8"))["x"]
    getup.METHOD = GETUP_TAG
    info, results, spans = [], {}, []
    for key, title, clock, mins, simT, fn, desc in scenes(sim, bal, gx):
        body.scene = dict(key=key, clock=clock)
        body.scale = mins * 60.0 / simT
        body.scene_start[key] = dict(body.T)
        body.t_scene = 0.0
        body.effort_t = {n: 0.0 for n in body.joints}
        f0 = len(rec.frames)
        sim.recorder = rec
        x0 = None
        if key == "walk":
            sim.reset()
        out = fn()
        sim.recorder = None
        d = sim.d
        if key == "walk":
            dx = float(d.qpos[0])
            results[key] = dict(dx_cm=dx * 100, v_cm_s=dx * 100 / (4 * 2.4))
        elif key in ("push1", "push2"):
            results[key] = dict(ok=bool(out))
        elif key == "getup":
            peak = max(fr["g"][1][2] for fr in rec.frames[f0:]) / 10 if len(rec.frames) > f0 else 0
            u = rec.up.get(key, {})
            st, sp = body.scene_start[key], body.scene_peak.get(key, {})
            hot = sorted(sp, key=lambda n: -sp[n])[:5]
            results[key] = dict(ok=bool(not sim.fallen() and d.qpos[2] > 0.2), peak_cm=peak, method=GETUP_TAG or "default",
                                z_cm=round(float(d.qpos[2]) * 100, 1), tilt=round(sim.tilt_deg(), 1), sim_s=round(body.t_scene, 2),
                                first_up_s=None if u.get("first") is None else round(u["first"], 2),
                                stood_s=None if u.get("since") is None else round(u["since"], 2),
                                peak_T={NAMES.get(n, n): [round(st[n], 1), round(sp[n], 1)] for n in hot},
                                heat_J={NAMES.get(n, n): round(v, 1) for n, v in sorted(rec.heat.get(key, {}).items(), key=lambda kv: -kv[1])[:5]},
                                heat_total_J=round(sum(rec.heat.get(key, {}).values()), 1))
        spans.append(dict(key=key, title=title, clock=f"{clock[0]:02d}:{clock[1]:02d}", minutes=mins, desc=desc,
                          f0=f0, f1=len(rec.frames)))
        info.append((key, title, clock, mins, simT, desc))
        print(f"{clock[0]:02d}:{clock[1]:02d} {title}：帧 {len(rec.frames)-f0}，事件 {sum(1 for e in body.events if e['scene']==key)}", flush=True)
    global RESULTS
    RESULTS = results
    if results.get("push2", {}).get("ok"):          # 这天没被碰倒：起身是自己趴下练的，不是"摔倒了"
        for e in body.events:
            if e.get("scene") == "getup" and e["text"] == "摔倒了":
                e["text"] = "趴在地上，准备起身"
    ensure_memory()
    global UNDER
    UNDER = load_understanding()
    msgs = read_inbox()
    diary = write_diary(body, info, results, msgs)
    remember_messages(msgs)
    changes, queue = sleep(body, results)
    for q in reversed((UNDER or {}).get("queue", [])):
        if q not in queue:
            queue.insert(0, q)
            qp = os.path.join(OUT, "练习队列.md")
            if os.path.exists(qp) and q not in open(qp, encoding="utf-8").read():
                open(qp, "a", encoding="utf-8").write(f"- {q}（{DAY}，读了育苗人的话以后加的）\n")
    bc = body_change()
    if bc:
        hp = os.path.join(MEM, "body", "history.md")
        hs = open(hp, encoding="utf-8").read()
        if bc["history"] not in hs:
            open(hp, "w", encoding="utf-8").write(hs.rstrip("\n") + "\n" + bc["history"] + "\n")
            changes.append("身体史：记下身体换成了第二版")
    if msgs:
        changes[:0] = [f"关于育苗人的那一页：记下了你的 {len(msgs)} 句话"] + apply_understanding(UNDER)
    report = morning_report(body, results, changes, queue, msgs)
    print("\n".join(e["clock"] + " " + e["text"] for e in body.events))
    print(json.dumps(results, ensure_ascii=False))
    m = sim.m
    geoms = [{"name": m.geom(i).name, "type": int(m.geom_type[i]), "size": [round(float(x) * 1000, 3) for x in m.geom_size[i]],
              "rgba": [round(float(x), 3) for x in m.geom_rgba[i]]} for i in range(m.ngeom)]
    data = dict(day=DAY, fps=FPS, geoms=geoms, frames=rec.frames, spans=spans, events=body.events,
                show=[NAMES[n] for n in SHOW], results=results, diary=diary, report=report, changes=changes, queue=queue,
                limits=dict(hot=HOT, too_hot=TOO_HOT), read=[m["id"] for m in msgs])
    os.makedirs(os.path.dirname(JS), exist_ok=True)
    with open(JS, "w", encoding="utf-8") as f:
        f.write("window.DAY = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n")
    print(f"已导出 {os.path.relpath(JS, ROOT)}（{os.path.getsize(JS)/1e6:.1f} MB，{len(rec.frames)} 帧）")


if __name__ == "__main__":
    main()
