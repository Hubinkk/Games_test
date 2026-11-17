"""
夜里练习的一圈（第一版，只有"规则"这一层，不接大模型）：
    练习队列 → 判定（night_practice.py）→ 按规则表选下一次怎么练 → 训练（train_walk.py）→ 自动验收 → 写结果 → 没全过、还有预算就再来一次。

用法（在 sim/rl/ 里，用 ~/venv-sys 的 python 跑这个脚本；训练用 .venv，评测用 ~/venv-sys，都是子进程）：
    python night_loop.py --from=w3b --knobs=effort=0.08,stop=20,push=12,lr=0.0002
            # 把一次已经练完的训练（walk_full_w3b.npz + walk_policy_w3b_<步数>M.npz）当作第 1 次尝试：先验收，再按规则接着练
    python night_loop.py                        # 接着上次没跑完的（状态在 night_loop_state.json）
    python night_loop.py --hours=7 --attempts=3 --steps=50000000
    python night_loop.py --dry-run              # 只判定、选规则、打印要跑的命令，不训练、不评测（用已有的评测 JSON）
    python night_loop.py --smoke-train          # 训练那一步换成 train_walk.py --smoke（几分钟，只测"开训练、等它结束"这一段；不验收）
    python night_loop.py --fresh                # 不接上次的状态，从头开始

写的东西（只写这些地方，写之前检查路径）：
    sim/rl/night_loop_state.json        状态：第几次尝试、旋钮、完整存档的名字、每条检查的数（断了能接着来）
    sim/rl/walk_*_<tag>*、eval_*.json   训练和评测的产物（由子进程写）；sim/day_walk_compare_<策略>.json（发热，由子进程写）
    practice-results/<日期>.json/.md    night_practice.py 的格式（云端 day.py 读最新的 .json）
    practice-results/<日期>-log.md      夜里的记录：每次尝试改了什么旋钮、为什么、练了多久、每条检查的数

写死的边界（违反就拒绝运行）：只改训练的命令行旋钮（ALLOWED_KNOBS，有硬上下限）；不改技能卡片、验收标准（就是这里的 ACCEPT 表）、
sim/full_build.py；不写 digital-huaran/、digital_huaran_day.js；不换 walk_policy.npz；扭矩上限和 10 rad/s 转速上限不动
（walk_common.RATE 必须是 10）；同一时间只有一个训练（有别的 train_*.py 在跑就等，不杀）；预算 --hours、--attempts，开训练前估时间。
全部检查都过了也不换 walk_policy.npz：结果里写"练成了，等育苗人决定换不换"和候选策略的文件名。
"""
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.normpath(os.path.join(HERE, ".."))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
PY_TRAIN = os.path.join(HERE, ".venv", "bin", "python")
PY_EVAL = os.path.expanduser("~/venv-sys/bin/python")
STATE = os.path.join(HERE, "night_loop_state.json")
RESULTS = os.path.join(ROOT, "practice-results")

# ---------------- 边界 ----------------
# 训练命令行里允许改的旋钮和硬上下限（规则表的上下限不能超出这里）
ALLOWED_KNOBS = {"effort": (0.02, 0.16), "stop": (0.0, 80.0), "push": (6.0, 14.0), "lr": (1e-5, 3e-4), "qvel": (0.0, 8.0)}
FIXED_FLAGS = ["--upper", "--smooth", "--envs=4096"]            # 每次都带、不改
# 绝对不能写的地方（写文件前检查）和绝对不能变的文件（开始时记下指纹，每一步之后对一次；变了就停）
FORBIDDEN_PATHS = [os.path.join(ROOT, "digital-huaran"), os.path.join(ROOT, "digital_huaran_day.js"),
                   os.path.join(HERE, "walk_policy.npz"), os.path.join(SIM, "full_build.py")]
PROTECTED_FILES = [os.path.join(HERE, "walk_policy.npz"), os.path.join(SIM, "full_build.py"), os.path.join(SIM, "figure55_full.xml"),
                   os.path.join(ROOT, "digital_huaran_day.js"), os.path.join(ROOT, "digital-huaran", "memory", "skills", "走路.md")]
WRITE_OK = [HERE, RESULTS]

# ---------------- 验收表（docs/handoff-opus.md 第 4 节的 5 条 + 技能卡片 digital-huaran/memory/skills/走路.md 的回归测试） ----------------
# 每一行：id、名字（"发热"两个字要留着，云端 day.py 靠它认发热这条）、来源、更好的方向（判"有没有改善"用）
ACCEPT = [
    dict(id="no_fall_basic", unit="falls", name="基本项目不摔（--hold、--hold --upper、--hold --smooth：走停、走+推5N、快走、原地转、侧移后退）", src="交接第 4 节①；卡片：平地走 6 步不摔、走路中被推 3 N 不摔", better="lower"),
    dict(id="speed15", unit="cm_s", name="指令 15 cm/s 实际 14–16", src="交接第 4 节①", better="closer15"),
    dict(id="back8", unit="cm_s", name="后退指令 8 cm/s 实际 ≥ 7", src="交接第 4 节①", better="higher"),
    dict(id="drift", unit="cm", name="停下后最后 2 秒挪动平均 < 1 cm（上身动时 < 2 cm）", src="交接第 4 节①", better="lower"),
    dict(id="stand_push8", unit="falls", name="站着被推 8 N 全过", src="交接第 4 节②", better="lower"),
    dict(id="stand_push12", unit="falls", name="站着被推 12 N 过一半以上", src="交接第 4 节②", better="lower"),
    dict(id="walk_push12", unit="falls", name="走着被推 12 N 过一半以上", src="交接第 4 节②", better="lower"),
    dict(id="goto", unit="failed_runs", name="\"去那里\"不推 48/48", src="交接第 4 节③", better="lower"),
    dict(id="goto_push5", unit="failed_runs", name="\"去那里\"推 5 N 48/48", src="交接第 4 节③", better="lower"),
    dict(id="goto_push5_upper", unit="failed_runs", name="\"去那里\"推 5 N + 上身随意摆 ≥ 43/48", src="交接第 4 节③", better="lower"),
    dict(id="heat", unit="max_10min_C", name="发热：连续走 10 分钟，任何电机不超过 70 °C", src="交接第 4 节④；卡片", better="lower"),
    dict(id="turn", unit="falls", name="原地转身 90°（原地转不摔、转速 > 0.2 rad/s）", src="卡片", better="lower"),
    dict(id="fast20", unit="cm_s", name="走到 20 cm/s 以上（快走实际）", src="卡片", better="higher"),
    dict(id="joint_normal", unit="max_rad_s", name="关节转速不超过 10 rad/s，膝盖扭矩不超过 2.8 N·m（正常走）", src="卡片", better="lower"),
    dict(id="joint_push", unit="max_rad_s", name="被推后 1 秒内：转速只记录（已豁免），膝盖扭矩不超过 2.8 N·m", src="卡片；转速豁免是用户 2026-11-17 的决定", better="lower"),
]
SCREEN_IDS = ["no_fall_basic", "speed15", "back8", "drift", "stand_push8", "stand_push12", "walk_push12", "turn", "fast20", "joint_normal"]

# ---------------- 规则表（走路；来自第 2、3 轮和 w3b 的经验，见 wiki/灵活度.md） ----------------
# 一行：哪些检查没过（并且满足 when）→ 改哪个旋钮、怎么改、改到哪为止。conflict：和哪条规则抢同一个旋钮、方向相反
RULES = [
    dict(id="heat", checks=["heat"], when="any", knob="effort", op="mul", by=2.0, limit=0.16, why="发热没过 → 费力扣分加倍"),
    dict(id="stop", checks=["drift", "goto", "goto_push5", "goto_push5_upper"], when="drift_cause", knob="stop", op="set_then_mul", set=20.0, by=2.0, limit=80.0,
         why="停下后挪动没过，或\"去那里\"没过而且是停不住（没摔、离目标不远但还在挪）→ 指令为零时动的扣分从 0 到 20，再不过加倍"),
    dict(id="slow", checks=["speed15"], when="slow", knob="effort", op="div", by=1.5, limit=0.02, conflict="heat",
         why="速度偏慢（指令 15 实际 < 14）→ 费力扣分除以 1.5"),
    dict(id="push", checks=["stand_push8", "stand_push12", "walk_push12", "no_fall_basic"], when="falls", knob="push", op="add", by=2.0, limit=14.0,
         why="被推摔得多 → 训练里的推力上限加 2 N"),
    # 2026-11-17 加：正常走时关节转速超 10 rad/s。不动 effort（n1117b 发热 68.9 °C，离 70 只差 1 °C），也不动力矩上限和 10 rad/s 的目标角限速，
    # 用 train_walk.py 新加的 --qvel（腿上关节转速超过 8 rad/s 的部分的平方扣分，默认 0）
    dict(id="qvel", checks=["joint_normal"], when="speed", knob="qvel", op="set_then_mul", set=1.0, by=2.0, limit=4.0,
         why="正常走时关节转速超 10 rad/s → 转速超过 8 rad/s 的部分的扣分从 0 到 1，再不过加倍"),
]


def arg(name, default=None):
    for a in sys.argv[1:]:
        if a.startswith(f"--{name}="):
            return a.split("=", 1)[1]
    return default


def flag(name):
    return f"--{name}" in sys.argv[1:]


def sha1(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None


def check_write(path):
    p = os.path.normpath(os.path.abspath(path))
    for f in FORBIDDEN_PATHS:
        if p == f or p.startswith(f + os.sep):
            raise SystemExit(f"拒绝：要写的 {p} 在不许写的范围里（{f}）")
    if not any(p == w or p.startswith(w + os.sep) for w in WRITE_OK):
        raise SystemExit(f"拒绝：要写的 {p} 不在允许写的目录里（{WRITE_OK}）")
    return p


def write_text(path, text, mode="w"):
    with open(check_write(path), mode, encoding="utf-8", newline="\n") as f:
        f.write(text)


def write_json(path, obj):
    write_text(path, json.dumps(obj, ensure_ascii=False, indent=1))


def preflight():
    """开跑前的检查：规则表不能超出硬上下限、旋钮只能是允许的那几个、转速上限没被改。"""
    for r in RULES:
        if r["knob"] not in ALLOWED_KNOBS:
            raise SystemExit(f"拒绝：规则 {r['id']} 改的旋钮 {r['knob']} 不在允许的范围里")
        lo, hi = ALLOWED_KNOBS[r["knob"]]
        if not lo <= r["limit"] <= hi:
            raise SystemExit(f"拒绝：规则 {r['id']} 的上下限 {r['limit']} 超出了硬边界 {lo}–{hi}")
        for c in r["checks"]:
            if c not in {a["id"] for a in ACCEPT}:
                raise SystemExit(f"拒绝：规则 {r['id']} 指向不存在的检查 {c}")
    sys.path.insert(0, HERE)
    import walk_common as W
    if abs(W.RATE - 10.0) > 1e-9:
        raise SystemExit(f"拒绝：walk_common.RATE 是 {W.RATE}，转速上限必须是 10 rad/s")
    for p in (PY_TRAIN, PY_EVAL):
        if not os.path.exists(p):
            raise SystemExit(f"找不到 {p}")


def guard(fp):
    """受保护的文件（在用的策略、身体模型、技能卡片……）一步之后有没有变。变了就停：这一圈不该碰它们。"""
    now = {p: sha1(p) for p in PROTECTED_FILES}
    bad = [p for p in PROTECTED_FILES if now[p] != fp[p]]
    if bad:
        raise SystemExit("停：这些受保护的文件在跑的过程中变了（不是这个脚本改的也停，交给人看）：" + "、".join(os.path.relpath(p, ROOT) for p in bad))


def other_training():
    """有没有别的 train_*.py 在跑（只看，不杀）。"""
    out = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout
    return [l.strip() for l in out.splitlines() if re.search(r"python\S*\s+\S*train_\w+\.py", l) and str(os.getpid()) not in l.split()[0]]


# ---------------- 日志 ----------------
class Log:
    def __init__(self, day):
        self.path = os.path.join(RESULTS, f"{day}-log.md")
        if not os.path.exists(self.path):
            write_text(self.path, f"# 夜里的记录 {day}\n\n由 `sim/rl/night_loop.py` 写（规则这一层，不接大模型）。数字都是仿真里的，温度是估算。\n\n")

    def __call__(self, *lines):
        stamp = datetime.datetime.now().strftime("%H:%M")
        text = "\n".join(lines)
        print(f"[{stamp}] {text}", flush=True)
        write_text(self.path, f"- {stamp} " + text.replace("\n", "\n  ") + "\n", mode="a")


# ---------------- 评测（子进程调现有脚本） ----------------
def run(cmd, cwd, log_to=None):
    t0 = time.time()
    if log_to:
        with open(check_write(log_to), "w") as f:
            r = subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT)
    else:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} 出错（返回 {r.returncode}）" + ("" if log_to else "：" + (r.stderr or "")[-1500:]))
    return (time.time() - t0) / 60


def load(p):
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


EVAL_WALK = {"hold": ["--hold"], "upper": ["--hold", "--upper"], "smooth": ["--hold", "--smooth"]}
EVAL_GOTO = {"goto": [], "goto_push5": ["--push=5"], "goto_push5_upper": ["--push=5", "--upper"]}


def walk_json(name, k):
    return os.path.join(HERE, f"eval_walk_{name}_" + "_".join(a.strip("-") for a in EVAL_WALK[k]) + ".json")


def goto_json(name, k):
    return os.path.join(HERE, f"eval_goto_{name}" + "".join("_" + a.strip("-").replace("=", "") for a in EVAL_GOTO[k]) + ".json")


def heat_json(name):
    return os.path.join(SIM, f"day_walk_compare_{name}.json")


def evaluate(name, parts, dry):
    """parts：要哪些评测（hold/upper/smooth/goto.../heat）。已有结果而且是新版 eval_walk（记了转速）就不重跑。返回用了几分钟。"""
    mins = 0.0
    for k in parts:
        if k in EVAL_WALK:
            p = walk_json(name, k)
            d = load(p)
            if d and d.get("trials") == 8 and all("qvel_max" in r for r in d["runs"]):
                continue
            cmd = [PY_EVAL, "eval_walk.py", f"{name}.npz", *EVAL_WALK[k], "--trials=8", "--procs=15"]
            cwd = HERE
        elif k in EVAL_GOTO:
            if load(goto_json(name, k)):
                continue
            cmd, cwd = [PY_EVAL, "eval_goto.py", f"{name}.npz", *EVAL_GOTO[k], "--procs=15"], HERE
        else:                                         # heat
            if load(heat_json(name)):
                continue
            cmd, cwd = [PY_EVAL, "day_walk_compare.py", f"--policy={name}.npz"], SIM
        if dry:
            print("  （dry-run，不跑）" + " ".join(os.path.basename(c) for c in cmd))
            continue
        mins += run(cmd, cwd)
    return mins


def cells(runs, test):
    """按地面分组：{摩擦: [runs]}"""
    out = {}
    for r in runs:
        if r["test"] == test:
            out.setdefault(r["friction"], []).append(r)
    return out


def seg_mean(rs, i, key):
    xs = [r["segs"][i][key] for r in rs if len(r["segs"]) > i and key in r["segs"][i]]
    return sum(xs) / len(xs) if xs else None


PUSH_EXEMPT = "被推后 1 秒内的转速已豁免：外力带动，用户 2026-11-17 决定；只记录，不算不过关"


def judge(name, only=None, screen=False):
    """按 ACCEPT 表判一份策略。返回 {id: dict(ok, value, note, detail)}；缺评测的是 ok=None。
    screen=True（筛副本）：只有 --hold 的结果也照判（标明"只看 --hold"），不然这几条全是"没量"、筛不出东西。"""
    W = {k: load(walk_json(name, k)) for k in EVAL_WALK}
    G = {k: load(goto_json(name, k)) for k in EVAL_GOTO}
    H = load(heat_json(name))
    res = {}
    have = [k for k in EVAL_WALK if W[k]]
    allruns = [r for k in have for r in W[k]["runs"]]

    def put(i, ok, value, note, **detail):
        res[i] = dict(ok=ok, value=value, note=note, **detail)

    if not have:
        for a in ACCEPT:
            if a["id"] not in ("goto", "goto_push5", "goto_push5_upper", "heat"):
                put(a["id"], None, None, "没有 eval_walk 结果")
    else:
        full = len(have) == 3 or screen
        miss = "" if full else ((f"（筛副本：只看 {'、'.join(have)}）") if screen else f"（只有 {'、'.join(have)}，缺 {'、'.join(k for k in EVAL_WALK if k not in have)}）")
        basic = [r for r in allruns if r["test"] in ("走停", "走+推5N", "快走", "原地转", "侧移后退")]
        nf = sum(1 for r in basic if not r["ok"])
        falls = sorted({f"{r['test']}" for r in basic if not r["ok"]})
        put("no_fall_basic", (nf == 0) if full else (False if nf else None), nf, f"摔 {nf}/{len(basic)} 次" + (f"（{'、'.join(falls)}）" if falls else "") + miss, falls=nf)
        sp = [(k, f, seg_mean(rs, 0, "vx") * 100) for k in have for f, rs in cells(W[k]["runs"], "走停").items() if seg_mean(rs, 0, "vx") is not None]
        lo, hi = min(v for *_, v in sp), max(v for *_, v in sp)
        put("speed15", (lo >= 14 and hi <= 16) if full else (False if lo < 14 or hi > 16 else None), round(lo if abs(lo - 15) > abs(hi - 15) else hi, 2),
            f"实际 {lo:.1f}–{hi:.1f} cm/s（每种地面、每种上身设置的平均）" + miss, slow=lo < 14, lo=round(lo, 2), hi=round(hi, 2))
        bk = [-seg_mean(rs, 1, "vx") * 100 for k in have for f, rs in cells(W[k]["runs"], "侧移后退").items() if seg_mean(rs, 1, "vx") is not None]
        put("back8", (min(bk) >= 7) if full else (False if min(bk) < 7 else None), round(min(bk), 2), f"后退 {min(bk):.1f}–{max(bk):.1f} cm/s" + miss)
        dr = {k: max(seg_mean(rs, 1, "drift_cm") for rs in cells(W[k]["runs"], "走停").values()) for k in have}
        dok = all(v < (1.0 if k == "hold" else 2.0) for k, v in dr.items())
        put("drift", dok if full else (False if not dok else None), round(max(dr.values()), 2),
            "停下后最后 2 秒挪 " + "，".join(f"{k} {v:.1f} cm" for k, v in dr.items()) + "（每种地面平均里最大的）" + miss)

        def tally(test, need):
            rs = [r for r in allruns if r["test"] == test]
            ok = sum(1 for r in rs if r["ok"])
            good = (ok == len(rs)) if need == "all" else (ok * 2 > len(rs))
            return good, len(rs) - ok, f"{ok}/{len(rs)} 没摔"
        for i, t, need in (("stand_push8", "站+推8N", "all"), ("stand_push12", "站+推12N", "half"), ("walk_push12", "走+推12N", "half")):
            g, nfail, note = tally(t, need)
            put(i, g if full else (False if not g else None), nfail, note + miss, falls=nfail)
        tr = [r for r in allruns if r["test"] == "原地转"]
        wz = [abs(seg_mean([r], 0, "wz") or 0) for r in tr if r["ok"]]
        g = bool(tr) and all(r["ok"] for r in tr) and wz and min(wz) > 0.2
        put("turn", g, sum(1 for r in tr if not r["ok"]), f"原地转 {sum(r['ok'] for r in tr)}/{len(tr)}" + (f"，转速最小 {min(wz):.2f} rad/s" if wz else ""))
        fz = [seg_mean(rs, 0, "vx") * 100 for f, rs in cells(W["hold"]["runs"] if W["hold"] else allruns, "快走").items()]
        put("fast20", min(fz) >= 20, round(min(fz), 2), f"指令 25 实际 {min(fz):.1f}–{max(fz):.1f} cm/s")
        for i, k in (("joint_normal", "normal"), ("joint_push", "push")):
            rs = [r for r in allruns if r["ok"] and k in r.get("qvel_max", {})]
            if not rs:
                put(i, None, None, "评测里没有记转速（旧版 eval_walk.py）")
                continue
            q = max(rs, key=lambda r: r["qvel_max"][k][0])
            kn = max(r["knee_Nm_max"][k] for r in rs)
            over_q = [r for r in rs if r["qvel_max"][k][0] > 10.0]
            over_k = [r for r in rs if r["knee_Nm_max"][k] > 2.8]
            # 哪些关节超过 10 rad/s、各超了几次（每次试验记的是最快的那个关节）
            jc = {}
            for r in over_q:
                jc[r["qvel_max"][k][1]] = jc.get(r["qvel_max"][k][1], 0) + 1
            n_over = len(over_k) if k == "push" else len({id(r) for r in over_q + over_k})
            note = (f"转速最大 {q['qvel_max'][k][0]} rad/s（{q['qvel_max'][k][1]}，{q['test']}），膝盖最大 {kn} N·m；"
                    f"转速超 10 的 {len(over_q)}/{len(rs)} 次" + (f"（{'、'.join(sorted({r['test'] for r in over_q}))}；关节 "
                    + "、".join(f"{j} {n} 次" for j, n in sorted(jc.items(), key=lambda x: -x[1])) + "）" if over_q else "")
                    + (f"；膝盖超 2.8 N·m 的 {len(over_k)} 次" if over_k else ""))
            if k == "push":
                note += f"。{PUSH_EXEMPT}"
            put(i, n_over == 0, q["qvel_max"][k][0], note + miss, speed_over=bool(over_q) and k == "normal", joints_over=jc)
    for i in ("goto", "goto_push5", "goto_push5_upper"):
        g = G[i]
        if not g:
            put(i, None, None, "没有 eval_goto 结果")
            continue
        rs = g["runs"]
        ok = sum(1 for r in rs if r["ok"])
        need = 43 if i == "goto_push5_upper" else len(rs)
        bad = [r for r in rs if not r["ok"]]
        fell = sum(1 for r in bad if r.get("fell_at") is not None)
        put(i, ok >= need, len(rs) - ok, f"{ok}/{len(rs)} 到了并停住" + (f"；没成的 {len(bad)} 次里摔 {fell} 次，其余是停不住（离目标 "
            + f"{min(r['err_cm'] for r in bad if r.get('fell_at') is None) if len(bad) > fell else '-'}–{max(r['err_cm'] for r in bad if r.get('fell_at') is None) if len(bad) > fell else '-'} cm）" if bad else ""),
            drift_cause=bool(bad) and fell == 0, falls=fell)
    if H and (H.get("continuous_walk_heat") or {}).get("rl"):
        js = H["continuous_walk_heat"]["rl"]["joints"]
        hot = max(js.items(), key=lambda kv: kv[1]["T_10min"])
        over = {j: v["T_10min"] for j, v in js.items() if v["T_10min"] > 70}
        put("heat", not over, hot[1]["T_10min"], f"连续走 10 分钟最热的是{hot[0]} {hot[1]['T_10min']} °C（day.py 热模型，估算，只算腿上 12 个电机）"
            + (f"；超过 70 °C：" + "、".join(f"{j} {t}" for j, t in over.items()) if over else ""))
    else:
        put("heat", None, None, f"没有发热结果（{os.path.relpath(heat_json(name), ROOT)}）")
    if only:
        res = {k: v for k, v in res.items() if k in only}
    return res


def n_fail(res):
    return sum(1 for v in res.values() if v["ok"] is False)


# ---------------- 规则 ----------------
def rule_applies(rule, res):
    hits = [c for c in rule["checks"] if res.get(c, {}).get("ok") is False]
    if not hits:
        return []
    if rule["when"] == "slow":
        hits = [c for c in hits if res[c].get("slow")]
    elif rule["when"] == "drift_cause":
        hits = [c for c in hits if c == "drift" or res[c].get("drift_cause")]
    elif rule["when"] == "speed":
        hits = [c for c in hits if res[c].get("speed_over")]
    elif rule["when"] == "falls":
        hits = [c for c in hits if (res[c].get("falls") or 0) > 0]
    return hits


def new_value(rule, cur):
    lo, hi = ALLOWED_KNOBS[rule["knob"]]
    if rule["op"] == "mul":
        v = cur * rule["by"]
    elif rule["op"] == "div":
        v = cur / rule["by"]
    elif rule["op"] == "add":
        v = cur + rule["by"]
    else:                                             # set_then_mul
        v = rule["set"] if cur <= 0 else cur * rule["by"]
    v = min(v, rule["limit"]) if rule["op"] != "div" else max(v, rule["limit"])
    v = min(max(v, lo), hi)
    return round(v, 6)


def improved(check, before, after):
    a = next(x for x in ACCEPT if x["id"] == check)
    b, c = before.get(check, {}).get("value"), after.get(check, {}).get("value")
    if b is None or c is None:
        return False
    if a["better"] == "lower":
        return c < b
    if a["better"] == "higher":
        return c > b
    return abs(c - 15) < abs(b - 15)


def decide(state):
    """按规则表选下一次的旋钮。返回 (新旋钮, 用了哪些规则, 理由行) 或 (None, None, 停下的理由)。"""
    atts = state["attempts"]
    last = atts[-1]
    res = last["checks"]
    knobs = dict(last["knobs"])
    fails = [i for i, v in res.items() if v["ok"] is False]
    nulls = [i for i, v in res.items() if v["ok"] is None]
    if not fails and not nulls:
        return None, None, "全部检查都过了：练成了"
    if not fails:
        return None, None, "没有没过的，但有没量的：" + "、".join(nulls)
    # 来回：某条检查"过了 → 没过"出现两次
    for c in ACCEPT:
        seq = [a["checks"].get(c["id"], {}).get("ok") for a in atts]
        flips = sum(1 for x, y in zip(seq, seq[1:]) if x is True and y is False)
        if flips >= 2:
            return None, None, f"规则用完了，要人看：「{c['name']}」修好又坏、来回两次"
    chosen, lines, covered = [], [], set()
    for rule in RULES:
        hits = rule_applies(rule, res)
        if not hits:
            continue
        # 同一条规则连用两次都没改善
        used = [a for a in atts[1:] if rule["id"] in a.get("rules", [])]
        if len(atts) >= 3 and all(rule["id"] in a.get("rules", []) for a in atts[-2:]):
            prev = atts[-3]["checks"]
            if not any(improved(h, prev, res) for h in hits):
                return None, None, f"规则用完了，要人看：规则「{rule['why']}」连用两次，{'、'.join(hits)} 都没有改善"
        chosen.append((rule, hits))
        covered.update(hits)
    # 冲突：同一个旋钮被两条规则往相反方向拉
    ids = {r["id"] for r, _ in chosen}
    for r, _ in chosen:
        if r.get("conflict") in ids:
            return None, None, f"两头顶住了，交给人：{r['why']}，但另一条规则（{r['conflict']}）要把同一个旋钮往反方向调"
    if not chosen:
        return None, None, "没有对应的规则，交给人：没过的是 " + "、".join(next(a["name"] for a in ACCEPT if a["id"] == f) for f in fails)
    applied = []
    for rule, hits in chosen:
        cur = knobs.get(rule["knob"], 0.0)
        nv = new_value(rule, cur)
        if abs(nv - cur) < 1e-12:
            lines.append(f"规则 {rule['id']}：{rule['knob']} 已经到 {cur}（上下限 {rule['limit']}），改不动了")
            continue
        knobs[rule["knob"]] = nv
        applied.append(rule["id"])
        lines.append(f"规则 {rule['id']}（{'、'.join(hits)} 没过）：{rule['why']}；{rule['knob']} {cur} → {nv}")
    if not applied:
        return None, None, "旋钮到头了，交给人：" + "；".join(lines)
    left = [f for f in fails if f not in covered]
    if left:
        lines.append("没有规则管的没过项（这次不动，留给人看）：" + "、".join(left))
    return knobs, applied, "\n".join(lines)


# ---------------- 训练 ----------------
def train_cmd(tag, init, knobs, steps, smoke):
    # --smoke 时 train_walk.py 自己用 32 个环境的小批量，不能再给 --envs=4096（批量对不上会报错）
    cmd = [PY_TRAIN, "train_walk.py"] + (["--smoke"] if smoke else [f"--tag={tag}"]) + [f"--init={init}"] + [f for f in FIXED_FLAGS if not (smoke and f.startswith("--envs"))]
    for k, v in knobs.items():
        if k not in ALLOWED_KNOBS:
            raise SystemExit(f"拒绝：旋钮 {k} 不允许")
        lo, hi = ALLOWED_KNOBS[k]
        if not lo <= v <= hi:
            raise SystemExit(f"拒绝：{k}={v} 超出硬边界 {lo}–{hi}")
        cmd.append(f"--{k}={v:g}")
    cmd.append(f"--steps={steps}")
    return cmd


def train_minutes_est(state, steps):
    """估训练要几分钟：用上一次（或作为起点的那次）的 walk_progress_<tag>.json 的速度；没有就按第 3 轮的 0.86 分钟 / 百万步。"""
    for a in reversed(state["attempts"]):
        prog = load(os.path.join(HERE, f"walk_progress_{a['tag']}.json"))
        if prog and len(prog) >= 2 and prog[-1]["steps"] > prog[0]["steps"]:
            rate = (prog[-1]["minutes"] - prog[0]["minutes"]) / ((prog[-1]["steps"] - prog[0]["steps"]) / 1e6)
            return prog[0]["minutes"] + rate * steps / 1e6 + 2
    return 6 + 0.86 * steps / 1e6


def wait_for_gpu(log, deadline):
    while True:
        others = other_training()
        if not others:
            return True
        if time.time() > deadline:
            log("别的训练一直在跑，等到预算用完也没等到：" + others[0][:160])
            return False
        log("有别的训练在跑，等它结束（不去碰它）：" + others[0][:160])
        time.sleep(300)


# ---------------- 结果 ----------------
def copies(tag):
    out = []
    for f in os.listdir(HERE):
        m = re.match(rf"walk_policy_{re.escape(tag)}_(\d+)M\.npz$", f)
        if m:
            out.append((int(m.group(1)), f[:-4]))
    return [n for _, n in sorted(out)]


def accept(att, log, dry, n_copies):
    """挑最后几份副本：先各跑 --hold 筛，选最好的一份跑全套。"""
    cs = copies(att["tag"])[-n_copies:]
    if not cs:
        raise RuntimeError(f"找不到 walk_policy_{att['tag']}_<步数>M.npz")
    mins = 0.0
    screen = {}
    for c in cs:
        mins += evaluate(c, ["hold"], dry)
        screen[c] = judge(c, only=SCREEN_IDS, screen=True)
        r = screen[c]
        log(f"  筛副本 {c}：没过 {n_fail(r)} 条；" + "；".join(f"{a['id']} {'过' if r[a['id']]['ok'] else ('没量' if r[a['id']]['ok'] is None else '没过')}"
                                                       for a in ACCEPT if a["id"] in r))

    def score(c):
        r = screen[c]
        return (n_fail(r), r.get("drift", {}).get("value") or 99, abs((r.get("speed15", {}).get("value") or 0) - 15))
    best = min(cs, key=score)
    log(f"  挑了 {best}，对它跑全套（--hold --upper、--hold --smooth、去那里三种、发热）")
    mins += evaluate(best, ["upper", "smooth", "goto", "goto_push5", "goto_push5_upper", "heat"], dry)
    att.update(copies=cs, best=best, checks=judge(best), accept_minutes=round(mins, 1))
    return att


def table_lines(res):
    out = []
    for a in ACCEPT:
        v = res.get(a["id"])
        if not v:
            continue
        out.append(f"  - {'过' if v['ok'] else ('没量' if v['ok'] is None else '**没过**')}：{a['name']}。{v['note']}")
    return out


def write_results(state, day, queue_items, stop_reason, log):
    """practice-results/<日期>.json/.md（night_practice.py 的格式）。走路那一条换成这一夜最好的那次尝试的候选策略。"""
    atts = [a for a in state["attempts"] if a.get("checks")]
    if not atts:
        return
    best = min(atts, key=lambda a: (n_fail(a["checks"]), sum(1 for v in a["checks"].values() if v["ok"] is None)))
    res = best["checks"]
    oks = [v["ok"] for v in res.values()]
    status = "还没过关" if False in oks else ("没量完" if None in oks else "练成了")
    m = re.search(r"_(\d+)M$", best["best"])
    pol = f"sim/rl/{best['best']}.npz" + (f"（{int(m.group(1)) * 100} 万步）" if m else "")
    caveat = (f"这是夜里练出来的候选（第 {best['n']} 次尝试，旋钮 {best['knobs']}），白天在用的 walk_policy.npz 没有换。"
              + ("练成了，等育苗人决定换不换。" if status == "练成了" else f"这一夜停在：{stop_reason}"))
    items = []
    for it in queue_items:
        if it.get("skill") != "走路":
            items.append(it)
            continue
        items.append(dict(text=it["text"], source=it.get("source", ""), skill="走路", status=status, policy=pol,
                          policy_detail=dict(file=pol.split("（")[0], steps=int(m.group(1)) * 1_000_000 if m else None, in_use=False,
                                             note="夜里练出来的候选，没有换上"),
                          body="第三版", mujoco=(load(walk_json(best["best"], "hold")) or {}).get("mujoco"),
                          numbers={f"{k}_{next(a['unit'] for a in ACCEPT if a['id'] == k)}": v["value"] for k, v in res.items()},
                          checks=[dict(name=a["name"], ok=res[a["id"]]["ok"], note=res[a["id"]]["note"]) for a in ACCEPT if a["id"] in res],
                          caveat=caveat, candidate=pol.split("（")[0], night_loop=dict(attempts=len(state["attempts"]), stop=stop_reason,
                                                                                         log=f"practice-results/{day}-log.md")))
    write_json(os.path.join(RESULTS, f"{day}.json"), dict(day=day, queue="digital-huaran/练习队列.md", items=items, by="sim/rl/night_loop.py"))
    md = [f"# 练习结果 {day}", "", "由 `sim/rl/night_loop.py` 生成（夜里练习的一圈，规则这一层）。数字都是仿真里的；温度是估算。"
          f"过程见 `practice-results/{day}-log.md`。", ""]
    for it in items:
        md += [f"## {it['text']}", "", f"**{it['status']}**（技能：{it.get('skill', '')}）", ""]
        if it.get("skill") == "走路":
            md += [f"- 候选策略：`{it['policy']}`；身体：第三版；mujoco {it['mujoco']}。", f"- {it['caveat']}"] + [l[2:] for l in table_lines(res)]
        md.append("")
    md += ["## 给云端的话", "", "- 白天在用的 `walk_policy.npz` 没有换；换不换由育苗人定。", ""]
    write_text(os.path.join(RESULTS, f"{day}.md"), "\n".join(md))
    log(f"写了 practice-results/{day}.json/.md：{status}；候选 {pol}")


# ---------------- 主流程 ----------------
def main():
    preflight()
    day = arg("day", datetime.date.today().isoformat())
    hours, max_att = float(arg("hours", 7)), int(arg("attempts", 3))
    steps, n_copies = int(arg("steps", 50_000_000)), int(arg("copies", 3))
    dry, smoke = flag("dry-run"), flag("smoke-train")
    t_start = time.time()
    deadline = t_start + hours * 3600
    fp = {p: sha1(p) for p in PROTECTED_FILES}
    os.makedirs(RESULTS, exist_ok=True)
    log = Log(day)

    # a. 判定：night_practice.py 看队列和白天在用的策略
    tmp = tempfile.mkdtemp(prefix="night_loop_")
    subprocess.run([PY_EVAL, "night_practice.py", f"--day={day}", f"--out={tmp}"], cwd=HERE, capture_output=True, text=True, check=True)
    qres = load(os.path.join(tmp, f"{day}.json")) or {"items": []}
    walk_items = [it for it in qres["items"] if it.get("skill") == "走路"]
    log(f"开始（预算 {hours} 小时、{max_att} 次尝试、每次 {steps / 1e6:.0f}M 步{'，dry-run' if dry else ''}{'，训练用 --smoke' if smoke else ''}）。"
        f"队列里走路的条目：" + ("、".join(f"「{it['text']}」{it['status']}" for it in walk_items) or "没有"))
    if not walk_items or all(it["status"] == "练成了" for it in walk_items):
        log("队列里没有要练的走路，结束。")
        return

    state = None if flag("fresh") else load(STATE)
    if state and state.get("day") != day:
        log(f"状态文件是 {state.get('day')} 的，不接，从头开始（旧的留在里面的 previous 里）")
        state = dict(day=day, attempts=[], previous=state)
    if arg("from"):
        if state and state.get("attempts"):
            raise SystemExit("已经有状态了，--from 只能在新的一夜用（或者加 --fresh）")
        knobs = {k: float(v) for k, v in (x.split("=") for x in arg("knobs", "").split(",") if x)}
        tag = arg("from")
        if not os.path.exists(os.path.join(HERE, f"walk_full_{tag}.npz")):
            raise SystemExit(f"找不到 walk_full_{tag}.npz")
        state = dict(day=day, attempts=[dict(n=1, tag=tag, knobs=knobs, init="（之前已经练完的）", phase="trained", rules=[], why="作为第 1 次尝试")])
    if not state or not state.get("attempts"):
        raise SystemExit("没有起点：第一次用要给 --from=<已练完的 tag> --knobs=...（这一版不会从零开一次训练）")
    write_json(STATE, state)

    stop_reason = None
    while True:
        guard(fp)
        att = state["attempts"][-1]
        if att["phase"] == "planned":
            if smoke:
                cmd = train_cmd(att["tag"], att["init"], att["knobs"], 300_000, smoke=True)
            else:
                if os.path.exists(os.path.join(HERE, f"walk_full_{att['tag']}.npz")):
                    raise SystemExit(f"拒绝：walk_full_{att['tag']}.npz 已经存在，不覆盖")
                cmd = train_cmd(att["tag"], att["init"], att["knobs"], steps, smoke=False)
            est = train_minutes_est(state, 300_000 if smoke else steps) + (0 if smoke else 70)
            left = (deadline - time.time()) / 60
            log(f"第 {att['n']} 次尝试：要跑 `{' '.join(os.path.basename(c) if i < 2 else c for i, c in enumerate(cmd))}`；估计训练 + 验收约 {est:.0f} 分钟，预算还剩 {left:.0f} 分钟")
            if est > left:
                stop_reason = f"预算不够：第 {att['n']} 次估计要 {est:.0f} 分钟，只剩 {left:.0f} 分钟，没开"
                break
            if dry:
                stop_reason = "dry-run：到这里就停（不开训练）"
                break
            if not wait_for_gpu(log, deadline):
                stop_reason = "等不到显卡"
                break
            att["phase"] = "training"
            att["started"] = datetime.datetime.now().isoformat(timespec="minutes")
            write_json(STATE, state)
            tlog = os.path.join(HERE, f"walk_{'smoke' if smoke else att['tag']}.log")
            try:
                mins = run(cmd, HERE, log_to=tlog)
            except RuntimeError as e:
                att["phase"] = "failed"
                write_json(STATE, state)
                stop_reason = f"第 {att['n']} 次训练出错，交给人看（日志 {os.path.relpath(tlog, ROOT)} 最后几行：{open(tlog).read()[-300:].strip()}）"
                log(str(e))
                break
            att.update(phase="trained", train_minutes=round(mins, 1))
            write_json(STATE, state)
            log(f"第 {att['n']} 次训练完了：{mins:.0f} 分钟（日志 sim/rl/walk_{'smoke' if smoke else att['tag']}.log）")
            guard(fp)
            if smoke:
                stop_reason = "--smoke-train：只测开训练、等它结束，不验收"
                break
        if att["phase"] == "failed":
            stop_reason = f"第 {att['n']} 次训练上次出错了，交给人看（改好以后把状态文件里这一次的 phase 改回 planned 再跑）"
            break
        if att["phase"] == "training":
            # 断在训练中间：完整存档里是最后一次评估时的，训练不能续，重新开这一次
            log(f"第 {att['n']} 次训练上次没跑完，重新开")
            for f in (f"walk_full_{att['tag']}.npz",):
                p = os.path.join(HERE, f)
                if os.path.exists(p):
                    os.rename(check_write(p), check_write(p + ".interrupted"))
            att["phase"] = "planned"
            write_json(STATE, state)
            continue
        if att["phase"] == "trained":
            log(f"第 {att['n']} 次尝试（{att['tag']}，旋钮 {att['knobs']}）：自动验收")
            try:
                accept(att, log, dry, n_copies)
            except RuntimeError as e:
                stop_reason = f"第 {att['n']} 次验收出错，交给人看：{str(e)[:400]}"
                log(stop_reason)
                break
            att["phase"] = "accepted"
            write_json(STATE, state)
            log(f"第 {att['n']} 次验收完（{att.get('accept_minutes', 0):.0f} 分钟）：{att['best']} 没过 {n_fail(att['checks'])} 条", *table_lines(att["checks"]))
            guard(fp)
        # b. 选下一次
        knobs, rules, why = decide(state)
        if knobs is None:
            stop_reason = why
            break
        if len(state["attempts"]) >= max_att:
            stop_reason = f"尝试次数用完（{max_att} 次）。下一次本来会这样改：\n{why}"
            break
        n = len(state["attempts"]) + 1
        tag = f"n{day[5:].replace('-', '')}{chr(ord('a') + n - 1)}"
        state["attempts"].append(dict(n=n, tag=tag, knobs=knobs, init=f"walk_full_{att['tag']}.npz", phase="planned", rules=rules, why=why))
        write_json(STATE, state)
        log(f"选了第 {n} 次怎么练：\n{why}")
    state["stop"] = stop_reason
    write_json(STATE, state)
    log(f"停下：{stop_reason}")
    if not dry:
        write_results(state, day, qres["items"], stop_reason, log)
    log(f"这一夜用了 {(time.time() - t_start) / 60:.0f} 分钟。没有换 walk_policy.npz。")
    guard(fp)


if __name__ == "__main__":
    main()
