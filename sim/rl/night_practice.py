"""
夜里练习（第一版）：把数字花然的"练习队列"和本地的强化学习接起来。

    python night_practice.py                 # 读队列 → 对照已有的评测结果 → 写一份"练习结果"
    python night_practice.py --queue=路径    # 换一份队列（默认 ../../digital-huaran/练习队列.md）
    python night_practice.py --day=2026-11-17
    python night_practice.py --policy=walk_policy_w3_132M   # 对别的走路策略出同样的结果，写到 practice-results/compare/（默认是白天在用的 WALK_IN_USE）
    python night_practice.py --out=/tmp/x                   # 写到别的目录（试读用）

每一条结果带 policy（文件名、步数）、body、mujoco（从评测 JSON 取）；status：全部检查 ok == true 才"练成了"，有 false 是"还没过关"，
否则有 null 是"没量完"。走路的检查照技能卡片 digital-huaran/memory/skills/走路.md 的回归测试，包括关节转速 / 膝盖扭矩
（eval_walk.py 记的最大值）和发热（sim/day_walk_compare*.json 的 continuous_walk_heat，连续走 10 分钟任何电机 ≤ 70 °C）。

这一版做什么、不做什么：
  - 只读 digital-huaran/练习队列.md，不改 digital-huaran/ 里的任何东西（那边归云端，见 docs/next-local.md）。
  - 队列里的每一条按关键词对到一个"配方"（RECIPES）：哪项技能、看哪份评测结果、过关标准、要练的话用什么命令。
  - 不自己开训练：一次训练要 1–2 小时显卡，第一版只汇报"现有的策略过没过"，没过的给出该跑的命令。
  - 结果写到 practice-results/<日期>.md 和 .json（仓库根目录下），留给云端的 day.py 早上读，或者人来读。
数字全部来自仿真（完整身体，sim/rl/eval_walk.py 和 eval_goto.py 存下的 JSON）。只用标准库，哪个 python 都能跑。
"""
import datetime
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))


def arg(name, default):
    for a in sys.argv[1:]:
        if a.startswith(f"--{name}="):
            return a.split("=", 1)[1]
    return default


def load(name):
    p = os.path.join(HERE, name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def read_queue(path):
    """练习队列里的 '- [ ] 走快一点（来自 2026-11-15 的日记）' → [(done, 文字, 来源)]"""
    out = []
    if not os.path.exists(path):
        return out
    for line in open(path, encoding="utf-8"):
        m = re.match(r"\s*-\s*\[( |x|X)\]\s*(.+?)\s*$", line)
        if m:
            text = m.group(2)
            src = re.search(r"（(.+?)）\s*$", text)
            out.append((m.group(1) != " ", re.sub(r"（.+?）\s*$", "", text).strip(), src.group(1) if src else ""))
    return out


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def policy_info(name):
    """策略的文件名和训练步数（从名字里的 '81M' 读出来）。"""
    m = re.search(r"_(\d+)M$", name)
    return dict(file=f"sim/rl/{name}.npz", steps=int(m.group(1)) * 1_000_000 if m else None,
                in_use=name == WALK_IN_USE, note="白天在用的 walk_policy.npz 是它的副本" if name == WALK_IN_USE else "不是白天在用的那份")


def mujoco_version(*jsons):
    """评测 JSON 里记的 mujoco 版本；没有就看运行这个脚本的环境里装的是哪个版本；都没有返回 None。"""
    for j in jsons:
        if j and j.get("mujoco"):
            return j["mujoco"]
    try:
        import mujoco
        return mujoco.__version__ + "（运行 night_practice.py 的环境，评测 JSON 里没记）"
    except ImportError:
        return None


HEAT_LIMIT = 70.0          # 身体地图"太热"那条线；技能卡片的回归测试：连续走 10 分钟任何电机不超过它
PUSH_EXEMPT = "被推后 1 秒内的转速已豁免：外力带动，用户 2026-11-17 决定；只记录，不算不过关"


def heat_result(name):
    """连续走 10 分钟的电机温度：从 sim/day_walk_compare*.json 的 continuous_walk_heat 取（day.py 的热模型，估算）。
    sim/day_walk_compare_<策略>.json 是用 --policy= 算的；sim/day_walk_compare.json 是用 walk_policy.npz 算的，只有它的说明里写着这份策略时才算数。"""
    sim_dir = os.path.join(HERE, "..")
    for fn in (f"day_walk_compare_{name}.json", "day_walk_compare.json"):
        p = os.path.join(sim_dir, fn)
        if not os.path.exists(p):
            continue
        d = json.load(open(p, encoding="utf-8"))
        if fn == "day_walk_compare.json" and name not in d.get("note", ""):
            continue
        rl = (d.get("continuous_walk_heat") or {}).get("rl")
        if not rl or not rl.get("joints"):
            continue
        return dict(file=f"sim/{fn}", walked_cm=rl.get("dist_cm"), fell=rl.get("fell"),
                    T_10min={j: v["T_10min"] for j, v in rl["joints"].items()},
                    T_steady={j: v["T_steady"] for j, v in rl["joints"].items()})
    return None


def walk_check(name=None):
    """对照技能卡片 digital-huaran/memory/skills/走路.md 的回归测试，看一份走路策略（默认是白天在用的 WALK_IN_USE）。"""
    name = name or WALK_IN_USE
    d = load(f"eval_walk_{name}_hold.json")
    meta = dict(policy=policy_info(name), body="第三版", mujoco=mujoco_version(d))
    if d is None:
        return dict(ok=None, lines=[f"没有找到评测结果 eval_walk_{name}_hold.json：先跑 eval_walk.py {name}.npz --hold"], numbers={}, **meta)
    runs = d["runs"]

    def tally(prefix):
        rs = [r for r in runs if r["test"].startswith(prefix)]
        return sum(1 for r in rs if r["ok"]), len(rs)

    speeds = {}                                   # 指令速度 → 实际速度（只算往前走、没摔的）
    for r in runs:
        if not r["ok"]:
            continue
        for s in r.get("segs", []):
            c = s["cmd"]
            if c[0] > 0 and c[1] == 0 and c[2] == 0 and s.get("done"):
                speeds.setdefault(round(c[0] * 100), []).append(s["vx"] * 100)
    turn = [abs(s["wz"]) for r in runs if r["ok"] and r["test"].startswith("原地转")
            for s in r.get("segs", []) if abs(s["cmd"][2]) > 0 and s.get("done")]
    n_ok, n_all = sum(1 for r in runs if r["ok"]), len(runs)
    w_ok, w_all = tally("走停")
    p_ok, p_all = tally("走+推5N")
    t_ok, t_all = tally("原地转")
    fast = max(speeds) if speeds else None
    v_fast = mean(speeds.get(fast, []))
    fell = {}
    for r in runs:
        if not r["ok"]:
            fell[r["test"]] = fell.get(r["test"], 0) + 1
    lines = [
        f"评测：{d['policy']}，完整身体，每格 {d['trials']} 次，共 {n_all} 次，没摔 {n_ok} 次"
        + ("（摔的：" + "、".join(f"{k} {v} 次" for k, v in fell.items()) + "）" if fell else "") + "。",
        "实际速度：" + "、".join(f"指令 {c} cm/s → {mean(v):.1f} cm/s" for c, v in sorted(speeds.items())) + "（手写的静步行约 1.1 cm/s）。",
    ]
    checks = [
        ("平地走 6 步不摔", w_all > 0 and w_ok == w_all, f"走了再停 {w_ok}/{w_all}（每次走的远不止 6 步）"),
        ("走路中被推 3 N 不摔", p_all > 0 and p_ok == p_all, f"走着被推 5 N {p_ok}/{p_all}（比 3 N 更重）"),
        ("原地转身 90°", t_all > 0 and t_ok == t_all and bool(turn) and mean(turn) > 0.2,
         f"原地转 {t_ok}/{t_all}" + (f"，转速 {mean(turn):.2f} rad/s，转 90° 约 {1.571 / mean(turn):.1f} 秒" if turn else "")),
        ("走到 20 cm/s 以上（技能卡片 v2 的目标）", v_fast is not None and v_fast >= 20, f"最快的指令 {fast} cm/s，实际 {v_fast:.1f} cm/s" if v_fast else "没有数"),
    ]
    numbers = dict(speeds_cm_s={str(c): round(mean(v), 1) for c, v in speeds.items()}, no_fall_runs=[n_ok, n_all])

    # 关节转速、膝盖扭矩：eval_walk.py 每次试验记的最大值（2026-11-17 起才有）。只算没摔的试验（摔倒时的数没有意义，摔的另算在上面）
    stand = [r for r in runs if r["ok"] and "qvel_max" in r]
    if not stand:
        checks.append(("关节转速不超过 10 rad/s，膝盖扭矩不超过 2.8 N·m", None,
                       f"评测结果里没有记录关节转速和膝盖扭矩：用新的 eval_walk.py 重跑 {name}.npz --hold"))
    else:
        parts, ok_sk = [], True
        numbers["max_joint_speed_rad_s"], numbers["max_knee_torque_Nm"] = {}, {}
        for k, label in (("normal", "正常走（没在被推）"), ("push", "被推开始后 1 秒内")):
            rs = [r for r in stand if k in r["qvel_max"]]
            if not rs:
                continue
            q = max(rs, key=lambda r: r["qvel_max"][k][0])
            kn = max(rs, key=lambda r: r["knee_Nm_max"][k])
            qv, kv = q["qvel_max"][k][0], kn["knee_Nm_max"][k]
            numbers["max_joint_speed_rad_s"][k] = qv
            numbers["max_knee_torque_Nm"][k] = kv
            over_q = [r for r in rs if r["qvel_max"][k][0] > 10.0]
            over_k = [r for r in rs if r["knee_Nm_max"][k] > 2.8]
            # 被推后 1 秒内的转速只记录、不算不过关（外力带动；用户 2026-11-17 决定）。膝盖扭矩照样算
            ok_sk = ok_sk and (k == "push" or not over_q) and not over_k
            s = f"{label}：转速最大 {qv} rad/s（{q['qvel_max'][k][1]}，{q['test']}），膝盖扭矩最大 {kv} N·m（{kn['test']}）"
            if over_q:
                tests = sorted({r["test"] for r in over_q})
                s += f"；转速超过 10 的有 {len(over_q)}/{len(rs)} 次（{'、'.join(tests)}），最多超 {qv - 10:.1f} rad/s"
            if k == "push":
                s += f"（{PUSH_EXEMPT}）"
                numbers["joint_speed_push_exempt"] = PUSH_EXEMPT
            if over_k:
                tests = sorted({r["test"] for r in over_k})
                s += f"；膝盖超过 2.8 的有 {len(over_k)}/{len(rs)} 次（{'、'.join(tests)}），最多超 {kv - 2.8:.2f} N·m"
            parts.append(s)
        checks.append(("关节转速不超过 10 rad/s（被推后 1 秒内只记录），膝盖扭矩不超过 2.8 N·m", ok_sk,
                       "；".join(parts) + f"。只算没摔的 {len(stand)} 次，每个小步（0.5 ms）都看。"))

    # 发热：连续走 10 分钟，任何电机不超过 70 °C
    h = heat_result(name)
    if h is None:
        checks.append((f"发热：连续走 10 分钟，任何电机不超过 {HEAT_LIMIT:.0f} °C", None,
                       f"没有找到这份策略的发热结果：先跑 cd sim && python day_walk_compare.py --policy={name}.npz"))
    else:
        hot = max(h["T_10min"].items(), key=lambda kv: kv[1])
        over = {j: t for j, t in h["T_10min"].items() if t > HEAT_LIMIT}
        numbers["heat_10min_C"] = h["T_10min"]
        checks.append((f"发热：连续走 10 分钟，任何电机不超过 {HEAT_LIMIT:.0f} °C", not over,
                       f"按 day.py 的热模型估算（{h['file']}，连续走了 {h['walked_cm']} cm 量扭矩）：最热的是{hot[0]} {hot[1]} °C"
                       + (f"；超过 {HEAT_LIMIT:.0f} °C 的：" + "、".join(f"{j} {t} °C" for j, t in over.items()) if over else "")
                       + "。只算了腿上 12 个电机（走路时上身不动）；热阻和时间常数是估算的，发热测试做完要重算"))
    g = load(f"eval_goto_{name}.json")
    if g:
        ok = sum(1 for r in g["runs"] if r["ok"])
        lines.append(f"走到指定的点（1 m 外 8 个方向）：{ok}/{len(g['runs'])}。")
        numbers["goto_ok_runs"] = [ok, len(g["runs"])]
    oks = [c[1] for c in checks]
    ok = False if False in oks else (None if None in oks else True)
    caveat = "" if name == WALK_IN_USE else f"这份不是白天在用的策略（白天在用 {WALK_IN_USE}），这里只是对比。"
    if name == "walk_policy_w3_132M":
        caveat += "第 3 轮验收时停下以后站不稳（停下后最后 2 秒还挪 1.5–2.4 cm），所以没换上（wiki/灵活度.md 第 6 条）。"
    return dict(ok=ok, lines=lines, checks=checks, numbers=numbers, caveat=caveat, **meta,
                train="cd sim/rl && .venv/bin/python train_walk.py --tag=<新名字> --upper --smooth --push=12 --effort=0.08 --steps=150000000 --envs=4096"
                      "（约 2 小时 10 分钟显卡；练完用 eval_walk.py <策略> --hold、eval_goto.py 和 ../day_walk_compare.py --policy= 验收）")


# 现在在用的走路策略（walk_policy.npz 是它的副本）。换了策略要改这里：评测结果的文件名跟着策略的名字走。
WALK_IN_USE = "walk_policy_w2_81M"

# 关键词 → 技能和检查办法。队列里出现新的说法，在这里加一行。
RECIPES = [
    (("走快", "走路", "走得", "走稳", "转身"), "走路", walk_check),
]


def status_of(ok):
    """全部检查 ok == true 才"练成了"；有 false 是"还没过关"；否则有 null 是"没量完"。"""
    return "练成了" if ok is True else ("还没过关" if ok is False else "没量完")


def main():
    day = arg("day", datetime.date.today().isoformat())
    qpath = arg("queue", os.path.join(ROOT, "digital-huaran", "练习队列.md"))
    pol = arg("policy", "").replace(".npz", "") or None          # --policy=walk_policy_w3_132M：对别的策略出同样的结果
    queue = read_queue(qpath)
    # 默认写 practice-results/<日期>.md/.json（云端 day.py 读这里最新的 .json）。
    # 对比别的策略时写到 practice-results/compare/，免得云端把对比当成当天的结果读走。--out= 可以换目录（试读用）
    out_dir = arg("out", os.path.join(ROOT, "practice-results", "compare") if pol and pol != WALK_IN_USE else os.path.join(ROOT, "practice-results"))
    stem = day if not pol or pol == WALK_IN_USE else f"{day}_{pol}"
    os.makedirs(out_dir, exist_ok=True)
    md = [f"# 练习结果 {day}" + (f"（对比：{pol}）" if pol and pol != WALK_IN_USE else ""), "",
          f"由 `sim/rl/night_practice.py` 生成。队列读自 `{os.path.relpath(qpath, ROOT)}`（只读，没有改它）。数字都是仿真里的；温度是估算。", ""]
    result = dict(day=day, queue=os.path.relpath(qpath, ROOT), items=[])
    if not queue:
        md.append("队列是空的（或者没找到文件）。")
    cache = {}
    for done, text, src in queue:
        md.append(f"## {text}" + (f"（{src}）" if src else ""))
        md.append("")
        rec = next((r for r in RECIPES if any(k in text for k in r[0])), None)
        if done:
            md += ["队列里已经勾掉了，跳过。", ""]
            result["items"].append(dict(text=text, status="已勾掉"))
            continue
        if rec is None:
            md += ["**没有对应的训练配方。** 本地现在只会练：走路、从趴着起身。要练这一条，得先有人定下\"练成什么样算过关\"。", ""]
            result["items"].append(dict(text=text, status="没有配方"))
            continue
        skill, fn = rec[1], rec[2]
        if skill not in cache:
            cache[skill] = fn(pol) if skill == "走路" else fn()
        r = cache[skill]
        status = status_of(r["ok"])
        md.append(f"**{status}**（技能：{skill}）")
        md.append("")
        if r.get("policy"):
            p = r["policy"]
            md.append(f"- 策略：`{p['file']}`" + (f"（{p['steps'] // 10000} 万步）" if p.get("steps") else "") + f"，{p['note']}；身体：{r.get('body')}；mujoco {r.get('mujoco')}。")
        md += ["- " + s for s in r["lines"]]
        for name, ok, note in r.get("checks", []):
            md.append(f"- {'没量' if ok is None else ('过' if ok else '没过')}：{name}。{note}")
        if r.get("caveat"):
            md.append(f"- 要注意：{r['caveat']}")
        if r["ok"] is not True and r.get("train"):
            md.append(f"- 要练的话：`{r['train']}`")
        md.append("")
        p = r.get("policy")
        result["items"].append(dict(text=text, source=src, skill=skill, status=status,
                                    policy=(p["file"] + (f"（{p['steps'] // 10000} 万步）" if p.get("steps") else "")) if p else None,
                                    policy_detail=p, body=r.get("body"), mujoco=r.get("mujoco"), numbers=r.get("numbers", {}),
                                    checks=[dict(name=n, ok=o, note=t) for n, o, t in r.get("checks", [])], caveat=r.get("caveat", "")))
    md += ["## 给云端的话", "",
           "- 这份结果没有写进 `digital-huaran/`。要不要勾掉队列、要不要更新技能卡片（`走路.md` 的 v2 现在还写着\"待训练\"），由云端和育苗人定。",
           "- \"练成了\"只说明过了技能卡片上的回归测试，不等于可以当默认用：看每一条下面的\"要注意\"。", ""]
    open(os.path.join(out_dir, f"{stem}.md"), "w", encoding="utf-8", newline="\n").write("\n".join(md))
    json.dump(result, open(os.path.join(out_dir, f"{stem}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main()
