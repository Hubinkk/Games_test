"""
夜里练习（第一版）：把数字花然的"练习队列"和本地的强化学习接起来。

    python night_practice.py                 # 读队列 → 对照已有的评测结果 → 写一份"练习结果"
    python night_practice.py --queue=路径    # 换一份队列（默认 ../../digital-huaran/练习队列.md）
    python night_practice.py --day=2026-11-17

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


def walk_check():
    """对照 robot-memory/skills/走路.md 的回归测试和"目标 20 cm/s 以上"，看现在在用的走路策略（walk_policy.npz）。"""
    d = load(f"eval_walk_{WALK_IN_USE}_hold.json")
    if d is None:
        return dict(ok=None, lines=[f"没有找到评测结果 eval_walk_{WALK_IN_USE}_hold.json：先跑 eval_walk.py {WALK_IN_USE}.npz --hold"], numbers={})
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
        ("关节转速不超过 10 rad/s，膝盖扭矩不超过 2.8 N·m", None,
         "目标角每秒最多变 10 rad、电机扭矩上限沿用模型里的，这两条是写死的；但实际关节转速评测里没有记录，所以算\"没量\""),
    ]
    g = load(f"eval_goto_{WALK_IN_USE}.json")
    if g:
        ok = sum(1 for r in g["runs"] if r["ok"])
        lines.append(f"走到指定的点（1 m 外 8 个方向）：{ok}/{len(g['runs'])}。")
    ok = all(c[1] for c in checks if c[1] is not None)
    return dict(ok=ok, lines=lines, checks=checks,
                numbers=dict(speeds={str(c): round(mean(v), 1) for c, v in speeds.items()}, no_fall=[n_ok, n_all]),
                caveat="发热没算在这里：这份策略连续走 10 分钟，脚踝侧摆估算会超过 80 °C（sim/day_walk_compare.json，wiki/灵活度.md）。"
                       "所以 day.py 默认还没换成它。",
                train="cd sim/rl && .venv/bin/python train_walk.py --tag=<新名字> --upper --smooth --push=12 --effort=0.08 --steps=150000000 --envs=4096"
                      "（约 2 小时 10 分钟显卡；练完用 eval_walk.py <策略> --hold 和 eval_goto.py 验收）")


# 现在在用的走路策略（walk_policy.npz 是它的副本）。换了策略要改这里：评测结果的文件名跟着策略的名字走。
WALK_IN_USE = "walk_policy_w2_81M"

# 关键词 → 技能和检查办法。队列里出现新的说法，在这里加一行。
RECIPES = [
    (("走快", "走路", "走得", "走稳", "转身"), "走路", walk_check),
]


def main():
    day = arg("day", datetime.date.today().isoformat())
    qpath = arg("queue", os.path.join(ROOT, "digital-huaran", "练习队列.md"))
    queue = read_queue(qpath)
    out_dir = os.path.join(ROOT, "practice-results")
    os.makedirs(out_dir, exist_ok=True)
    md = [f"# 练习结果 {day}", "",
          f"由 `sim/rl/night_practice.py` 生成。队列读自 `{os.path.relpath(qpath, ROOT)}`（只读，没有改它）。数字都是仿真里的。", ""]
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
        r = cache.setdefault(skill, fn())
        status = "没有评测结果" if r["ok"] is None else ("练成了" if r["ok"] else "还没过关")
        md.append(f"**{status}**（技能：{skill}）")
        md.append("")
        md += ["- " + s for s in r["lines"]]
        for name, ok, note in r.get("checks", []):
            md.append(f"- {'没量' if ok is None else ('过' if ok else '没过')}：{name}。{note}")
        if r.get("caveat"):
            md.append(f"- 要注意：{r['caveat']}")
        if not r["ok"] and r.get("train"):
            md.append(f"- 要练的话：`{r['train']}`")
        md.append("")
        result["items"].append(dict(text=text, source=src, skill=skill, status=status, numbers=r.get("numbers", {}),
                                    checks=[dict(name=n, ok=o, note=t) for n, o, t in r.get("checks", [])], caveat=r.get("caveat", "")))
    md += ["## 给云端的话", "",
           "- 这份结果没有写进 `digital-huaran/`。要不要勾掉队列、要不要更新技能卡片（`走路.md` 的 v2 现在还写着\"待训练\"），由云端和育苗人定。",
           "- \"练成了\"只说明过了技能卡片上的回归测试，不等于可以当默认用：看每一条下面的\"要注意\"。", ""]
    open(os.path.join(out_dir, f"{day}.md"), "w", encoding="utf-8", newline="\n").write("\n".join(md))
    json.dump(result, open(os.path.join(out_dir, f"{day}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main()
