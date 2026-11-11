"""
同一天（默认数字花然第 5 天，2026-11-14）分别用手写的起身（--getup=hands）和强化学习的策略（--getup=rl）各过一次，对比起身。

用法：
    python full_build.py
    python day_getup_compare.py [--day=2026-11-14]      # 写 day_getup_compare.json

两次都用 day.py 的 --out= 写到临时目录（从现在的 digital-huaran/ 复制一份记忆接着过），不动网页数据。
对比的是：站没站起来（骨盆高 > 20 cm 且倾斜 < 20°，和 rl/eval_full.py 相同）、起身用时、起身时关节温度和发热、后面的场景是否照常走完。
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
DAY = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--day=")), "2026-11-14")


def load_js(p):
    s = open(p, encoding="utf-8").read()
    return json.loads(s[s.index("=") + 1:].rstrip().rstrip(";"))


def one(tag, base):
    out = os.path.join(base, tag)
    shutil.rmtree(out, ignore_errors=True)
    r = subprocess.run([sys.executable, "day.py", f"--day={DAY}", f"--getup={tag}", f"--out={out}"], cwd=HERE,
                       capture_output=True, text=True)
    if r.returncode != 0:
        return dict(method=tag, crashed=True, stderr=r.stderr[-2000:])
    data = load_js(os.path.join(out, "digital_huaran_day.js"))
    g = data["results"]["getup"]
    keys = [s["key"] for s in data["spans"]]
    after = keys[keys.index("getup") + 1:]
    falls_after = [e for e in data["events"] if e["kind"] == "摔倒" and e["scene"] in after]
    hot_after = [e for e in data["events"] if e["kind"] in ("热", "保护") and e["scene"] in after]
    rep = os.path.join(out, "reports")
    return dict(method=tag, getup=g,
                scenes=keys, scenes_after_getup=after, falls_after_getup=len(falls_after),
                heat_events_after_getup=[f"{e['clock']} {e['text']}" for e in hot_after],
                getup_events=[f"{e['clock']} {e['kind']} {e['text']}" for e in data["events"] if e["scene"] == "getup"],
                walk=data["results"]["walk"], push=[data["results"]["push1"]["ok"], data["results"]["push2"]["ok"]],
                battery_end=data["frames"][-1]["bat"], frames=len(data["frames"]),
                report_written=sorted(os.listdir(rep)) if os.path.isdir(rep) else [],
                stdout_tail=r.stdout.splitlines()[-3:])


def main():
    base = os.path.join(tempfile.gettempdir(), "huaran_day_compare")
    os.makedirs(base, exist_ok=True)
    res = {"day": DAY, "note": "仿真（mujoco 3.17，默认地面摩擦 1.0，整体模型）", "runs": [one(t, base) for t in ("hands", "rl")]}
    ref = os.path.join(ROOT, "digital_huaran_day.js")
    if os.path.exists(ref):
        d = load_js(ref)
        if d.get("day") == DAY:
            res["committed_hands"] = {k: d["results"]["getup"].get(k) for k in ("ok", "peak_cm")}
    json.dump(res, open(os.path.join(HERE, "day_getup_compare.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for r in res["runs"]:
        if r.get("crashed"):
            print(r["method"], "运行出错：", r["stderr"])
            continue
        g = r["getup"]
        print(f"{r['method']:>5}：{'站起来了' if g['ok'] else '没站起来'}，结束时骨盆 {g['z_cm']} cm、倾斜 {g['tilt']}°；"
              f"第一次站着 {g['first_up_s']} s，从 {g['stood_s']} s 起一直站着（场景共 {g['sim_s']} s）；"
              f"发热合计 {g['heat_total_J']} J，最多的 {g['heat_J']}；温度（开始→最高）{g['peak_T']}；"
              f"后面场景 {r['scenes_after_getup']}，后面摔倒 {r['falls_after_getup']} 次；傍晚电量 {r['battery_end']}%")


if __name__ == "__main__":
    main()
