"""
同一天（默认数字花然第 5 天，2026-11-14，起身用 --getup=rl）分别用手写的静步行和学会的走路（--walk=rl）各过一次，对比"走到窗边"那一段。

用法：
    python day_walk_compare.py [--day=2026-11-14]      # 写 day_walk_compare.json

两次都用 day.py 的 --out= 写到临时目录（从现在的 digital-huaran/ 复制一份记忆接着过），不动网页数据，不带 --main。
对比：摔没摔、走这一段用时、走了多远、走完的朝向偏差、腿上各电机走路期间的最高温度和这一天的最高温度、后面的场景是否照常。

关于温度（要看清楚记法）：day.py 里每个场景代表一段真实时长，"走到窗边"按 10 分钟算，热量按"真实时长 / 仿真时长"的倍数累计。
静步行这一段仿真 12 秒、全是在走，所以等于"静步行走 10 分钟"；学会的走路只走 0.8 秒（11 cm）就到了，后面 5 秒是站着和交回，
所以这一段的温度大部分是"站着 10 分钟"。为了知道"一直这样走会多热"，这里另外在新的身体上连续走 10 秒，
量腿上各电机扭矩平方的平均，按 day.py 同一个热模型算"从 25 °C 开始连续走 10 分钟"的温度和长期走下去的稳态温度。
"""
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
DAY = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--day=")), "2026-11-14")
LEGS = [f"{s}_{j}" for s in "lr" for j in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch", "ankle_roll")]


def load_js(p):
    s = open(p, encoding="utf-8").read()
    return json.loads(s[s.index("=") + 1:].rstrip().rstrip(";"))


def one(tag, base):
    out = os.path.join(base, tag)
    shutil.rmtree(out, ignore_errors=True)
    args = [sys.executable, "day.py", f"--day={DAY}", "--getup=rl", f"--out={out}"] + (["--walk=rl"] if tag == "rl" else [])
    r = subprocess.run(args, cwd=HERE, capture_output=True, text=True)
    if r.returncode != 0:
        return dict(method=tag, crashed=True, stderr=r.stderr[-3000:])
    data = load_js(os.path.join(out, "digital_huaran_day.js"))
    w = data["results"]["walk"]
    keys = [s["key"] for s in data["spans"]]
    after = keys[keys.index("walk") + 1:]
    # 这一天每个关节的最高温度：从回放帧里取（帧里只记了 SHOW 那 12 个关节，腿上的髋、膝、踝俯仰在里面）
    show = data["show"]
    day_max = {n: max(f["T"][i] for f in data["frames"]) for i, n in enumerate(show)}
    walk_frames = [f for f in data["frames"] if f["scene"] == "walk"]
    walk_max = {n: max(f["T"][i] for f in walk_frames) for i, n in enumerate(show)} if walk_frames else {}
    ev = data["events"]
    return dict(method=tag, walk=w,
                walk_events=[f"{e['clock']} {e['kind']} {e['text']}" for e in ev if e["scene"] == "walk"],
                falls=[f"{e['clock']} {e['scene']}" for e in ev if e["kind"] == "摔倒"],
                protect=[f"{e['clock']} {e['scene']} {e['text']}" for e in ev if e["kind"] == "保护"],
                hot_events=[f"{e['clock']} {e['scene']} {e['text']}" for e in ev if e["kind"] == "热"],
                scenes_after_walk=after, results_after={k: data["results"].get(k) for k in ("push1", "push2")},
                getup_ok=data["results"]["getup"]["ok"], show_T_walk_max=walk_max, show_T_day_max=day_max,
                battery_end=data["frames"][-1]["bat"],
                report=sorted(os.listdir(os.path.join(out, "reports"))), stdout_tail=r.stdout.splitlines()[-2:])


def continuous_heat():
    """在新的身体上连续走，量腿上电机的扭矩平方平均，按 day.py 的热模型算温度。"""
    import day
    import getup_hands as G
    import walk
    import walk_rl
    from balance import Balance
    sim = G.get_sim()
    B = day.Body(sim)
    out = {}
    # 学会的走路：15 cm/s 连续 10 秒（前 1 秒起步不算）
    t2, dx, fell = walk_rl.mean_tau2(sim, LEGS, 10.0)
    out["rl"] = dict(desc="学会的走路 15 cm/s，连续 10 秒（去掉前 1 秒）", dist_cm=round(dx * 100, 1), fell=fell, tau2=t2)
    # 静步行：和一天里一样的 4 步、12 秒，全程都算
    acc, n = {j: 0.0 for j in LEGS}, [0]

    def rec():
        tq = sim.torques()
        for j in LEGS:
            acc[j] += float(tq.get(j, 0.0)) ** 2
        n[0] += 1
    sim.recorder = rec
    fell2, dx2, _, _ = walk.walk(sim, Balance(sim), 4, 0.03, T=2.4)
    sim.recorder = None
    out["static"] = dict(desc="手写静步行 4 步 12 秒", dist_cm=round(dx2 * 100, 1), fell=fell2, tau2={j: acc[j] / max(1, n[0]) for j in LEGS})
    for v in out.values():
        rows = {}
        for j, s in v["tau2"].items():
            if j not in B.k:
                continue
            P = B.k[j] * s
            rise = B.R[j] * P
            rows[day.NAMES.get(j, j)] = dict(rms_Nm=round(math.sqrt(s), 3), P_W=round(P, 3),
                                             T_10min=round(day.T_AMB + rise * (1 - math.exp(-600.0 / B.tau[j])), 1),
                                             T_steady=round(day.T_AMB + rise, 1))
        v["joints"] = rows
        v.pop("tau2")
    return out


def main():
    base = os.path.join(tempfile.gettempdir(), "huaran_walk_compare")
    os.makedirs(base, exist_ok=True)
    res = {"day": DAY, "note": "仿真（mujoco 3.17，地面摩擦 1.0，整体模型）；两次起身都用 --getup=rl",
           "thresholds": {"有点热": 55, "太热": 70, "过热保护（自动减力）": 80},
           "runs": [one(t, base) for t in ("static", "rl")]}
    res["continuous_walk_heat"] = continuous_heat()
    json.dump(res, open(os.path.join(HERE, "day_walk_compare.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for r in res["runs"]:
        if r.get("crashed"):
            print(r["method"], "运行出错：", r["stderr"])
            continue
        w = r["walk"]
        print(f"{r['method']:>6}：走 {w['dx_cm']:.1f} cm、用 {w['walk_s']} s（这一段仿真共 {w['sim_s']} s），{w['v_cm_s']:.1f} cm/s；"
              f"摔 {w['fell']}；朝向 {w['heading_deg']}°，横向 {w['dy_cm']} cm；腿温（开始→这一段最高）{w['leg_T']}；"
              f"这一天最高 {r['show_T_day_max']}；后面 {r['scenes_after_walk']} 起身 {r['getup_ok']}；保护 {r['protect']}")
    for k, v in res["continuous_walk_heat"].items():
        print(k, v["desc"], v["dist_cm"], "cm", "摔", v["fell"])
        for j, row in v["joints"].items():
            print("   ", j, row)


if __name__ == "__main__":
    main()
