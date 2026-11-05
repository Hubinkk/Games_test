"""
灵巧手 v2 · 打字速度测试

用法：
    python hand2_speed.py

做法：
  - 打一句完整的话 "hello how are you"（含空格）
  - 比保守版更快的动作安排：悬停高度降到 1.5 mm；按完直接移向下一个键（抬起和移动合并）；
    不再等手完全停稳
  - 逐档提高速度，统计：按对率、误触（移动途中指尖擦到别的键，在触摸屏上就是打错字）、
    指尖定位误差
  - 选出不出错的最高速度，导出为网页回放里的"快速打字"场景
"""
import math

import mujoco
import numpy as np

import hand2_build as hb
import hand2_demo as hd

TEXT = "hello how are you"
HOVER = 0.0015
SPEEDS = [  # (每键移动时间, 按下时间) 秒
    (0.45, 0.18), (0.30, 0.12), (0.22, 0.10), (0.16, 0.08), (0.12, 0.07), (0.09, 0.06), (0.07, 0.05)]


def type_fast(h, text, t_move, t_press, label="快速打字"):
    h.reset(); h.pen_on(False)
    hover = dict(index_mcp=0.35, index_pip=0.5, middle_mcp=1.45, middle_pip=1.5, ring_mcp=1.45, ring_pip=1.5,
                 little_mcp=1.45, little_pip=1.5, thumb_opp=0.3, thumb_cmc=-0.2, thumb_mcp=0.3)
    h.set_q(**hover)
    base = np.array(hb.ARM["pos"])
    d = mujoco.MjData(h.m)
    press_pose = {**h.q, "index_mcp": 0.55, "index_pip": 0.75, "index_dip": 0.75 * 0.75}
    for j, v in press_pose.items():
        d.qpos[h.qadr[j]] = v
    mujoco.mj_kinematics(h.m, d)
    R = 0.0019
    pad0 = d.site_xpos[h.m.site("index_tip").id] - base - np.array([0, 0, R])
    top = 0.0005
    joints = ["wrist_yaw", "index_abd", "index_mcp", "index_pip"]

    touched = set()

    def watch():
        for i in range(h.d.ncon):
            c = h.d.contact[i]
            n1, n2 = h.m.geom(c.geom1).name, h.m.geom(c.geom2).name
            key = next((n for n in (n1, n2) if n.startswith("key_")), None)
            fing = next((n for n in (n1, n2) if not n.startswith("key_") and n != "phone"), None)
            if key and fing:
                touched.add((key[4:], fing))

    h.on_step = watch
    results, mistouch = [], []
    # 先移到第一个键上方（不计入打字时间）
    first = True
    prev = None
    t0 = None
    for ch in text:
        name = hb.key_name(ch)
        kx, ky = h.keys[ch]
        gy = (round((ky + hb.KEY_PITCH / 2) / (2 * hb.KEY_PITCH)) * 2 * hb.KEY_PITCH) - hb.KEY_PITCH / 2
        if ch == " ":
            gy = ky
        ax, ay = kx - pad0[0], gy - pad0[1]
        az = top - 0.0006 - pad0[2] - base[2]
        qh, _ = h.ik("index_tip", [kx, ky, top + HOVER + R], joints,
                     q0={**h.q, "arm_x": ax, "arm_y": ay, "arm_z": az})
        h.set_q(arm_x=ax, arm_y=ay, arm_z=az, **{k: qh[k] for k in joints})
        touched.clear()
        if first:  # 起手：先在键盘上方 6 mm 处移到位，再垂直落下，避免一路擦过别的键
            h.set_q(arm_z=az + 0.006)
            h.run(0.6, label)
            h.set_q(arm_z=az)
            h.run(0.3, label)
            touched.clear()
        else:
            h.run(t_move, label)
        if first:
            t0 = h.d.time
            first = False
        # 移动途中碰到的、既不是上一个键也不是目标键的，算误触
        for key, fing in touched:
            if key not in (name, prev):
                mistouch.append((ch, key, fing))
        touched.clear()
        qp, _ = h.ik("index_tip", [kx, ky, top - 0.0006 + R], joints, q0=h.q)
        h.set_q(**{k: qp[k] for k in joints})
        peak = None
        start = h.d.ctrl.copy(); h.apply(); goal = h.d.ctrl.copy()
        steps = max(1, int(t_press / h.m.opt.timestep)); every = int(1 / hd.FPS / h.m.opt.timestep)
        for k in range(steps):
            a = min(1.0, (k + 1) / (0.5 * steps))
            h.d.ctrl[:] = start + (goal - start) * a
            h.step()
            kc = h.key_contact()
            if kc and (peak is None or kc[2] > peak[2]):
                peak = kc
            if k % every == 0:
                h.record(label)
        for key, fing in touched:
            if key != name and not fing.startswith("index_g3"):
                mistouch.append((ch, key, fing))
        hit, pos, force = peak if peak else (None, None, 0.0)
        err = float(np.hypot(pos[0] - kx, pos[1] - ky) * 1000) if pos is not None else None
        results.append({"key": name, "hit": hit, "error_mm": err, "force_N": force,
                        "abd_deg": math.degrees(qp["index_abd"]), "yaw_deg": math.degrees(qp["wrist_yaw"])})
        h.events.append({"t": h.frames[-1]["t"] if h.frames else 0, "type": "key", "key": name, "hit": hit,
                         "error_mm": None if err is None else round(err, 2), "force_N": round(force, 2)})
        prev = name
    elapsed = h.d.time - t0
    h.set_q(**{k: qh[k] for k in joints})
    h.run(0.3, label)
    h.on_step = None
    hits = sum(r["hit"] == r["key"] for r in results)
    errs = [r["error_mm"] for r in results if r["error_mm"] is not None]
    return {"results": results, "hits": hits, "n": len(text), "mistouch": mistouch, "elapsed": elapsed,
            "cps": len(text) / elapsed, "wpm": len(text) / 5 / elapsed * 60,
            "err_mean": float(np.mean(errs)) if errs else None, "err_max": float(max(errs)) if errs else None}


def main():
    h = hd.Hand2()
    h.recording = False
    print(f'文字："{TEXT}"（{len(TEXT)} 个字符）\n')
    print(f"  {'每键用时':<10}{'速度':<22}{'按对':<8}{'误触':<6}{'平均误差':<10}{'最大误差'}")
    best = None
    rows = []
    for tm, tp in SPEEDS:
        r = type_fast(h, TEXT, tm, tp)
        ok = r["hits"] == r["n"] and not r["mistouch"]
        rows.append((tm, tp, r))
        print(f"  {tm + tp:.2f} s{'':<5}{r['cps']:.1f} 字符/秒（{r['wpm']:.0f} 词/分）{'':<2}"
              f"{r['hits']}/{r['n']}{'':<4}{len(r['mistouch']):<6}"
              f"{(r['err_mean'] or 0):.2f} mm{'':<3}{(r['err_max'] or 0):.2f} mm{'  ✓' if ok else ''}")
        if r["mistouch"]:
            print("      误触：" + "，".join(f"打 {c!r} 时 {f} 碰到 {k}" for c, k, f in r["mistouch"][:4]))
        if ok:
            best = (tm, tp, r)
    if best is None:
        print("\n没有找到完全不出错的速度档。")
        return
    tm, tp, r = best
    print(f"\n不出错的最高速度：每键 {tm + tp:.2f} 秒，{r['cps']:.1f} 字符/秒，约 {r['wpm']:.0f} 词/分")

    # 用这一档重新录制，和其他场景一起导出给网页
    def scene_fast(hh):
        hh.recording = True
        out = type_fast(hh, TEXT, tm, tp)
        out["speeds"] = [{"t_key": a + b, "cps": rr["cps"], "wpm": rr["wpm"], "hits": rr["hits"], "n": rr["n"],
                          "mistouch": len(rr["mistouch"]), "err_mean": rr["err_mean"], "err_max": rr["err_max"]}
                         for a, b, rr in rows]
        out["mistouch"] = [list(x) for x in out["mistouch"]]
        return out

    print("\n重新导出全部回放场景：")
    hd.main(extra=[("fast", scene_fast)])


if __name__ == "__main__":
    main()
