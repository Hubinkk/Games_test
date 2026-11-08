"""
触手舌头"像不像人"：声音像不像、形状像不像，以及能不能两者兼得。

用法：
    python tongue_like.py      # 生成 tongue_like.json（数据）和 tongue_audio.json（声音，base64 WAV）

"像"的两个尺度：
  1. 听起来像：共振峰误差；以及按最近的元音去听，会不会听成别的元音（5 个元音里听对几个）。
  2. 形状像：声道最窄处（收紧点）离声门多远，与人的典型位置相差多少。
     人的参考值取经典的元音发音测量（成年男性，声门到嘴唇约 17 cm）的大致位置，量级 ±1 cm：
       a 约 5 cm（咽部）、o 约 7.5 cm（咽上部 / 小舌）、u 约 10.5 cm（软腭）、e 约 11.5 cm、i 约 12.5 cm（硬腭）
三种舌头比较：
  - 人：voice.py 的发音器官模型（已校验）
  - 触手 · 只求声音：两段 + 舌根两轴滑台 + 双气腔（舌体前部充气、舌根放气），只优化共振峰
  - 触手 · 声音和形状都求：同一结构，代价里加上收紧点位置
"""
import base64
import io
import json
import math
import os
import wave

import numpy as np

import tongue as T
import voice as V

HERE = os.path.dirname(os.path.abspath(__file__))
REF_LOC = {"a": 5.0, "o": 7.5, "u": 10.5, "e": 11.5, "i": 12.5}
LIKE_TOL = 1.5          # 收紧点相差 1.5 cm 以内算"形状像"
KEYS = T.CONFIGS["两段+升降+双气腔"]
SPEAK_SCALE = 14.5 / 17  # 试听时统一用成年女性的声道长度
SR = 24000


def loc_tentacle(p):
    a, seg, _ = T.area_fn(p)
    pos = np.cumsum(seg) - seg / 2
    i = 2 + int(np.argmin(a[2:-3]))
    return float(pos[i]), float(a[i])


def loc_human(p):
    a = V.area(p)
    n = len(a)
    _, _, lip_n, _ = V.geometry(1.0, p["round"])
    lo = int(n * 0.12)
    i = lo + int(np.argmin(a[lo:n - lip_n]))
    return (i + 0.5) / n * V.geometry(1.0, p["round"])[0], float(a[i])


def heard_as(f):
    """按 log(F1)、log(F2) 的最近距离，听成哪个元音。"""
    best, bd = None, 1e9
    for v, (t1, t2) in V.TARGETS.items():
        d = math.hypot(math.log(f[0] / t1), math.log(f[1] / t2))
        if d < bd:
            best, bd = v, d
    return best


def like_cost(v, w):
    base = T.vowel_cost(V.TARGETS[v])
    def c(p):
        e = base(p)
        if w:
            loc, _ = loc_tentacle(p)
            e += w * max(0.0, abs(loc - REF_LOC[v]) - 0.5)
        return e
    return c


# ---------------- 发声（时域波导，面积序列逐块更新） ----------------
def resample(a, seg, scale, n):
    edges = np.r_[0, np.cumsum(seg)] / seg.sum()
    x = (np.arange(n) + 0.5) / n
    idx = np.clip(np.searchsorted(edges, x) - 1, 0, len(a) - 1)
    return np.maximum(a[idx] * scale ** 2, V.A_MIN)


def speak(area_seq, durs, f0=210.0, scale=SPEAK_SCALE, seed=3):
    """area_seq：每个音的 (面积, 段长)；逐个线性过渡（60 ms），共用一个声门声源。"""
    rng = np.random.default_rng(seed)
    n = int(round(17.0 * scale * V.FS / V.C))
    shapes = [resample(a, s, scale, n) for a, s in area_seq]
    nose = V.nose_area(scale, 0.0)
    t = V.Tract(shapes[0][None, :], nose[None, :], int(round(n * 0.52)))
    out, ph = [], 0.0
    total = sum(durs)
    tt = 0.0
    prev = shapes[0]
    for k, (shape, dur) in enumerate(zip(shapes, durs)):
        steps = int(dur * V.FS)
        for s0 in range(0, steps, 48):
            w = min(1.0, (s0 + 48) / (V.FS * 0.06))
            cur = prev + (shape - prev) * w
            nt = V.Tract(cur[None, :], nose[None, :], t.v)
            nt.R, nt.L, nt.NR, nt.NL = t.R, t.L, t.NR, t.NL
            t = nt
            for _ in range(48):
                tt += 1 / V.FS
                u = tt / total
                fq = f0 * (1.06 - 0.16 * u) * (1 + 0.003 * rng.normal())
                ph = (ph + fq / V.FS) % 1.0
                amp = min(1.0, tt / 0.03, (total - tt) / 0.05)
                g = V.glottal(ph, 0.6, 0.5) * 0.02 * max(amp, 0) + rng.normal() * 0.0006 * max(amp, 0)
                out.append(t.step(np.array([g]))[0])
        prev = shape
    y = np.array(out)
    y = 0.5 * (y[0::2][: len(y) // 2] + y[1::2][: len(y) // 2])          # 48 kHz
    for _ in range(2):
        y = np.convolve(y, [0.25, 0.5, 0.25], mode="same")
    y = y[::2]                                                            # 24 kHz
    return y / (np.max(np.abs(y)) + 1e-9) * 0.85


def wav_b64(y):
    buf = io.BytesIO()
    w = wave.open(buf, "wb")
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((np.clip(y, -1, 1) * 32767).astype("<i2").tobytes())
    w.close()
    return base64.b64encode(buf.getvalue()).decode()


def human_areas(p):
    a = V.area(p)
    return a, np.full(len(a), V.C / V.FS)


if __name__ == "__main__":
    import time
    t0 = time.time()
    T._calibrate()
    HUMAN = {  # voice.py 反推的人的姿势
        "a": dict(V.NEUTRAL, front=.017, height=0, jaw=.915, lips=.852),
        "i": dict(V.NEUTRAL, front=.938, height=.94, jaw=.242, lips=.558),
        "u": dict(V.NEUTRAL, front=.088, height=.967, jaw=.057, lips=.368, round=.3),
        "e": dict(V.NEUTRAL, front=.92, height=.521, jaw=.481, lips=.876),
        "o": dict(V.NEUTRAL, front=0, height=.239, jaw=.672, lips=.65, round=.7),
    }
    data = {"ref_loc": REF_LOC, "tol": LIKE_TOL, "sets": {}}
    sols = {"human": HUMAN}
    for name, w in (("sound", 0.0), ("both", 0.25)):
        sols[name] = {}
        for v in "aiueo":
            p, c = T.search(like_cost(v, w), KEYS, dict(T.REST, **T.FIXED[v]), 4, iters=10, seed=4)
            sols[name][v] = p
            print(f"{name} {v}: {T.formants_fast(p)[:2]} 收紧点 {loc_tentacle(p)[0]:.1f} cm", flush=True)
    for name in ("human", "sound", "both"):
        rows = {}
        for v in "aiueo":
            p = sols[name][v]
            if name == "human":
                f = list(V.formants([p])[0]); loc, amin = loc_human(p)
            else:
                f = T.formants_fast(p); loc, amin = loc_tentacle(p)
            tgt = V.TARGETS[v]
            err = max(abs(f[0] / tgt[0] - 1), abs(f[1] / tgt[1] - 1))
            rows[v] = dict(f=[float(f[0]), float(f[1])], err=float(err), heard=heard_as(f),
                           loc=round(loc, 2), dloc=round(loc - REF_LOC[v], 2), amin=round(amin, 2),
                           p={k: round(float(x), 3) for k, x in p.items()})
            print(f"  [{name}] {v}: F {f[0]:.0f}/{f[1]:.0f} 误差 {err*100:.0f}%  听成 {rows[v]['heard']}  收紧点 {loc:.1f} cm（人 {REF_LOC[v]}，差 {loc-REF_LOC[v]:+.1f}）")
        data["sets"][name] = rows
    # 声音
    audio = {}
    for name in ("human", "sound", "both"):
        get = human_areas if name == "human" else (lambda p: T.area_fn(p)[:2])
        for v in "aiueo":
            audio[f"{name}_{v}"] = wav_b64(speak([get(sols[name][v])], [0.55]))
        audio[f"{name}_aiueo"] = wav_b64(speak([get(sols[name][v]) for v in "aiueo"], [0.26, 0.26, 0.26, 0.26, 0.36]))
        print("audio", name, flush=True)
    json.dump(data, open(os.path.join(HERE, "tongue_like.json"), "w"), ensure_ascii=False, indent=1)
    json.dump(audio, open(os.path.join(HERE, "tongue_audio.json"), "w"))
    print(f"用时 {time.time()-t0:.0f} s")
