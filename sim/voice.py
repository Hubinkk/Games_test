"""
声道发声模型（发音器官合成）：声音由"身体的形状"产生，而不是拼接录音

用法：
    python voice.py            # 校验元音共振峰、反推发音姿势、比较不同声道长度

模型：
  - 声道 = 一串很短的圆管（Kelly-Lochbaum 波导）。每段圆管有一个截面积，
    声波在相邻两段之间按面积之比反射和透射。软腭处分出一条鼻腔支路（三通接头）。
  - 发音器官（舌位前后、舌位高低、下巴、舌尖、嘴唇开合、圆唇、软腭）决定每段的截面积。
  - 声源：声门脉冲（开放比、紧张度控制音色）＋ 气声噪声；摩擦音在最窄处注入湍流噪声。
  - 时间步：每个采样点，声波在每个方向上正好走过一段圆管，
    所以段数 N = 声道长度 × 内部采样率 / 声速。

网页版 Voice.html 用的是同一套面积函数和同一个波导，这里是它的校验和数据来源。
"""
import math

import numpy as np

C = 35000.0          # 声速，cm/s
FS = 96000           # 内部采样率（网页用 2 × 声卡采样率）
L_ADULT = 17.0       # 成年男性声道长度，cm（其他尺寸按比例缩放）
GLOTTAL_REFL = 0.75
LIP_REFL = -0.85
NOSE_REFL = -0.85
FADE = 0.999         # 每段的传播损耗
A_MIN = 1e-3         # 完全闭合时的最小面积，cm²（避免除零）

PARAMS = ("front", "height", "jaw", "tip", "lips", "round", "velum")
NEUTRAL = dict(front=0.5, height=0.35, jaw=0.35, tip=0.0, lips=0.6, round=0.0, velum=0.0)

# 日语成年男性元音的典型共振峰（Hz），文献数值相差约 ±15%
TARGETS = {
    "a": (750, 1200),
    "i": (280, 2250),
    "u": (320, 1350),
    "e": (480, 1900),
    "o": (480, 850),
}


def lerp(a, b, t):
    return a + (b - a) * t


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def geometry(scale, rnd):
    """声道总长（cm）、段长、段数、嘴唇段数、软腭位置。圆唇会把嘴唇往前伸，声道变长。"""
    length = (L_ADULT + 1.0 * rnd) * scale
    n = max(8, int(round(length * FS / C)))
    lip_n = max(1, int(round(1.0 * scale * (1 + rnd) * FS / C)))
    vel = int(round(n * 0.52 / (1 + 0.06 * rnd)))
    return length, n, lip_n, vel


def area(p, scale=1.0):
    """发音参数（0–1）→ 声道各段截面积（cm²，从声门到嘴唇）。"""
    _, n, lip_n, _ = geometry(scale, p["round"])
    x = (np.arange(n) + 0.5) / n
    f, h, jaw = p["front"], p["height"], p["jaw"]
    # 中性声道：声门上方一小段较细（喉前庭），其余约 3 cm²
    a0 = 1.2 + 1.8 * smoothstep(0.05, 0.12, x)
    # 舌体：在某处形成收窄。舌后低 → 咽部收窄（a）；舌后高 → 软腭处（u）；舌前高 → 硬腭处（i）
    hh = np.clip(h - 0.25 * (jaw - 0.35), 0, 1)        # 下巴张开会把舌头一起带低
    xc = lerp(lerp(0.30, 0.60, hh), lerp(0.62, 0.74, hh), f)
    ac = lerp(lerp(0.32, 0.40, hh), lerp(2.2, 0.22, hh), f)
    g = np.exp(-((x - xc) / (0.13 + 0.05 * f * hh)) ** 2)
    a = a0 - (a0 - ac) * g
    # 舌头不可压缩：舌前移时咽腔变宽，舌后缩时口腔前部变宽
    a = a + 2.4 * f * np.exp(-((x - 0.26) / 0.10) ** 2) + 1.4 * (1 - f) * np.exp(-((x - 0.84) / 0.07) ** 2)
    # 下巴：口腔前部随张口变大
    oral = smoothstep(0.68, 0.86, x)
    a = a * (1 + (jaw - 0.35) * 2.2 * oral)
    # 舌尖：齿龈处的收窄 / 闭合（n、t、s）
    tg = np.exp(-((x - 0.90) / 0.035) ** 2)
    a = a - (a - A_MIN) * p["tip"] * tg
    # 嘴唇
    a_lip = (0.05 + 3.2 * p["lips"] ** 1.5) * (1 - 0.7 * p["round"]) * (0.7 + 0.6 * jaw)
    a[n - lip_n:] = np.minimum(a[n - lip_n:], a_lip) if p["lips"] > 0.02 else A_MIN
    a = np.maximum(a, A_MIN) * scale ** 2
    return a


def nose_area(scale, velum):
    nn = max(6, int(round(11.0 * scale * FS / C)))
    x = (np.arange(nn) + 0.5) / nn
    a = (0.6 + 1.6 * np.sin(np.pi * np.clip(x * 1.15, 0, 1)) + 0.4 * x) * scale ** 2
    a[0] = max(velum * 0.6 * scale ** 2, A_MIN)
    return a


class Tract:
    """一批声道同时仿真（形状 B × N），用于快速搜索。"""

    def __init__(self, areas, nose, vel):
        self.A = np.asarray(areas, float)            # (B, N)
        self.An = np.asarray(nose, float)             # (B, Nn)
        self.v = vel
        b, n = self.A.shape
        self.R = np.zeros((b, n)); self.L = np.zeros((b, n))
        self.NR = np.zeros_like(self.An); self.NL = np.zeros_like(self.An)
        s = self.A[:, :-1] + self.A[:, 1:]
        self.wa, self.wb = 2 * self.A[:, :-1] / s, 2 * self.A[:, 1:] / s
        sn = self.An[:, :-1] + self.An[:, 1:]
        self.nwa, self.nwb = 2 * self.An[:, :-1] / sn, 2 * self.An[:, 1:] / sn
        v = vel
        s3 = self.A[:, v - 1] + self.A[:, v] + self.An[:, 0]
        self.j3 = (2 * self.A[:, v - 1] / s3, 2 * self.A[:, v] / s3, 2 * self.An[:, 0] / s3)

    def step(self, src, noise=None, noise_at=None):
        R, L, NR, NL, v = self.R, self.L, self.NR, self.NL, self.v
        p = self.wa * R[:, :-1] + self.wb * L[:, 1:]
        k1, k2, k3 = self.j3
        pv = k1 * R[:, v - 1] + k2 * L[:, v] + k3 * NL[:, 0]
        p[:, v - 1] = pv
        nR = np.empty_like(R); nL = np.empty_like(L)
        nR[:, 1:] = p - L[:, 1:]
        nL[:, :-1] = p - R[:, :-1]
        nR[:, 0] = GLOTTAL_REFL * L[:, 0] + src
        nL[:, -1] = LIP_REFL * R[:, -1]
        # 鼻腔
        pn = self.nwa * NR[:, :-1] + self.nwb * NL[:, 1:]
        nNR = np.empty_like(NR); nNL = np.empty_like(NL)
        nNR[:, 1:] = pn - NL[:, 1:]
        nNL[:, :-1] = pn - NR[:, :-1]
        nNR[:, 0] = pv - NL[:, 0]
        nNL[:, -1] = NOSE_REFL * NR[:, -1]
        if noise is not None:
            nR[np.arange(len(noise)), noise_at] += noise / 2
            nL[np.arange(len(noise)), noise_at] += noise / 2
        self.R, self.L = nR * FADE, nL * FADE
        self.NR, self.NL = nNR * FADE, nNL * FADE
        return R[:, -1] * (1 + LIP_REFL) + NR[:, -1] * (1 + NOSE_REFL)


def batch(plist, scale):
    """同一几何尺寸的一批参数（圆唇程度要相同，才能有相同段数）。"""
    areas = np.array([area(p, scale) for p in plist])
    nose = np.array([nose_area(scale, p["velum"]) for p in plist])
    _, _, _, vel = geometry(scale, plist[0]["round"])
    return Tract(areas, nose, vel)


def formants(plist, scale=1.0, steps=4096, nf=3):
    """冲激响应 → 频谱 → 共振峰。返回 (B, nf) 的数组，缺的填 nan。"""
    t = batch(plist, scale)
    out = np.zeros((len(plist), steps))
    src = np.zeros(len(plist))
    for i in range(steps):
        src[:] = 1.0 if i == 0 else 0.0
        out[:, i] = t.step(src)
    nfft = 16384
    spec = np.abs(np.fft.rfft(out, nfft)) + 1e-12
    db = 20 * np.log10(spec)
    freq = np.fft.rfftfreq(nfft, 1 / FS)
    k = np.ones(5) / 5
    res = np.full((len(plist), nf), np.nan)
    fmax = 6000 / scale
    for b in range(len(plist)):
        d = np.convolve(db[b], k, mode="same")
        pk = [i for i in range(2, len(d) - 2) if freq[i] > 150 and freq[i] < fmax
              and d[i] > d[i - 1] and d[i] >= d[i + 1] and d[i] - min(d[max(0, i - 40):i + 40]) > 3]
        for j, i in enumerate(pk[:nf]):
            res[b, j] = freq[i]
    return res


def invert(target, start, rounds=5, pop=96, seed=0, prior=0.6):
    """分析—合成：只给声音（F1、F2），从一个粗略的初始姿势出发，反推发音姿势。
    prior 让结果别离初始姿势太远：同一个声音可以由很多种姿势发出（多对一），
    没有这个约束，搜索会找到"声音对、姿势怪"的解。"""
    rng = np.random.default_rng(seed)
    keys = ("front", "height", "jaw", "lips")
    center, width = dict(start), 0.25
    best, best_err = dict(start), 1e9
    for _ in range(rounds):
        cands = [dict(center)]
        for _ in range(pop - 1):
            c = dict(center)
            for k in keys:
                c[k] = float(np.clip(center[k] + rng.normal(0, width), 0.0, 1))
            cands.append(c)
        f = formants(cands, steps=2048, nf=2)
        fe = np.nansum(np.abs(np.log(f / np.array(target))), axis=1) + 10 * np.isnan(f).any(axis=1)
        pe = np.array([sum((c[k] - start[k]) ** 2 for k in keys) for c in cands])
        err = fe + prior * pe
        i = int(np.argmin(err))
        if err[i] < best_err:
            best, best_err = cands[i], float(err[i])
        center = best
        width *= 0.6
    return best, best_err


# 粗略的初始姿势：按语音学教科书的描述（舌位前后 / 高低、张口、圆唇）手工给出
GUESS = {
    "a": dict(NEUTRAL, front=0.25, height=0.0, jaw=0.85, lips=0.9),
    "i": dict(NEUTRAL, front=1.0, height=1.0, jaw=0.15, lips=0.5),
    "u": dict(NEUTRAL, front=0.15, height=0.95, jaw=0.2, lips=0.35, round=0.3),
    "e": dict(NEUTRAL, front=0.9, height=0.6, jaw=0.35, lips=0.6),
    "o": dict(NEUTRAL, front=0.05, height=0.5, jaw=0.45, lips=0.3, round=0.7),
}


# ---------------- 声源与整句合成（和网页版一致） ----------------

def glottal(phase, oq, tense):
    """一个周期内的声门气流导数。oq = 开放比，tense 越大闭合越陡。"""
    tp = oq * (0.55 + 0.25 * tense)
    tn = oq - tp
    if phase < tp:
        return math.sin(math.pi * phase / tp) * (math.pi / tp) * 0.5
    if phase < oq:
        return -math.sin(math.pi / 2 * (phase - tp) / tn) * (math.pi / 2 / tn) * (0.6 + 0.6 * tense)
    return 0.0


def render(segments, scale=1.0, f0=220.0, oq=0.6, tense=0.5, breath=0.05, seed=1):
    """segments: [(参数字典, 持续秒数, 是否发声, 摩擦噪声强度)]，逐段线性过渡。返回 48 kHz 波形。"""
    rng = np.random.default_rng(seed)
    out, phase = [], 0.0
    cur = dict(segments[0][0])
    for p, dur, voiced, fric in segments:
        steps = int(dur * FS)
        start = dict(cur)
        for s in range(0, steps, 48):              # 每 0.5 ms 更新一次形状
            t = min(1.0, (s + 48) / (FS * 0.04))   # 40 ms 过渡
            cur = {k: lerp(start[k], p[k], t) for k in PARAMS}
            cur["round"] = p["round"]
            tr = batch([cur], scale) if s == 0 else _reshape(tr, cur, scale)
            a = tr.A[0]
            at = int(np.argmin(a[: len(a) - 1])) + 1
            for _ in range(48):
                phase += f0 * (1 + 0.004 * rng.normal()) / FS
                phase %= 1.0
                g = glottal(phase, oq, tense) * (1.0 if voiced else 0.0) * 0.02
                asp = rng.normal() * (breath + (0.0 if voiced else 0.02)) * 0.02
                nz = rng.normal() * fric * 0.004 * (1 if a[at - 1] < 0.4 * scale ** 2 else 0)
                y = tr.step(np.array([g + asp]), np.array([nz]), np.array([at]))
                out.append(y[0])
    y = np.array(out)
    # 声源已经是气流的导数（相当于嘴唇辐射的 +6 dB/倍频程），这里只做低通 + 降到 48 kHz
    y = 0.5 * (y[0::2][: len(y) // 2] + y[1::2][: len(y) // 2])
    for _ in range(2):
        y = np.convolve(y, [0.25, 0.5, 0.25], mode="same")
    return y / (np.max(np.abs(y)) + 1e-9) * 0.8


def _reshape(tr, p, scale):
    """只改面积，不清空波（声音连续）。"""
    nt = batch([p], scale)
    if nt.A.shape == tr.A.shape:
        nt.R, nt.L, nt.NR, nt.NL = tr.R, tr.L, tr.NR, tr.NL
    return nt


if __name__ == "__main__":
    import time
    t0 = time.time()
    print("1) 分析—合成：只给出目标共振峰，反推发音姿势（成年男性声道 17 cm）")
    presets = {}
    for v, tgt in TARGETS.items():
        g = formants([GUESS[v]])[0]
        p, err = invert(tgt, GUESS[v])
        f = formants([p])[0]
        presets[v] = p
        print(f"  {v}: 目标 F1/F2 = {tgt[0]}/{tgt[1]} Hz，初始猜测 {g[0]:.0f}/{g[1]:.0f} → 调整后 {f[0]:.0f}/{f[1]:.0f}/{f[2]:.0f} Hz"
              f"  (误差 {abs(f[0]/tgt[0]-1)*100:.0f}% / {abs(f[1]/tgt[1]-1)*100:.0f}%)   姿势 "
              + " ".join(f"{k}={p[k]:.2f}" for k in ("front", "height", "jaw", "lips", "round")))
    print("\n2) 同一套姿势，换不同长度的声道（共振峰按长度反比上升）")
    sizes = [("成年男性", 1.0), ("成年女性", 14.5 / 17), ("十岁儿童", 12 / 17), ("55 cm 人偶头内", 8 / 17)]
    for name, s in sizes:
        row = []
        for v in "aiueo":
            f = formants([presets[v]], s)[0]
            row.append(f"{v} {f[0]:.0f}/{f[1]:.0f}")
        print(f"  {name:10s} {17*s:5.1f} cm: " + "  ".join(row))
    print("\nPRESETS = {")
    for v, p in presets.items():
        print(f'  {v}: {{{", ".join(f"{k}: {p[k]:.3f}" for k in PARAMS)}}},')
    print("}")
    print(f"\n用时 {time.time()-t0:.0f} s")
