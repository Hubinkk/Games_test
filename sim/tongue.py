"""
两段绳驱"触手舌头"＋ 弹性皮肤：能发出多少日语音？

用法：
    python tongue.py           # 元音、t/n、k、s 的可达性，三种结构对比，输出 tongue_shapes.svg

结构（每一段都是"常曲率"连续体，像章鱼触手 / 医疗内窥镜）：
  - 舌根滑台：整根舌头前后移动（1 个电机）；可选再加升降（1 个电机），像人的舌骨和颏舌肌把整个舌体推上去
  - 第一段（舌体，6.5 cm）：一对拮抗的线，让它整体前卷或后仰（1 个电机，差动绕法）
  - 第二段（舌尖，2.2 cm）：一对线，舌尖上卷 / 下压（1 个电机）
  - 可选：内部气腔，充气让舌体变厚（气泵 + 阀）
  - 下巴：转动，带着整根舌头和下唇一起下降（1 个电机）
外面是硅胶皮肤，舌体最厚约 2 cm，舌尖约 0.25 cm。
尺寸按成年人（声道约 17 cm）建模，便于与人的元音数据对比；人偶上整体缩到 8/17。

声学：在中矢状面上用"半极坐标网格"量出舌面到上腭 / 咽后壁的距离 d，
截面积 A = 1.5·d^1.5（嘴唇处按唇宽 × 开口），再算共振峰。
搜索时用频域的传输线模型（快），最后用 voice.py 的时域波导复核。
"""
import math
import os

import numpy as np

import voice as V

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = 8.0 / 17.0                      # 人偶 / 成人 的尺寸比

# ---------------- 固定的上壁（成人，cm；x 向前，y 向上，声门在原点） ----------------
WALL = np.array([(-0.3, 0.0), (-0.3, 7.0), (-0.1, 8.2), (0.5, 9.0), (1.5, 9.5), (2.8, 9.9),
                 (4.2, 10.1), (5.4, 9.9), (6.1, 9.5), (6.6, 8.9), (7.4, 8.8), (8.4, 8.8)])
TMJ = np.array([0.8, 9.4])            # 下巴的转轴
O = np.array([3.2, 6.6])              # 口腔网格的中心
PHAR_Y = np.arange(1.0, 7.01, 0.5)    # 咽部的水平网格线
RAY_DEG = np.arange(160, 31, -8.0)    # 口腔的放射网格线（从软腭到齿龈）
ALV = RAY_DEG <= 48                   # 齿龈区（t、n、s）
VEL = (RAY_DEG >= 112) & (RAY_DEG <= 152)   # 软腭区（k）
L1, L2 = 6.5, 2.2                     # 两段长度
CABLE_R = 0.5                         # 线到中轴的距离（成人尺寸），人偶上 ×8/17

# 各结构的可动参数与范围
RANGES = {
    "slide": (-0.9, 0.9),     # 舌根前后，cm
    "lift": (-0.6, 1.2),      # 舌根升降，cm（像人的舌骨 / 颏舌肌把整个舌体推上去）
    "phi1": (-2.4, 0.6),      # 第一段总弯角，rad（负 = 向前卷）
    "phi2": (-1.2, 1.8),      # 第二段总弯角，rad（正 = 舌尖上卷）
    "bulge": (-0.4, 0.5),     # 气腔：厚度 ×(1+bulge)
    "jaw": (0.0, 1.0),        # 下巴张开（0–0.22 rad）
    "lips": (0.0, 1.0),       # 嘴唇开口
    "round": (0.0, 1.0),      # 圆唇 / 前伸
}
CONFIGS = {                     # 只搜索舌头本身；下巴和嘴唇按人的姿势固定（它们是另外的机构）
    "一段": ("slide", "phi1"),
    "两段": ("slide", "phi1", "phi2"),
    "两段+升降": ("slide", "lift", "phi1", "phi2"),
    "两段+升降+气腔": ("slide", "lift", "phi1", "phi2", "bulge"),
}
MOTORS = {"一段": 2, "两段": 3, "两段+升降": 4, "两段+升降+气腔": 4}
BEST = "两段+升降"
GRID = {"一段": 41, "两段": 15, "两段+升降": 9, "两段+升降+气腔": 6}
# 每个音的下巴、嘴唇（接近人的做法）
FIXED = {
    "a": dict(jaw=0.9, lips=0.85, round=0.0),
    "i": dict(jaw=0.2, lips=0.35, round=0.0),
    "u": dict(jaw=0.15, lips=0.25, round=0.4),
    "e": dict(jaw=0.45, lips=0.55, round=0.0),
    "o": dict(jaw=0.6, lips=0.45, round=0.7),
    "cons": dict(jaw=0.2, lips=0.45, round=0.0),
}
REST = dict(lift=0.0, slide=0.0, phi1=-1.0, phi2=0.0, bulge=0.0, jaw=0.3, lips=0.5, round=0.0)


def rot(p, c, a):
    s, co = math.sin(a), math.cos(a)
    d = p - c
    return c + np.array([co * d[0] - s * d[1], s * d[0] + co * d[1]])


def thickness(u, bulge):
    """舌头厚度沿长度的分布（u：0 舌根 → 1 舌尖）。"""
    body = 0.25 + 2.15 * np.sin(np.pi * np.clip(u / 0.82, 0, 1)) ** 0.7
    body = np.where(u > 0.82, 0.25 + (body - 0.25) * 0, body)
    return np.maximum(body * (1 + bulge * (u < 0.8)), 0.2)


def tongue(p):
    """返回舌背（上表面）的折线。中轴是两段常曲率圆弧，表面 = 中轴向背侧偏移厚度。"""
    ja = -0.22 * p["jaw"]
    base = rot(np.array([2.6 + p["slide"], 3.0 + p.get("lift", 0.0)]), TMJ, ja)
    th = math.radians(95) + ja
    pts, normals, us = [], [], []
    n1, n2 = 40, 16
    pos = base.copy()
    for seg, (length, phi, n) in enumerate(((L1, p["phi1"], n1), (L2, p.get("phi2", 0.0), n2))):
        k = phi / length
        ds = length / n
        for i in range(n):
            pts.append(pos.copy())
            normals.append(np.array([-math.sin(th), math.cos(th)]))
            us.append((seg * L1 + i * ds) / (L1 + L2))
            pos = pos + ds * np.array([math.cos(th + k * ds / 2), math.sin(th + k * ds / 2)])
            th += k * ds
    pts.append(pos.copy()); normals.append(np.array([-math.sin(th), math.cos(th)])); us.append(1.0)
    pts, normals, us = np.array(pts), np.array(normals), np.array(us)
    h = thickness(us, p.get("bulge", 0.0))
    return pts + normals * h[:, None], pts


def lower_lip(p):
    ja = -0.22 * p["jaw"]
    tip = rot(np.array([7.0, 8.75]), TMJ, ja)          # 下门牙 / 下唇
    return tip[1] - 0.15 - 1.1 * p["lips"]


def seg_intersections(o, d, poly):
    """射线 o + t·d（t>0）与折线的所有交点的 t。"""
    a, b = poly[:-1], poly[1:]
    e = b - a
    den = d[0] * e[:, 1] - d[1] * e[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = ((a[:, 0] - o[0]) * e[:, 1] - (a[:, 1] - o[1]) * e[:, 0]) / den
        s = ((a[:, 0] - o[0]) * d[1] - (a[:, 1] - o[1]) * d[0]) / den
    ok = (np.abs(den) > 1e-12) & (s >= 0) & (s <= 1) & (t > 0)
    return t[ok]


def midsagittal(p):
    """沿网格线量出 舌面—上壁 的距离 d（负数 = 舌头压进了上壁的深度）与中线位置。"""
    surf, _ = tongue(p)
    dist, mids = [], []
    # 咽部：水平线，向前看，咽后壁 → 舌根
    for y in PHAR_Y:
        o, d = np.array([-0.3, y]), np.array([1.0, 0.0])
        ts = seg_intersections(np.array([-5.0, y]), d, surf)
        front = (ts.min() - 5.0 + 0.3 * 0) if len(ts) else 4.0
        x_t = -5.0 + ts.min() if len(ts) else 3.7
        if y < 3.5:
            x_t = min(x_t, 1.6)          # 舌根以下：会厌和喉咽的前壁
        dist.append(x_t - (-0.3)); mids.append(((-0.3 + x_t) / 2, y))
    # 口腔：从中心 O 往上壁放射
    for deg in RAY_DEG:
        d = np.array([math.cos(math.radians(deg)), math.sin(math.radians(deg))])
        tw = seg_intersections(O, d, WALL)
        tw = tw.min() if len(tw) else 6.0
        ts = seg_intersections(O - d * 6, d, surf)          # 从远处射过来，取最靠外的交点
        tt = (ts.max() - 6) if len(ts) else -6.0
        dist.append(tw - tt); mid = (tw + max(tt, 0)) / 2
        mids.append(tuple(O + d * mid))
    # 齿与嘴唇：竖线，上唇 → 下唇
    ll = lower_lip(p)
    for x in (6.9, 7.4, 7.9):
        dist.append(8.8 - ll); mids.append((x, (8.8 + ll) / 2))
    return np.array(dist), np.array(mids)


def area_fn(p):
    """中矢状距离 → 截面积序列 + 每段长度（从声门到嘴唇）。"""
    d, mids = midsagittal(p)
    n_lip = 3
    a = 1.5 * np.maximum(d, 0.0) ** 1.5
    lipw = 2.6 * (1 - 0.5 * p["round"])
    a[-n_lip:] = 0.8 * lipw * np.maximum(d[-n_lip:], 0.0)
    seg = np.r_[np.linalg.norm(mids[0] - np.array([0, 0.5])), np.linalg.norm(np.diff(mids, axis=0), axis=1)]
    seg[-n_lip:] = 0.5 * (1 + p["round"])                    # 圆唇时嘴唇前伸
    seg = seg * GEO
    # 声门上方 1 cm 的喉前庭
    a = np.r_[1.2, 1.2, a]
    seg = np.r_[0.5, 0.5, seg]
    return np.maximum(a, V.A_MIN), seg, d


GEO = 1.0


def _calibrate():
    """网格中线的长度比真实声道略长；按静止姿势把总长归一到 17 cm。"""
    global GEO
    GEO = 1.0
    _, seg, _ = area_fn(dict(REST))
    GEO = 17.0 / seg.sum()


def transfer(a, seg, fmax=5000, df=5.0):
    """频域传输线（带一点损耗与嘴唇辐射）：|U_lip / U_glottis|。"""
    f = np.arange(df, fmax, df)
    w = 2 * np.pi * f
    k = w / V.C * (1 - 0.006j) - 0.0004j * np.sqrt(f)
    A11 = np.ones_like(f, complex); A12 = np.zeros_like(f, complex)
    A21 = np.zeros_like(f, complex); A22 = np.ones_like(f, complex)
    for ai, li in zip(a, seg):
        c, s = np.cos(k * li), np.sin(k * li)
        b12, b21 = 1j * s / ai, 1j * s * ai
        A11, A12, A21, A22 = A11 * c + A12 * b21, A11 * b12 + A12 * c, A21 * c + A22 * b21, A21 * b12 + A22 * c
    rad = math.sqrt(a[-1] / math.pi)
    ka = w / V.C * rad
    zr = (0.5 * ka ** 2 + 0.8j * ka) / a[-1]
    return f, np.abs(1 / (A21 * zr + A22))


def peaks(f, h, nf=3):
    db = 20 * np.log10(h + 1e-12)
    out = [f[i] for i in range(1, len(db) - 1) if db[i] > db[i - 1] and db[i] >= db[i + 1] and f[i] > 150]
    return (out + [np.nan] * nf)[:nf]


def formants_fast(p):
    a, seg, _ = area_fn(p)
    return peaks(*transfer(a, seg))


def formants_kl(p, scale=1.0):
    """用 voice.py 的时域波导复核：把面积按段长重采样成等长小管。"""
    a, seg, _ = area_fn(p)
    total = seg.sum() * scale
    n = int(round(total * V.FS / V.C))
    edges = np.r_[0, np.cumsum(seg)] / seg.sum()
    x = (np.arange(n) + 0.5) / n
    idx = np.clip(np.searchsorted(edges, x) - 1, 0, len(a) - 1)
    A = a[idx] * scale ** 2
    nose = V.nose_area(scale, 0.0)
    vel = int(round(n * 0.52))
    t = V.Tract(A[None, :], np.resize(nose, (1, len(nose))), vel)
    out = np.zeros(4096)
    for i in range(4096):
        out[i] = t.step(np.array([1.0 if i == 0 else 0.0]))[0]
    spec = 20 * np.log10(np.abs(np.fft.rfft(out, 16384)) + 1e-12)
    fr = np.fft.rfftfreq(16384, 1 / V.FS)
    sm = np.convolve(spec, np.ones(5) / 5, mode="same")
    pk = [fr[i] for i in range(2, len(sm) - 2) if 150 < fr[i] < 6000 / scale and sm[i] > sm[i - 1]
          and sm[i] >= sm[i + 1] and sm[i] - sm[max(0, i - 40):i + 40].min() > 3]
    return (pk + [np.nan] * 3)[:3], total


# ---------------- 搜索 ----------------
def search(cost, keys, base, n_grid, starts=(), iters=6, pop=40, seed=0):
    """先在网格上全搜一遍（不会漏掉），再从最好的几个点局部细化。starts：额外的起点（比如更简单结构的最优解）。"""
    rng = np.random.default_rng(seed)
    axes = [np.linspace(*RANGES[k], n_grid) for k in keys]
    mesh = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, len(keys))
    scored = []
    for row in mesh:
        p = dict(base, **{k: float(v) for k, v in zip(keys, row)})
        scored.append((cost(p), p))
    for st in starts:
        p = dict(base, **{k: st.get(k, base[k]) for k in keys})
        scored.append((cost(p), p))
    scored.sort(key=lambda t: t[0])
    best_c, best = scored[0]
    for c0, center in scored[:4]:
        width = 1.0 / n_grid
        for _ in range(iters):
            for _ in range(pop):
                c = dict(center)
                for k in keys:
                    lo, hi = RANGES[k]
                    c[k] = float(np.clip(center[k] + rng.normal(0, width) * (hi - lo), lo, hi))
                cc = cost(c)
                if cc < c0:
                    c0, center = cc, c
            width *= 0.6
        if c0 < best_c:
            best_c, best = c0, center
    return best, best_c


def penetration(d):
    return max(0.0, -d.min())


def vowel_cost(tgt):
    def c(p):
        a, seg, d = area_fn(p)
        f = peaks(*transfer(a, seg))
        if np.isnan(f[0]) or np.isnan(f[1]):
            return 10
        e = abs(math.log(f[0] / tgt[0])) + abs(math.log(f[1] / tgt[1]))
        return e + 5 * (d[:-3].min() < 0.15) + 2 * penetration(d)    # 元音时声道不能闭合
    return c


def closure_cost(region, gap=0.0):
    """在某个区域做出闭合（gap=0）或指定宽度的窄缝，其他地方保持张开。"""
    def c(p):
        _, _, d = area_fn(p)
        dr = d[len(PHAR_Y):len(PHAR_Y) + len(RAY_DEG)]
        inside, outside = dr[region], np.r_[d[:len(PHAR_Y)], dr[~region]]
        m = inside.min()
        e = abs(max(m, 0) - gap) if gap > 0 else max(m, 0)
        if gap > 0:
            e += 2 * max(0, -m)                         # 窄缝不能变成闭合
        e += 2 * max(0, 0.35 - outside.min())           # 别处要通畅
        e += 3 * max(0, penetration(d) - 0.25)          # 软皮肤可以压进去一点，太深就不现实
        return e
    return c


def svg_shapes(shapes, path):
    """把每个音的舌形画成一张 SVG（成人尺寸，cm）。"""
    W, H = 220, 190
    def tx(x, y):
        return f"{(x + 1.2) * 22:.1f},{(11 - y) * 22:.1f}"
    cells = []
    for i, (name, p, ok) in enumerate(shapes):
        surf, spine = tongue(p)
        ll = lower_lip(p)
        gx, gy = (i % 4) * W, (i // 4) * H
        wall = " ".join(tx(*q) for q in WALL)
        tong = " ".join(tx(*q) for q in surf)
        sp = " ".join(tx(*q) for q in spine)
        cells.append(
            f'<g transform="translate({gx},{gy}) scale(0.95)">'
            f'<polyline points="{wall}" fill="none" stroke="var(--ink-2)" stroke-width="2"/>'
            f'<polygon points="{tong} {tx(*spine[0])}" fill="var(--tongue)" fill-opacity="0.85" stroke="var(--ink)" stroke-width="1"/>'
            f'<polyline points="{sp}" fill="none" stroke="var(--ink)" stroke-dasharray="3 3" stroke-width="1"/>'
            f'<line x1="{(6.6+1.2)*22:.1f}" y1="{(11-ll)*22:.1f}" x2="{(8.2+1.2)*22:.1f}" y2="{(11-ll)*22:.1f}" stroke="var(--tongue)" stroke-width="5" stroke-linecap="round"/>'
            f'<text x="10" y="22" font-size="17" font-weight="600" fill="var(--ink)">{name}</text>'
            f'<text x="10" y="42" font-size="12" fill="var(--{"ok" if ok else "no"})">{"可以" if ok else "不行"}</text></g>')
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {4*W} {((len(shapes)+3)//4)*H}" '
           f'font-family="sans-serif">' + "".join(cells) + "</svg>")
    open(path, "w", encoding="utf-8").write(svg)


if __name__ == "__main__":
    import time
    t0 = time.time()
    _calibrate()
    results, prev = {}, {}
    for cname, keys in CONFIGS.items():
        print(f"\n===== {cname}（舌头 {MOTORS[cname]} 个电机）：可动参数 {', '.join(keys)} =====")
        res = {}
        for v, tgt in V.TARGETS.items():
            base = dict(REST, **FIXED[v])
            p, c = search(vowel_cost(tgt), keys, base, GRID[cname], starts=[prev[v][0]] if v in prev else [])
            f = formants_fast(p)
            res[v] = (p, f, [abs(f[0] / tgt[0] - 1), abs(f[1] / tgt[1] - 1)])
            print(f"  {v}: 目标 {tgt[0]}/{tgt[1]} → {f[0]:.0f}/{f[1]:.0f}  误差 {res[v][2][0]*100:.0f}% / {res[v][2][1]*100:.0f}%")
        for name, region, gap in (("t/n（齿龈闭合）", ALV, 0.0), ("k（软腭闭合）", VEL, 0.0), ("s（齿龈 1 mm 窄缝）", ALV, 0.1)):
            base = dict(REST, **FIXED["cons"])
            p, c = search(closure_cost(region, gap), keys, base, GRID[cname], starts=[prev[name][0]] if name in prev else [])
            _, _, d = area_fn(p)
            dr = d[len(PHAR_Y):len(PHAR_Y) + len(RAY_DEG)]
            ok = c < 0.03
            res[name] = (p, dr[region].min(), ok, penetration(d))
            print(f"  {name}: {'可以' if ok else '不行'}  该区最小距离 {dr[region].min()*10:+.1f} mm，"
                  f"皮肤压入 {penetration(d)*10:.1f} mm（代价 {c:.3f}）")
        results[cname] = prev = res
    # 用时域波导复核最好结构的元音
    print(f"\n===== 复核：{BEST} 的元音，用 voice.py 的时域波导（成人 / 人偶 8 cm）=====")
    for v in V.TARGETS:
        p = results[BEST][v][0]
        fk, L = formants_kl(p)
        ff, Lf = formants_kl(p, FIG)
        print(f"  {v}: 频域 {results[BEST][v][1][0]:.0f}/{results[BEST][v][1][1]:.0f}  时域 {fk[0]:.0f}/{fk[1]:.0f}"
              f"  （声道 {L:.1f} cm）   人偶尺寸 {ff[0]:.0f}/{ff[1]:.0f}（{Lf:.1f} cm）")
    # s 的窄缝需要多精确的线
    p = results[BEST]["s（齿龈 1 mm 窄缝）"][0]
    def gap(pp):
        _, _, d = area_fn(pp)
        return d[len(PHAR_Y):len(PHAR_Y) + len(RAY_DEG)][ALV].min()
    eps = 1e-3
    g0 = gap(p)
    dg = (gap(dict(p, phi2=p["phi2"] + eps)) - g0) / eps
    tol_phi = 0.03 / abs(dg) if dg else float("inf")     # 缝宽 1 mm ± 0.3 mm（成人）
    print(f"\n===== s 的精度 =====\n  缝宽对舌尖弯角的灵敏度 {dg*10:.1f} mm/rad；缝宽要保持在 1±0.3 mm，"
          f"舌尖弯角误差需 < {tol_phi*1000:.0f} mrad")
    print(f"  换成线的位移：成人 {CABLE_R*tol_phi*1e4:.0f} µm，人偶尺寸（线距 {CABLE_R*FIG*10:.1f} mm，缝宽约 0.5 mm）"
          f" {CABLE_R*FIG*tol_phi*1e4:.0f} µm")
    # 速度：相邻元音之间线要走多远
    R = results[BEST]
    mv = max(abs(R[a][0][k] - R[b][0][k]) for a in "aiueo" for b in "aiueo" for k in ("phi1", "phi2"))
    ms = max(abs(R[a][0][k] - R[b][0][k]) for a in "aiueo" for b in "aiueo" for k in ("slide", "lift"))
    print(f"  元音之间最大的弯角变化 {mv:.2f} rad → 人偶上线的位移 {CABLE_R*FIG*mv*10:.1f} mm，70 ms 内完成需线速度 {CABLE_R*FIG*mv/0.07*10:.0f} mm/s")
    print(f"  舌根最大移动 {ms:.2f} cm → 人偶上 {ms*FIG*10:.1f} mm，70 ms 内完成需 {ms*FIG/0.07*10:.0f} mm/s")
    sh = [(v, results[BEST][v][0], max(results[BEST][v][2]) < 0.2) for v in "aiueo"]
    sh += [(n.split("（")[0], results[BEST][n][0], results[BEST][n][2]) for n in ("t/n（齿龈闭合）", "k（软腭闭合）", "s（齿龈 1 mm 窄缝）")]
    svg_shapes(sh, os.path.join(HERE, "tongue_shapes.svg"))
    print(f"\nPARAMS（{BEST}）=")
    for n, r in results[BEST].items():
        print("  ", n, {k: round(v, 3) for k, v in r[0].items()})
    print(f"\n用时 {time.time()-t0:.0f} s")
