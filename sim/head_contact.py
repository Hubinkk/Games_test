"""
起身时头碰地的是哪里、压了多大的力、在地上拖了多远（2026-11-11）

用法：
    python head_contact.py        # 用学到的策略起身（地面摩擦 1.0），脖子试 5 种姿势，写 head_contact_result.json

背景：这条起身动作在 2 秒左右是"头顶着地往前拱"。育苗人决定头着地的部位用光滑材料。
花然的眼睛（屏幕、摄像头、眼睑）在脸的正面，所以要知道着地的是头顶、额头，还是脸。
头在仿真里是一个半径 4.6 cm 的球；着地点用"离脸正前方多少度、离头顶多少度"来说。
"""
import json
import math
import os

import mujoco
import numpy as np

import getup_hands as G
import getup_rl

HERE = os.path.dirname(os.path.abspath(__file__))
EYE = [np.array([0.041, s * 0.016, 0.037 - 0.044]) for s in (1, -1)]      # 眼睛中心相对头的球心（见 figure55.xml）
EYE = [e / np.linalg.norm(e) for e in EYE]
EYE_HALF_DEG = math.degrees(math.asin(0.014 / 0.046))                        # 眼球半径 1.4 cm 在头的球面上占的角半径


NECK_VARIANTS = {          # 起身的头 4.5 秒，脖子不听策略的，改成固定的姿势（整体模型接口里的目标角，弧度）
    "原样": None,
    "低头收下巴": dict(neck_pitch=0.6, neck_low=0.6),
    "头转向左边": dict(neck_yaw=1.3),
    "头转向左边并低头": dict(neck_yaw=1.3, neck_pitch=0.4, neck_low=0.3),
    "仰头": dict(neck_pitch=-0.7, neck_low=-0.5),
}


def main(variant="原样"):
    sim = G.get_sim()
    over = NECK_VARIANTS[variant]
    orig_set = sim.set_targets
    if over:
        def patched(pose):
            if sim.d.time < 4.5:
                pose = dict(pose)
                pose.update(over)
            orig_set(pose)
        sim.set_targets = patched
    m, d = sim.m, sim.d
    hg = m.geom("head").id
    hb = m.geom_bodyid[hg]
    floor = m.geom("floor").id
    dt = m.opt.timestep
    log = []
    f6 = np.zeros(6)

    def rec():
        for c in range(d.ncon):
            con = d.contact[c]
            if {int(con.geom1), int(con.geom2)} == {hg, floor}:
                mujoco.mj_contactForce(m, d, c, f6)
                R = d.xmat[hb].reshape(3, 3)
                u = R.T @ (con.pos - d.geom_xpos[hg])
                u = u / max(np.linalg.norm(u), 1e-9)
                v = np.zeros(6)
                mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_GEOM, hg, v, 0)
                # 着地点在地面上的滑动速度：球心速度 + 角速度 × 半径
                w, lin = v[:3], v[3:]
                vp = lin + np.cross(w, con.pos - d.geom_xpos[hg])
                log.append((d.time, float(f6[0]), float(math.hypot(f6[1], f6[2])), u.copy(), float(np.linalg.norm(vp[:2]))))
    sim.recorder = rec
    r = getup_rl.run(sim)
    sim.recorder = None
    sim.set_targets = orig_set
    t = np.array([x[0] for x in log])
    fn = np.array([x[1] for x in log])
    ft = np.array([x[2] for x in log])
    U = np.array([x[3] for x in log])
    sp = np.array([x[4] for x in log])
    on = fn > 0.5                                           # 压力超过 0.5 N 才算真的压着
    front = np.degrees(np.arccos(np.clip(U[:, 0], -1, 1)))  # 离脸正前方多少度
    if not len(log):
        return dict(variant=variant, ok=r['ok'], total_contact_s=0.0)
    top = np.degrees(np.arccos(np.clip(U[:, 2], -1, 1)))    # 离头顶多少度
    eye = np.degrees(np.minimum(*[np.arccos(np.clip(U @ e, -1, 1)) for e in EYE]))
    phases = []
    for a, b in ((0.0, 1.5), (1.5, 2.6), (2.6, 4.0), (4.0, 20.0)):
        k = on & (t >= a) & (t < b)
        if k.sum() == 0:
            phases.append(dict(t=[a, b], contact_s=0.0))
            continue
        phases.append(dict(t=[a, b], contact_s=round(float(k.sum() * dt), 2), force_max_N=round(float(fn[k].max()), 1),
                           force_mean_N=round(float(fn[k].mean()), 1), friction_max_N=round(float(ft[k].max()), 1),
                           slide_cm=round(float((sp[k] * dt).sum()) * 100, 1),
                           from_face_front_deg=[round(float(front[k].min())), round(float(front[k].max()))],
                           from_top_deg=[round(float(top[k].min())), round(float(top[k].max()))],
                           nearest_to_eye_deg=round(float(eye[k].min()))))
    k = on
    over_eye = k & (eye < EYE_HALF_DEG)
    out = dict(ok=r["ok"], eye_half_angle_deg=round(EYE_HALF_DEG, 1), last_contact_s=round(float(t[k].max()), 2) if k.any() else None,
               total_contact_s=round(float(k.sum() * dt), 2), peak_force_N=round(float(fn[k].max()), 1),
               total_slide_cm=round(float((sp[k] * dt).sum()) * 100, 1),
               on_eye_s=round(float(over_eye.sum() * dt), 2), on_eye_peak_N=round(float(fn[over_eye].max()), 1) if over_eye.any() else 0.0,
               on_eye_slide_cm=round(float((sp[over_eye] * dt).sum()) * 100, 1), nearest_to_eye_deg=round(float(eye[k].min())), phases=phases)
    out["variant"] = variant
    # 压着地的时间里，着地点落在哪个区域（按离脸正前方的角度分）：脸 < 45°，侧面和头顶 45–110°，后脑 > 110°
    out["share"] = {"脸": round(float((k & (front < 45)).sum() / max(k.sum(), 1)), 2),
                    "侧面或头顶": round(float((k & (front >= 45) & (front <= 110)).sum() / max(k.sum(), 1)), 2),
                    "后脑": round(float((k & (front > 110)).sum() / max(k.sum(), 1)), 2)}
    print(f"【{variant}】着地点落在：{out['share']}")
    print(f"站起来了：{r['ok']}；头压着地共 {out['total_contact_s']} 秒，最后一次在 {out['last_contact_s']} 秒；最大压力 {out['peak_force_N']} N；在地上拖了 {out['total_slide_cm']} cm")
    print(f"压在眼睛范围里（离眼睛中心 {out['eye_half_angle_deg']}° 以内）：{out['on_eye_s']} 秒，最大 {out['on_eye_peak_N']} N，拖了 {out['on_eye_slide_cm']} cm；着地点离眼睛中心最近 {out['nearest_to_eye_deg']}°")
    return out


if __name__ == "__main__":
    res = [main(v) for v in NECK_VARIANTS]
    json.dump(res, open(os.path.join(HERE, "head_contact_result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
