"""
55 cm 人偶 · 第二版方案的账本（估算）：头部散热、续航、胸腔空间

用法：
    python plan2.py        # 打印并写 plan2_result.json

育苗人 2026-11-10 的要求：大脑不要离头太远、头部尽量加散热（背后可取出的模块、液冷）；续航尽量加长；
胸腔再省空间，给以后留余地。这里只是算账；重量变化后的站、蹲、走由 trim.py 仿真。
"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------- 1. 头部散热 ----------
Q = 15.0                                   # W：大脑全力思考时要带走的热（估算，本地计算单元 7–15 W）
c_water, dT_liquid = 4.18, 5.0             # J/(g·K)；冷却液进出温差
flow_g_s = Q / (c_water * dT_liquid)
flow_ml_min = flow_g_s * 60
tube_id, tube_len, mu = 0.002, 0.6, 1e-3   # 内径 2 mm，来回共 0.6 m
v = (flow_g_s * 1e-6) / (math.pi * (tube_id / 2) ** 2)
dp_kpa = 32 * mu * tube_len * v / tube_id ** 2 / 1000
rad_area_cm2 = Q / (35.0 * 15.0) * 1e4     # 小风扇吹着：35 W/(m²·K)，散热片比空气热 15 K
air_l_s = Q / (1.2 * 1005 * 10.0) * 1000   # 如果改成头后面直接风冷：空气升温 10 K 需要的风量
LIQUID = {"头里的冷板": 10, "泵": 15, "散热器": 30, "风扇": 8, "冷却液 12 mL": 12, "两根软管": 5}
AIR = {"后脑勺的风道和小风扇": 15}
cooling = dict(Q_W=Q, flow_ml_min=round(flow_ml_min), tube_dp_kPa=round(dp_kpa, 1), radiator_cm2=round(rad_area_cm2),
               liquid_g=sum(LIQUID.values()), liquid_in_head_g=10, liquid_power_W=0.9,
               air_l_s=round(air_l_s, 1), air_g=sum(AIR.values()), air_power_W=0.4)

# ---------- 2. 续航 ----------
WH_PER_KG = 250.0                          # 21700 电芯的公开规格：约 18 Wh / 69 g
USABLE = 0.85
FIXED = 3.1 + 0.5                          # 屏幕、摄像头、麦克风、控制器 + 散热（平均）
HEART, THINK, DUTY = 3.0, 12.0, 0.3        # 一直开着的"心" 3 W；"思考"全开 12 W，平均 30% 的时间在用
compute_avg = HEART + (THINK - HEART) * DUTY
MOTORS = {"坐着": 1.0, "站着说话、做表情": 4.0, "走路、练习": 10.0}
MIX = {"坐着": 0.6, "站着说话、做表情": 0.3, "走路、练习": 0.1}
endurance = {}
for wh in (30, 60, 90):
    row = {}
    for k, mw in MOTORS.items():
        row[k] = round(wh * USABLE / (FIXED + compute_avg + mw), 1)
    avg = sum(MIX[k] * (FIXED + compute_avg + mw) for k, mw in MOTORS.items())
    row["平常的一天（六成坐、三成站、一成走）"] = round(wh * USABLE / avg, 1)
    row["电池重 g"] = round(wh / WH_PER_KG * 1000)
    endurance[f"{wh} Wh"] = row

# ---------- 3. 胸腔 ----------
CHEST_AVAIL = 147
before = {"肩俯仰电机 ×2": 36, "手部电机 12 个 + 6 个锁": 48, "手部驱动板": 8, "控制器、IMU、电源板、风扇": 20, "脖子电机 4 个 + 空位": 20}
moves = {"控制器、IMU、电源板搬到腹部（电池搬去背包后，腹部空着 44 cm³）": -20,
         "每只手的 2 个手腕电机搬进前臂（前臂各有 10 cm³）": -14}
after = sum(before.values()) + sum(moves.values())
chest = dict(avail=CHEST_AVAIL, before=sum(before.values()), after=after, free=CHEST_AVAIL - after,
             abdomen_after=36 + 20, abdomen_avail=80, forearm_after=7, forearm_avail=10)

out = dict(cooling=cooling, compute_avg_W=round(compute_avg, 1), endurance=endurance, chest=chest)
print(json.dumps(out, ensure_ascii=False, indent=1))
json.dump(out, open(os.path.join(HERE, "plan2_result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
