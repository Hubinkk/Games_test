# 台式机（RTX 5070 Ti）交接：把人偶带进 Isaac Lab

目标：在 Isaac Lab 里用强化学习训练 55 cm 人偶**站立 → 行走 → 受损后的补偿**。
笔记本上的 MuJoCo 测试已经证明：这个身体能站、能深蹲、能单腿站立（脚宽 4.4 cm），膝关节 2.8 N·m 够用。

## 0. 先确认的两件事

- 内存是否 ≥ 32 GB，系统是否是 Windows 11 或 Ubuntu 22.04 / 24.04（交接摘要第 1 版里的待确认项）
- 按 Isaac Lab 官方文档安装 Isaac Sim 和 Isaac Lab。安装方式和版本变化较快，以官方文档为准

## 1. 准备好的文件

| 文件 | 内容 |
|---|---|
| `sim/figure55.xml` | MuJoCo 原模型（所有数值的来源） |
| `sim/figure55.urdf` | 由 `sim/mjcf_to_urdf.py` 生成，质量 2.950 kg、24 个关节，已核对与原模型一致 |
| `sim/isaac/figure55_cfg.py` | Isaac Lab 机器人配置**草稿**（还没有在 Isaac Lab 里运行过） |

如果改了 `figure55.xml`，重新生成 URDF：

```bash
python sim/mjcf_to_urdf.py --check
```

## 2. URDF 转 USD

在 Isaac Lab 目录下运行自带的转换工具（参数名随版本变化，先看 `--help`）：

```bash
./isaaclab.sh -p scripts/tools/convert_urdf.py --help
```

把输出路径设为 `sim/isaac/figure55.usd`。在 Windows 上把 `./isaaclab.sh` 换成 `isaaclab.bat`。

转换后先在 Isaac Sim 里打开 USD 看一眼：人偶是否站着、关节方向对不对、脚是否贴地（骨盆初始高度 0.29 m）。

## 3. 第一个训练任务：平地行走

不要从零写任务，**复制 Isaac Lab 自带的人形机器人平地行走任务**（H1 或 G1 的 flat velocity 任务），然后改这几处：

| 要改的地方 | 改成 |
|---|---|
| 机器人配置 | `FIGURE55_CFG`（来自 `sim/isaac/figure55_cfg.py`） |
| 躯干连杆名 | `chest`（摔倒判断、躯干姿态奖励用） |
| 脚连杆名 | `l_foot`、`r_foot`（着地、抬脚奖励用） |
| 目标速度 | **约 0.1–0.3 m/s**（对照：手写静步行只有约 0.01 m/s）（原任务是为 1.8 m 的机器人设计的；我们的人偶只有 55 cm，速度要按比例缩小） |
| 骨盆目标高度 | 约 0.28 m |
| 摔倒判断 | 骨盆低于约 0.15 m，或躯干倾斜超过约 35° |

建议的训练顺序：
1. **站立**：速度指令全部设为 0，先学会被推时站稳。对照基准：MuJoCo 里手写的平衡控制器能扛住 4 N 的推力
2. **平地行走**：0.1 → 0.3 m/s
3. **受损后的补偿**：训练时随机把某个关节的扭矩上限降到 30–70%（模拟右后膝线缆那样的故障），看它会学出什么样的走法

## 4. 训练出来以后要回答的问题

- 走路时膝关节的持续扭矩和峰值扭矩是多少？是否超过 `docs/knee-joint-v1.md` 的 1.35 / 2.8 N·m？
- 学出来的平衡比手写控制器（`sim/balance.py`）强多少？
- 某个关节变弱时，负载转移到了哪里？和身体地图里四足机器人的故事是否一样？

## 5. 开新会话时的第一句话

> 请先读 `docs/handoff.md` 和 `docs/desktop-isaac.md`，帮我在这台 5070 Ti 上把人偶导入 Isaac Lab，先训练站立。
