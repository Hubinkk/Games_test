"""
训练起身策略（PPO）。在 Colab 的显卡上跑；也可以在没有显卡的机器上用很小的规模试跑。

用法：
    python train_getup.py --steps=100000000 --envs=2048        # Colab T4：约 1–2 小时（估计，没有实测）
    python train_getup.py --smoke                               # 小规模试跑：只检查代码能跑通
产物：
    getup_policy.npz       策略网络的权重和观测的均值、方差（纯数字，不依赖训练库；用 policy_np.py 读）
    getup_progress.json    训练曲线
"""
import functools
import json
import os
import sys
import time

import jax
import numpy as np
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo

from getup_env import GetUp

HERE = os.path.dirname(os.path.abspath(__file__))
HIDDEN = (256, 256, 128)


def arg(name, default):
    for a in sys.argv[1:]:
        if a.startswith(f"--{name}="):
            return type(default)(a.split("=", 1)[1])
    return default


def flatten(tree, prefix=""):
    out = {}
    if isinstance(tree, dict):
        for k, v in tree.items():
            out.update(flatten(v, f"{prefix}{k}/"))
    elif hasattr(tree, "_fields"):
        for k in tree._fields:
            out.update(flatten(getattr(tree, k), f"{prefix}{k}/"))
    elif isinstance(tree, (list, tuple)):
        for i, v in enumerate(tree):
            out.update(flatten(v, f"{prefix}{i}/"))
    elif hasattr(tree, "__dataclass_fields__"):
        for k in tree.__dataclass_fields__:
            out.update(flatten(getattr(tree, k), f"{prefix}{k}/"))
    elif tree is not None:
        out[prefix.rstrip("/")] = np.asarray(tree)
    return out


def fill(tree, flat, prefix=""):
    """flatten 的反过来：按同样的命名把数组填回 tree 的结构里（找不到的保留原样）。"""
    import jax.numpy as jnp
    if isinstance(tree, dict):
        return {k: fill(v, flat, f"{prefix}{k}/") for k, v in tree.items()}
    if hasattr(tree, "_fields"):
        return type(tree)(*[fill(getattr(tree, k), flat, f"{prefix}{k}/") for k in tree._fields])
    if isinstance(tree, (list, tuple)):
        return type(tree)(fill(v, flat, f"{prefix}{i}/") for i, v in enumerate(tree))
    if hasattr(tree, "__dataclass_fields__"):
        return tree.replace(**{k: fill(getattr(tree, k), flat, f"{prefix}{k}/") for k in tree.__dataclass_fields__})
    key = prefix.rstrip("/")
    if tree is None or key not in flat or not hasattr(tree, "shape"):    # 设置项（比如归一化的 mode）不是数组，保持原样
        return tree
    v = flat[key]
    assert np.shape(v) == np.shape(tree), key
    return jnp.asarray(v, dtype=getattr(tree, "dtype", None))


def export(params, path):
    """params = (观测归一化, 策略网络, 价值网络)。只存前两个，摊平成一组数组。"""
    flat = {}
    flat.update(flatten(params[0], "norm/"))
    flat.update(flatten(params[1], "policy/"))
    np.savez(path, **flat)
    return sorted(flat)


def main():
    smoke = "--smoke" in sys.argv
    tag = "_smoke" if smoke else ""        # 试跑的产物另外起名，不和真正的训练结果混在一起
    if arg("tag", ""):                     # --tag=run4：产物写成 getup_policy_run4.npz，不覆盖正在用的 getup_policy.npz
        tag = "_" + arg("tag", "")
    steps = arg("steps", 200_000 if smoke else 100_000_000)
    envs = arg("envs", 16 if smoke else 2048)
    refs = os.path.join(HERE, "ref_getup.json")
    if arg("ref", "") == "turn":
        # --ref=turn（2026-11-12）：头偏向一边。左右两份参考各占一半；起始姿势多样；推 3–6 N、0.2 秒；
        # 地面摩擦 0.5–2.0，头和地面之间单独 0.3–1.0（头用光滑材料）。见 getup_env.py 开头和 make_ref_turn.py
        import getup_env as E
        refs = [os.path.join(HERE, f"ref_getup_turn{s}.json") for s in "LR"]
        E.VARY, E.PUSH_N, E.PUSH_STEPS = True, 5.0, 10
        E.FACE_W = arg("facew", 1.0)
        E.FRIC = tuple(float(x) for x in arg("fric", "0.5,2.0").split(","))
        E.HEAD_FRIC = tuple(float(x) for x in arg("headfric", "0.3,1.0").split(","))
        print(f"转头：地面摩擦 {E.FRIC}，头 {E.HEAD_FRIC}，推 {E.PUSH_N} N × {E.PUSH_STEPS} 步，起始姿势多样", flush=True)
    env = GetUp(os.path.join(HERE, "huaran_rl.xml"), refs)
    # 评估时一律从趴着开始、做完整的一次起身（训练时七成的身体从动作中间开始）
    eval_env = GetUp(os.path.join(HERE, "huaran_rl.xml"), refs, mid_start=False)
    episode = env.T + 25
    log = []
    t0 = time.time()

    def progress(n, metrics):
        row = dict(steps=int(n), minutes=round((time.time() - t0) / 60, 1),
                   reward=round(float(metrics.get("eval/episode_reward", 0.0)), 2),
                   success=round(float(metrics.get("eval/episode_success", 0.0)) / max(env.T - env.t_free, 1), 3),
                   height=round(float(metrics.get("eval/episode_height", 0.0)) / episode * 100, 1),
                   seconds=round(float(metrics.get("eval/avg_episode_length", 0.0)) * env.dt, 1),
                   face_s=round(float(metrics.get("eval/episode_face", 0.0)) * env.dt, 2))
        log.append(row)
        print(f"{row['steps']:>11,} 步  {row['minutes']:>6} 分钟  得分 {row['reward']:>7}  松手后站着的时间比例 {row['success']:.0%}  "
              f"平均骨盆高 {row['height']} cm  撑了 {row['seconds']} 秒才倒（满分 {episode * env.dt:.1f}）  脸着地 {row['face_s']} 秒", flush=True)
        json.dump(log, open(os.path.join(HERE, f"getup_progress{tag}.json"), "w"), indent=1)

    net = functools.partial(ppo_networks.make_ppo_networks, policy_hidden_layer_sizes=HIDDEN, value_hidden_layer_sizes=(256, 256, 256))
    restore = None
    if arg("init", ""):
        # --init=run3_final.npz：从一份已有的策略接着练（观测归一化和策略网络照搬，价值网络重新开始）
        _, tmpl, _ = ppo.train(environment=env, num_timesteps=0, num_envs=envs, episode_length=episode,
                               batch_size=256, num_minibatches=32, normalize_observations=True, network_factory=net,
                               seed=arg("seed", 0))
        z = np.load(os.path.join(HERE, arg("init", "")))
        flat = {k: z[k] for k in z.files}
        restore = (fill(tmpl[0], flat, "norm/"), fill(tmpl[1], flat, "policy/"), tmpl[2])
        print(f"从 {arg('init', '')} 接着练", flush=True)
    make_inference, params, _ = ppo.train(
        restore_params=restore, restore_value_fn=False,
        environment=env, eval_env=eval_env, num_timesteps=steps, num_envs=envs, episode_length=episode,
        num_evals=3 if smoke else 21, num_eval_envs=8 if smoke else 128, deterministic_eval=True,
        learning_rate=3e-4, entropy_cost=5e-3, discounting=arg("gamma", 0.99), gae_lambda=0.95, unroll_length=20,
        batch_size=16 if smoke else 256, num_minibatches=4 if smoke else 32, num_updates_per_batch=4,
        normalize_observations=True, reward_scaling=1.0, clipping_epsilon=0.2, max_grad_norm=1.0,
        network_factory=net, seed=arg("seed", 0), progress_fn=progress,
        # 每次评估都把当时的策略存一份，训练中途停下也有东西可以检验
        # 用了 --tag 时，每次评估另外留一份带步数的（getup_policy_run4_10M.npz），中途的也能拿去检验
        policy_params_fn=lambda n, make_policy, p: (export(p, os.path.join(HERE, f"getup_policy{tag}.npz")),
                                                    arg("tag", "") and export(p, os.path.join(HERE, f"getup_policy{tag}_{int(n) // 1_000_000}M.npz"))))
    # 注意：最后一份不一定是最好的。第二轮里 1000 万步那份在完整身体上是 4/5，1 亿步那份是 3/5。
    keys = export(params, os.path.join(HERE, f"getup_policy{tag}.npz"))
    print(f"已保存 getup_policy{tag}.npz：", len(keys), "个数组")
    return make_inference, params, env


if __name__ == "__main__":
    main()
