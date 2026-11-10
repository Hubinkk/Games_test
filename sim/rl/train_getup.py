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
    steps = arg("steps", 200_000 if smoke else 100_000_000)
    envs = arg("envs", 16 if smoke else 2048)
    env = GetUp(os.path.join(HERE, "huaran_rl.xml"), os.path.join(HERE, "ref_getup.json"))
    eval_env = GetUp(os.path.join(HERE, "huaran_rl.xml"), os.path.join(HERE, "ref_getup.json"))
    episode = env.T + 25
    log = []
    t0 = time.time()

    def progress(n, metrics):
        row = dict(steps=int(n), minutes=round((time.time() - t0) / 60, 1),
                   reward=round(float(metrics.get("eval/episode_reward", 0.0)), 2),
                   success=round(float(metrics.get("eval/episode_success", 0.0)) / max(env.T - env.t_free, 1), 3),
                   height=round(float(metrics.get("eval/episode_height", 0.0)) / episode * 100, 1))
        log.append(row)
        print(f"{row['steps']:>11,} 步  {row['minutes']:>6} 分钟  得分 {row['reward']:>7}  松手后站着的时间比例 {row['success']:.0%}  平均骨盆高 {row['height']} cm", flush=True)
        json.dump(log, open(os.path.join(HERE, f"getup_progress{tag}.json"), "w"), indent=1)

    net = functools.partial(ppo_networks.make_ppo_networks, policy_hidden_layer_sizes=HIDDEN, value_hidden_layer_sizes=(256, 256, 256))
    make_inference, params, _ = ppo.train(
        environment=env, eval_env=eval_env, num_timesteps=steps, num_envs=envs, episode_length=episode,
        num_evals=3 if smoke else 21, num_eval_envs=8 if smoke else 128, deterministic_eval=True,
        learning_rate=3e-4, entropy_cost=5e-3, discounting=0.97, gae_lambda=0.95, unroll_length=20,
        batch_size=16 if smoke else 256, num_minibatches=4 if smoke else 32, num_updates_per_batch=4,
        normalize_observations=True, reward_scaling=1.0, clipping_epsilon=0.2, max_grad_norm=1.0,
        network_factory=net, seed=arg("seed", 0), progress_fn=progress)
    keys = export(params, os.path.join(HERE, f"getup_policy{tag}.npz"))
    print(f"已保存 getup_policy{tag}.npz：", len(keys), "个数组")
    return make_inference, params, env


if __name__ == "__main__":
    main()
