"""
训练走路策略（PPO，MJX）。

用法（sim/rl/.venv，要显卡）：
    python build_walk_model.py                                   # 先生成 huaran_walk.xml
    python train_walk.py --tag=w1 --steps=150000000 --envs=4096
    python train_walk.py --smoke                                 # 小规模试跑：只检查代码能跑通
    python train_walk.py --tag=w2 --init=walk_policy_w1.npz      # 从一份已有的策略接着练
产物：
    walk_policy_<tag>.npz、walk_policy_<tag>_<步数>M.npz        每次评估存一份（用 policy_np.py 读；最后一份不一定最好）
    walk_progress_<tag>.json                                    训练曲线
"""
import functools
import json
import os
import sys
import time

import numpy as np
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo

from train_getup import arg, fill, flatten
from walk_env import Walk

HERE = os.path.dirname(os.path.abspath(__file__))
HIDDEN = (512, 256, 128)


def export(params, path):
    """只存策略用得到的：观测（state 那一份）的均值和方差、策略网络。名字和起身的策略一样，policy_np.py 直接能读。"""
    flat = {}
    for k, v in flatten(params[0], "norm/").items():
        if k.endswith("/state"):
            flat[k[:-len("/state")]] = v
    flat.update(flatten(params[1], "policy/"))
    np.savez(path, **flat)
    return sorted(flat)


def main():
    smoke = "--smoke" in sys.argv
    tag = "_smoke" if smoke else "_" + arg("tag", "w0")
    steps = arg("steps", 300_000 if smoke else 150_000_000)
    envs = arg("envs", 32 if smoke else 4096)
    xml = os.path.join(HERE, "huaran_walk.xml")
    env = Walk(xml)
    eval_env = Walk(xml)
    episode = 1000                         # 20 秒
    log = []
    t0 = time.time()

    def progress(n, metrics):
        L = max(float(metrics.get("eval/avg_episode_length", 1.0)), 1.0)
        row = dict(steps=int(n), minutes=round((time.time() - t0) / 60, 1),
                   reward=round(float(metrics.get("eval/episode_reward", 0.0)), 2),
                   seconds=round(L * env.dt, 1),
                   err_cm_s=round(float(metrics.get("eval/episode_err", 0.0)) / L * 100, 1),
                   werr=round(float(metrics.get("eval/episode_werr", 0.0)) / L, 2),
                   lift_cm=round(float(metrics.get("eval/episode_lift", 0.0)) / L * 100, 1))
        log.append(row)
        print(f"{row['steps']:>11,} 步  {row['minutes']:>6} 分钟  得分 {row['reward']:>6}  撑了 {row['seconds']} 秒（满 20）  "
              f"速度差 {row['err_cm_s']} cm/s  转速差 {row['werr']} rad/s  较高那只脚平均离地 {row['lift_cm']} cm", flush=True)
        json.dump(log, open(os.path.join(HERE, f"walk_progress{tag}.json"), "w"), indent=1)

    net = functools.partial(ppo_networks.make_ppo_networks, policy_hidden_layer_sizes=HIDDEN,
                            value_hidden_layer_sizes=(512, 256, 128), policy_obs_key="state", value_obs_key="privileged_state")
    common = dict(episode_length=episode, batch_size=32 if smoke else 256, num_minibatches=4 if smoke else 32,
                  normalize_observations=True, network_factory=net, seed=arg("seed", 0))
    restore = None
    if arg("init", ""):
        _, tmpl, _ = ppo.train(environment=env, num_timesteps=0, num_envs=envs, **common)
        z = np.load(os.path.join(HERE, arg("init", "")))
        flat = {k: z[k] for k in z.files}
        for k in ("norm/mean", "norm/std", "norm/summed_variance", "norm/count"):       # 存的时候去掉了 /state，这里加回去
            if k in flat:
                flat[k + "/state"] = flat[k]
        restore = (fill(tmpl[0], flat, "norm/"), fill(tmpl[1], flat, "policy/"), tmpl[2])
        print(f"从 {arg('init', '')} 接着练", flush=True)

    def save(n, make_policy, p):
        export(p, os.path.join(HERE, f"walk_policy{tag}.npz"))
        if not smoke:
            export(p, os.path.join(HERE, f"walk_policy{tag}_{int(n) // 1_000_000}M.npz"))

    _, params, _ = ppo.train(
        restore_params=restore, restore_value_fn=False,
        environment=env, eval_env=eval_env, num_timesteps=steps, num_envs=envs,
        num_evals=3 if smoke else arg("evals", 16), num_eval_envs=16 if smoke else 256, deterministic_eval=True,
        learning_rate=arg("lr", 3e-4), entropy_cost=arg("ent", 5e-3), discounting=arg("gamma", 0.97), gae_lambda=0.95, unroll_length=20,
        num_updates_per_batch=4, reward_scaling=1.0, clipping_epsilon=0.2, max_grad_norm=1.0,
        progress_fn=progress, policy_params_fn=save, **common)
    keys = export(params, os.path.join(HERE, f"walk_policy{tag}.npz"))
    print(f"已保存 walk_policy{tag}.npz：", len(keys), "个数组")


if __name__ == "__main__":
    main()
