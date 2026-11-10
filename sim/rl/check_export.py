"""检查：导出成 .npz 再用 numpy 算出来的动作，和训练库自己算的是否一样。  用法： python check_export.py"""
import os
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
from brax.training.acme import running_statistics, specs
from brax.training.agents.ppo import networks as ppo_networks

from policy_np import Policy
from train_getup import HIDDEN, export

OBS, ACT = 113, 26
net = ppo_networks.make_ppo_networks(OBS, ACT, preprocess_observations_fn=running_statistics.normalize,
                                     policy_hidden_layer_sizes=HIDDEN, value_hidden_layer_sizes=(256, 256, 256))
key = jax.random.PRNGKey(0)
norm = running_statistics.init_state(specs.Array((OBS,), jnp.dtype("float32")))
norm = running_statistics.update(norm, jax.random.normal(key, (500, OBS)) * 3.0 + 1.0)
pp = net.policy_network.init(jax.random.PRNGKey(1))
infer = ppo_networks.make_inference_fn(net)((norm, pp, None), deterministic=True)
obs = jax.random.normal(jax.random.PRNGKey(2), (20, OBS)) * 2.0
a_jax = np.asarray(jax.vmap(lambda o: infer(o, jax.random.PRNGKey(3))[0])(obs))
path = os.path.join(tempfile.mkdtemp(), "p.npz")
export((norm, pp, None), path)
a_np = Policy(path)(np.asarray(obs))
err = float(np.abs(a_jax - a_np).max())
print(f"训练库和 numpy 算出的动作最大相差 {err:.2e}（20 组观测 × 26 个电机）")
assert err < 1e-4, "导出的策略和训练库不一致"
print("一致。")
