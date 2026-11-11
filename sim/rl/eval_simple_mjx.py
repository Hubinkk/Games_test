"""
在训练用的简化身体（MJX，显卡）上按地面摩擦分开检验策略：每种摩擦 N 个身体，都从趴着开始做一整次起身，
手的软硬照训练时一样每个身体随机。用来区分"简化身体上就没学会"还是"简化身体会、完整身体不会"。

用法（训练环境）：
    .venv/bin/python eval_simple_mjx.py 策略.npz [--n=128] [--fric=0.6,1.0,1.2,1.5,1.8]
"""
import json
import math
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

import getup_env

HERE = os.path.dirname(os.path.abspath(__file__))


def arg(name, default):
    for a in sys.argv[1:]:
        if a.startswith(f"--{name}="):
            return a.split("=", 1)[1]
    return default


def load_policy(path):
    z = np.load(path)
    mean, std = jnp.array(z["norm/mean"]), jnp.array(z["norm/std"])
    n = len([k for k in z.files if k.startswith("policy/params/hidden_") and k.endswith("/kernel")])
    layers = [(jnp.array(z[f"policy/params/hidden_{i}/kernel"]), jnp.array(z[f"policy/params/hidden_{i}/bias"])) for i in range(n)]
    n_act = layers[-1][0].shape[1] // 2

    def pol(obs):
        x = (obs - mean) / std
        for i, (W, b) in enumerate(layers):
            x = x @ W + b
            if i < len(layers) - 1:
                x = jax.nn.silu(x)
        return jnp.tanh(x[..., :n_act])
    return pol


def main():
    path = [a for a in sys.argv[1:] if not a.startswith("--")][0]
    n = int(arg("n", 128))
    frics = [float(x) for x in arg("fric", "0.6,1.0,1.2,1.5,1.8").split(",")]
    pol = load_policy(path)
    out = {}
    for f in frics:
        getup_env.FRIC = (f, f)
        env = getup_env.GetUp(os.path.join(HERE, "huaran_rl.xml"), os.path.join(HERE, "ref_getup.json"), mid_start=False)
        steps = env.T + 25

        def one(key):
            s = env.reset(key)

            def body(c, _):
                s, alive = c
                s2 = env.step(s, pol(s.obs))
                alive2 = alive & (s2.done < 0.5)
                return (s2, alive2), None
            (s, alive), _ = jax.lax.scan(body, (s, jnp.array(True)), None, length=steps)
            z, up, _ = env._measure(s.pipeline_state)
            ok = alive & (z > 0.20) & (up > math.cos(math.radians(20.0)))
            return ok, alive, z
        keys = jax.random.split(jax.random.PRNGKey(1), n)
        ok, alive, z = jax.jit(jax.vmap(one))(keys)
        out[f] = dict(ok=int(ok.sum()), alive=int(alive.sum()), n=n, z_cm=round(float(jnp.mean(z)) * 100, 1))
        print(f"摩擦 {f}：{out[f]['ok']}/{n} 站起来（没被判倒的 {out[f]['alive']}，平均骨盆 {out[f]['z_cm']} cm）", flush=True)
    json.dump(dict(policy=os.path.basename(path), result=out),
              open(os.path.join(HERE, f"eval_simple_mjx_{os.path.splitext(os.path.basename(path))[0]}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
