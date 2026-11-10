"""
用纯 numpy 读训练好的策略（getup_policy.npz）并算出动作。不依赖训练库，所以可以放进整体模型的仿真里，
以后也能放到机器人的控制器上。

    pol = Policy("getup_policy.npz")
    action = pol(obs)          # obs：113 个数；action：26 个数，范围 −1 到 1

网络：观测先减均值、除以标准差 → 3 层（256、256、128，激活函数 SiLU）→ 输出 52 个数，前 26 个取 tanh 就是动作。
"""
import numpy as np


class Policy:
    def __init__(self, path):
        z = np.load(path)
        self.mean, self.std = z["norm/mean"].astype(np.float64), z["norm/std"].astype(np.float64)
        n = len([k for k in z.files if k.startswith("policy/params/hidden_") and k.endswith("/kernel")])
        self.layers = [(z[f"policy/params/hidden_{i}/kernel"].astype(np.float64), z[f"policy/params/hidden_{i}/bias"].astype(np.float64))
                       for i in range(n)]
        self.n_act = self.layers[-1][0].shape[1] // 2

    def __call__(self, obs):
        x = (np.asarray(obs, dtype=np.float64) - self.mean) / self.std
        for i, (W, b) in enumerate(self.layers):
            x = x @ W + b
            if i < len(self.layers) - 1:
                x = x / (1.0 + np.exp(-x))          # SiLU：x · sigmoid(x)
        return np.tanh(x[..., :self.n_act])
