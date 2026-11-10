#!/usr/bin/env bash
# 在台式机（RTX 5070 Ti）的 Linux 环境里准备训练环境并检查显卡。
# Ubuntu 或 Windows 里的 WSL2 都可以。在仓库的 sim/rl 目录下运行：
#     bash setup_desktop.sh
# 它只在当前目录下建一个 .venv 文件夹，不改系统里的其它东西。
set -e
cd "$(dirname "$0")"

echo "== 1. 显卡驱动 =="
if ! command -v nvidia-smi >/dev/null; then
  echo "找不到 nvidia-smi：Linux 环境里看不到显卡。"
  echo "如果是 WSL2：在 Windows 里装好最新的 NVIDIA 驱动，然后在 PowerShell 里运行 wsl --update，再重开这个窗口。"
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "== 2. Python =="
PY=python3
$PY --version
$PY -m venv .venv 2>/dev/null || { echo "建不了虚拟环境。先运行：sudo apt install -y python3-venv python3-pip"; exit 1; }
. .venv/bin/activate
pip install -q --upgrade pip

echo "== 3. 安装训练工具（这组版本在没有显卡的机器上试跑通过）=="
pip install -q "jax[cuda12]==0.7.2" "mujoco-mjx==4.0.0" "brax==0.14.2"

echo "== 4. 检查 JAX 看不看得到显卡 =="
python - <<'PYEOF'
import jax
devs = jax.devices()
print("JAX", jax.__version__, "设备：", devs)
if not any("cuda" in str(d).lower() or "gpu" in str(d).lower() for d in devs):
    raise SystemExit("JAX 没有用上显卡。把上面这几行输出发给 Claude。")
import jax.numpy as jnp
x = jnp.ones((4000, 4000))
print("显卡上的矩阵乘法：", float((x @ x).sum()))
PYEOF

echo
echo "准备好了。开始训练："
echo "    . .venv/bin/activate && python train_getup.py --steps=100000000 --envs=4096"
