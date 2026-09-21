#!/bin/bash
# 对应 run_cifar10_baselines.ps1：把 CIFAR-10 的 14 组 baseline 并行提交到 Slurm。
# 默认分区 debug 全是 RTX 4090，并用 --gres=gpu:rtx_4090:1 钉死卡型。
#
# 用法（在登录节点执行，不要在计算节点上再 sbatch）：
#   bash run_cifar10_baselines.sh
#   bash run_cifar10_baselines.sh --dry-run
# 查看 / 取消：
#   squeue -u "$USER"
#   scancel <jobid>

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

ACCOUNT="duanty"
PARTITION="debug"
GRES="gpu:rtx_4090:1"
CPUS=4
TIME="2-00:00:00"
CONDA_ENV="moe"
CONDA_ROOT="${HOME}/miniconda3"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
fi

JOBS=(
    "c10-resnet-ours-ours|--method ours --dataset cifar10 --backbone resnet --select ours --merge ours --device cuda"
    "c10-resnet-posthoc-ours|--method ours --dataset cifar10 --backbone resnet --select posthoc --merge ours --device cuda"
    "c10-resnet-fedrolex-ours|--method ours --dataset cifar10 --backbone resnet --select fedrolex --merge ours --device cuda"
    "c10-resnet-snip-ours|--method ours --dataset cifar10 --backbone resnet --select snip --merge ours --device cuda"
    "c10-resnet-ours-fedavg-old|--method ours --dataset cifar10 --backbone resnet --select ours --merge fedavg-old --device cuda"
    "c10-resnet-ours-fedavgm|--method ours --dataset cifar10 --backbone resnet --select ours --merge fedavgm --device cuda"
    "c10-resnet-ours-fedadam|--method ours --dataset cifar10 --backbone resnet --select ours --merge fedadam --device cuda"
    "c10-vgg11-ours-ours|--method ours --dataset cifar10 --backbone vgg11 --select ours --merge ours --device cuda"
    "c10-vgg11-posthoc-ours|--method ours --dataset cifar10 --backbone vgg11 --select posthoc --merge ours --device cuda"
    "c10-vgg11-fedrolex-ours|--method ours --dataset cifar10 --backbone vgg11 --select fedrolex --merge ours --device cuda"
    "c10-vgg11-snip-ours|--method ours --dataset cifar10 --backbone vgg11 --select snip --merge ours --device cuda"
    "c10-vgg11-ours-fedavg-old|--method ours --dataset cifar10 --backbone vgg11 --select ours --merge fedavg-old --device cuda"
    "c10-vgg11-ours-fedavgm|--method ours --dataset cifar10 --backbone vgg11 --select ours --merge fedavgm --device cuda"
    "c10-vgg11-ours-fedadam|--method ours --dataset cifar10 --backbone vgg11 --select ours --merge fedadam --device cuda"
)

mkdir -p logs

submit_one() {
    local name="$1"
    local args="$2"
    local job_script
    job_script="$(mktemp "${TMPDIR:-/tmp}/cifar10_bl.XXXXXX.sh")"

    cat > "$job_script" <<EOF
#!/bin/bash
#SBATCH -J ${name}
#SBATCH -A ${ACCOUNT}
#SBATCH -p ${PARTITION}
#SBATCH --gres=${GRES}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --time=${TIME}
#SBATCH --chdir=${ROOT}
#SBATCH -o logs/${name}_%j.out
#SBATCH -e logs/${name}_%j.err

set -euo pipefail
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate ${CONDA_ENV}
export PYTHONUNBUFFERED=1

echo "========================================"
echo "Job    : ${name}"
echo "Host   : \$(hostname)"
echo "Date   : \$(date)"
echo "GPU    : \${CUDA_VISIBLE_DEVICES:-unset}"
echo "Python : \$(which python)"
echo "Command: python -u main.py ${args}"
echo "========================================"

python -u main.py ${args}
EOF

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "[dry-run] sbatch  ${name}  python -u main.py ${args}"
        rm -f "$job_script"
        return
    fi

    local jobid
    jobid="$(sbatch --parsable "$job_script")"
    rm -f "$job_script"
    echo "submitted  ${jobid}  ${name}  python -u main.py ${args}"
}

echo "提交 ${#JOBS[@]} 个并行任务 -> 分区=${PARTITION}, GRES=${GRES}, 账号=${ACCOUNT}"
echo

for spec in "${JOBS[@]}"; do
    name="${spec%%|*}"
    args="${spec#*|}"
    submit_one "$name" "$args"
done

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo
    echo "这是 dry-run，没有真正提交。去掉 --dry-run 再执行即可。"
else
    echo
    echo "查看队列: squeue -u ${USER}"
    echo "日志目录: ${ROOT}/logs"
fi
