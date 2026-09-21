#!/bin/bash
# CIFAR-10 / ResNet 扫参，只跑一种方法：
#   Random + Ours        --select random --merge ours
#
# 网格：专家数 {4, 8, 16} × 骨干宽度系数 {0.5, 1} = 6 组配置。
# 每种配置提交一次，共 6 个任务。
# width_mult 只缩放骨干通道（0.5: 32-64-128-256；1: 64-128-256-512），
# 专家 FFN 隐层保持 512。
# --bw_max -1：带宽上界 = num_experts * expert_bw，使 K 仍能到 M。
#
# 用法（在登录节点执行，不要在计算节点上再 sbatch）：
#   bash run_cifar10_random_ours_sweep.sh
#   bash run_cifar10_random_ours_sweep.sh --dry-run
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

EXPERTS=(4 8 16)
WIDTHS=(0.5 1)
# tag|select|merge
METHODS=(
    "random|random|ours"
)

mkdir -p logs

submit_one() {
    local name="$1"
    local args="$2"
    local job_script
    job_script="$(mktemp "${TMPDIR:-/tmp}/cifar10_sweep.XXXXXX.sh")"

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

n_jobs=0
for nexp in "${EXPERTS[@]}"; do
    for width in "${WIDTHS[@]}"; do
        wtag="${width/./p}"
        for spec in "${METHODS[@]}"; do
            IFS='|' read -r tag select merge <<< "$spec"
            name="c10-e${nexp}-w${wtag}-${tag}"
            args="--method ours --dataset cifar10 --backbone resnet --select ${select} --merge ${merge} --device cuda --num_experts ${nexp} --width_mult ${width} --bw_max -1 --output_dir ./results/${name}"
            submit_one "$name" "$args"
            n_jobs=$((n_jobs + 1))
        done
    done
done

echo
echo "共 ${n_jobs} 个任务（6 组配置 × 1 种方法）-> 分区=${PARTITION}, GRES=${GRES}, 账号=${ACCOUNT}"

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo
    echo "这是 dry-run，没有真正提交。去掉 --dry-run 再执行即可。"
else
    echo
    echo "查看队列: squeue -u ${USER}"
    echo "日志目录: ${ROOT}/logs"
fi
