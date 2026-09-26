#!/bin/bash
# exp11：CIFAR-100 / ResNet，紧预算（K=1–3）下探测批次与全量解的一致率。
# 训练设置与主实验 Ours 相同：8 专家、Top-2、10 客户端、Dirichlet 0.1、
# 100 轮、本地 2 epoch、带宽 [10, 80]、聚合 ours。
# 测量写在本地训练之前，不改变选择和上传。
#
# 用法（在登录节点执行，不要在计算节点上再 sbatch）：
#   bash exp11_c100_resnet_probe_agreement.sh
#   bash exp11_c100_resnet_probe_agreement.sh --dry-run

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

EXP_ID="exp11"
ACCOUNT="duanty"
PARTITION="debug"
GRES="gpu:rtx_4090:1"
CPUS=4
TIME="2-00:00:00"
CONDA_ENV="moe"
CONDA_ROOT="${HOME}/miniconda3"
DATASET="cifar100"
BACKBONE="resnet"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
fi

mkdir -p logs

name="${EXP_ID}-c100-resnet-probe-agreement"
args="--dataset ${DATASET} --backbone ${BACKBONE} --method ours --select ours --merge ours --device cuda --num_experts 8 --num_clients 10 --rounds 100 --bw_min 10 --bw_max 80 --k_min 1 --k_max 3 --output_dir ./results/${name}"

job_script="$(mktemp "${TMPDIR:-/tmp}/${EXP_ID}.XXXXXX.sh")"
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
echo "Command: python -u probe_agreement.py ${args}"
echo "========================================"

python -u probe_agreement.py ${args}
EOF

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] sbatch  ${name}  python -u probe_agreement.py ${args}"
    rm -f "$job_script"
    exit 0
fi

jobid="$(sbatch --parsable "$job_script")"
rm -f "$job_script"
echo "submitted  ${jobid}  ${name}  python -u probe_agreement.py ${args}"
echo
echo "查看队列: squeue -u ${USER}"
echo "日志目录: ${ROOT}/logs"
