#!/bin/bash
# exp08：只补 CIFAR-100 上还没跑完的 Post-hoc（先训练全部专家，再按门控分数上传 Top-K）。
# Ours / Random 已在 exp01 跑满 100 轮，这里不再重复。
# 其余参数与 exp01 主实验相同：8 专家、Top-2、10 客户端、Dirichlet 0.1、
# 100 轮、本地 2 epoch、带宽 [10, 80]（--bw_max -1）、聚合均为 ours。
#
#   ResNet  --backbone resnet --select posthoc --merge ours
#   VGG11   --backbone vgg11  --select posthoc --merge ours
#
# Post-hoc 的 after_train 会先跑完整本地 SGD（不冻结任何专家），再做选择，
# 因此本地训练时间长于 Ours / Random。时限给到分区允许的 2 天。
#
# 用法（在登录节点执行，不要在计算节点上再 sbatch）：
#   bash exp08_c100_posthoc.sh
#   bash exp08_c100_posthoc.sh --dry-run
# 查看 / 取消：
#   squeue -u "$USER"
#   scancel <jobid>

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

EXP_ID="exp08"
ACCOUNT="duanty"
PARTITION="debug"
GRES="gpu:rtx_4090:1"
CPUS=4
TIME="2-00:00:00"
CONDA_ENV="moe"
CONDA_ROOT="${HOME}/miniconda3"
NUM_EXPERTS=8
DATASET="cifar100"
SELECT="posthoc"
MERGE="ours"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
fi

# tag|backbone
JOBS=(
    "resnet|resnet"
    "vgg11|vgg11"
)

mkdir -p logs

submit_one() {
    local name="$1"
    local args="$2"
    local job_script
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
for jspec in "${JOBS[@]}"; do
    IFS='|' read -r btag backbone <<< "$jspec"
    name="${EXP_ID}-c100-${btag}-e${NUM_EXPERTS}-posthoc"
    args="--method ours --dataset ${DATASET} --backbone ${backbone} --select ${SELECT} --merge ${MERGE} --device cuda --num_experts ${NUM_EXPERTS} --bw_max -1 --output_dir ./results/${name}"
    submit_one "$name" "$args"
    n_jobs=$((n_jobs + 1))
done

echo
echo "共 ${n_jobs} 个任务（CIFAR-100 / Post-hoc / 8 专家：ResNet, VGG11）-> 分区=${PARTITION}, GRES=${GRES}, 账号=${ACCOUNT}, 时限=${TIME}"

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo
    echo "这是 dry-run，没有真正提交。去掉 --dry-run 再执行即可。"
else
    echo
    echo "查看队列: squeue -u ${USER}"
    echo "日志目录: ${ROOT}/logs"
fi
