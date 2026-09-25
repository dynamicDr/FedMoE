#!/bin/bash
# exp10：CIFAR-100 Post-hoc，32 专家，两个带宽、两个骨干。
# 本地训练对全部专家做前向和反传，再按门控分数只上传 K 个。
# 其余与 exp09 相同：Top-2 路由、10 客户端、Dirichlet 0.1、100 轮、
# 本地 2 epoch、聚合 ours。只跑 ResNet 和 VGG11。
#
#   [1, 40]  --bw_min 1 --bw_max 40
#   [1, 80]  --bw_min 1 --bw_max 80
#
# 用法（在登录节点执行，不要在计算节点上再 sbatch）：
#   bash exp10_c100_posthoc_e32_bw.sh
#   bash exp10_c100_posthoc_e32_bw.sh --dry-run

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

EXP_ID="exp10"
ACCOUNT="duanty"
PARTITION="debug"
GRES="gpu:rtx_4090:1"
CPUS=4
TIME="2-00:00:00"
CONDA_ENV="moe"
CONDA_ROOT="${HOME}/miniconda3"
NUM_EXPERTS=32
DATASET="cifar100"
SELECT="posthoc"
MERGE="ours"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
fi

# tag|backbone|bw_min|bw_max
JOBS=(
    "resnet|resnet|1|40"
    "vgg11|vgg11|1|40"
    "resnet|resnet|1|80"
    "vgg11|vgg11|1|80"
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
    IFS='|' read -r btag backbone bw_min bw_max <<< "$jspec"
    name="${EXP_ID}-c100-${btag}-e${NUM_EXPERTS}-bw${bw_min}-${bw_max}-posthoc"
    args="--method ours --dataset ${DATASET} --backbone ${backbone} --select ${SELECT} --merge ${MERGE} --device cuda --num_experts ${NUM_EXPERTS} --bw_min ${bw_min} --bw_max ${bw_max} --output_dir ./results/${name}"
    submit_one "$name" "$args"
    n_jobs=$((n_jobs + 1))
done

echo
echo "共 ${n_jobs} 个任务（CIFAR-100 / Post-hoc / 32 专家 / 带宽 [1, 40] 与 [1, 80]：ResNet, VGG11）-> 分区=${PARTITION}, GRES=${GRES}, 账号=${ACCOUNT}, 时限=${TIME}"

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo
    echo "这是 dry-run，没有真正提交。去掉 --dry-run 再执行即可。"
else
    echo
    echo "查看队列: squeue -u ${USER}"
    echo "日志目录: ${ROOT}/logs"
fi
