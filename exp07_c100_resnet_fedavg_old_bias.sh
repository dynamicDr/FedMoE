#!/bin/bash
# exp07：FedMoE 选择（ours）+ FedAvg-old 的随机专家权重。
# 缺的专家仍然跳过，不填 0。共享参数仍按样本量平均。
# 每个专家、每个上传它的客户端，权重是 n_samples * U(0,1)，每轮重抽。
# 只跑消融总图里的 9 个场景（CIFAR-100 / ResNet），不含 bw [40, 80]。
#
#   bw [10, 40]、[1, 50]、[20, 80]，8 专家、10 终端
#   4 / 16 / 32 专家（--bw_max -1，10 终端）
#   20 / 30 / 50 终端（8 专家，带宽 [10, 80]）
#
# 用法（在登录节点执行，不要在计算节点上再 sbatch）：
#   bash exp07_c100_resnet_fedavg_old_bias.sh
#   bash exp07_c100_resnet_fedavg_old_bias.sh --dry-run

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

EXP_ID="exp07"
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

# tag|num_experts|num_clients|bw_min|bw_max
# bw_max=-1 表示上界 = num_experts * expert_bw
SETTINGS=(
    "bw10-40|8|10|10|40"
    "bw1-50|8|10|1|50"
    "bw20-80|8|10|20|80"
    "e4|4|10|10|-1"
    "e16|16|10|10|-1"
    "e32|32|10|10|-1"
    "n20|8|20|10|-1"
    "n30|8|30|10|-1"
    "n50|8|50|10|-1"
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
for sspec in "${SETTINGS[@]}"; do
    IFS='|' read -r stag num_experts num_clients bw_min bw_max <<< "$sspec"
    name="${EXP_ID}-c100-resnet-${stag}-fedavg-old-bias"
    args="--method ours --dataset ${DATASET} --backbone ${BACKBONE} --select ours --merge fedavg-old-bias --device cuda --num_experts ${num_experts} --num_clients ${num_clients} --bw_min ${bw_min} --bw_max ${bw_max} --output_dir ./results/${name}"
    submit_one "$name" "$args"
    n_jobs=$((n_jobs + 1))
done

echo
echo "共 ${n_jobs} 个任务（9 个消融场景 × FedAvg-old 随机专家权重）-> 分区=${PARTITION}, GRES=${GRES}, 账号=${ACCOUNT}"

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo
    echo "这是 dry-run，没有真正提交。去掉 --dry-run 再执行即可。"
else
    echo
    echo "查看队列: squeue -u ${USER}"
    echo "日志目录: ${ROOT}/logs"
fi
