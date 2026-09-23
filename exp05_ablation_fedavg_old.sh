#!/bin/bash
# exp05：Ablation 全部条件改用 FedAvg-old，并加一档少传专家。
# 数据集 / 骨干 / 专家数 / 终端数 / 带宽与 exp03、exp04 的消融一致。
# 选择器仍是 ours，只把聚合从 fedavg（缺失专家填 0）换成 fedavg-old（跳过未上传专家）。
# 每个条件两档：
#   fedavg-old        --merge fedavg-old
#   fedavg-old-drop25 --merge fedavg-old --drop_expert_prob 0.25
# drop25：每个客户端每一轮，在带宽给出的 K 上以 25% 概率再少传 1 个专家，K 不低于 0。
# 10 个条件 × 2 档 = 20 个任务。
# 默认分区 debug 全是 RTX 4090，并用 --gres=gpu:rtx_4090:1 钉死卡型。
#
# 条件（CIFAR-100 / ResNet）：
#   bw [10, 40]、[1, 50]、[20, 80]、[40, 80]，均为 8 专家、10 终端
#   4 / 16 / 32 专家（--bw_max -1，10 终端）
#   20 / 30 / 50 终端（8 专家，带宽 [10, 80]）
#
# 用法（在登录节点执行，不要在计算节点上再 sbatch）：
#   bash exp05_ablation_fedavg_old.sh
#   bash exp05_ablation_fedavg_old.sh --dry-run
# 查看 / 取消：
#   squeue -u "$USER"
#   scancel <jobid>

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

EXP_ID="exp05"
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

# tag|select|merge|extra
METHODS=(
    "fedavg-old|ours|fedavg-old|"
    "fedavg-old-drop25|ours|fedavg-old|--drop_expert_prob 0.25"
)

# tag|num_experts|num_clients|bw_min|bw_max
# bw_max=-1 表示上界 = num_experts * expert_bw
SETTINGS=(
    "bw10-40|8|10|10|40"
    "bw1-50|8|10|1|50"
    "bw20-80|8|10|20|80"
    "bw40-80|8|10|40|80"
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
    for mspec in "${METHODS[@]}"; do
        IFS='|' read -r mtag select merge extra <<< "$mspec"
        name="${EXP_ID}-c100-resnet-${stag}-${mtag}"
        args="--method ours --dataset ${DATASET} --backbone ${BACKBONE} --select ${select} --merge ${merge} --device cuda --num_experts ${num_experts} --num_clients ${num_clients} --bw_min ${bw_min} --bw_max ${bw_max} --output_dir ./results/${name}"
        if [[ -n "${extra}" ]]; then
            args="${args} ${extra}"
        fi
        submit_one "$name" "$args"
        n_jobs=$((n_jobs + 1))
    done
done

echo
echo "共 ${n_jobs} 个任务（10 个消融条件 × FedAvg-old / FedAvg-old+drop25）-> 分区=${PARTITION}, GRES=${GRES}, 账号=${ACCOUNT}"

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo
    echo "这是 dry-run，没有真正提交。去掉 --dry-run 再执行即可。"
else
    echo
    echo "查看队列: squeue -u ${USER}"
    echo "日志目录: ${ROOT}/logs"
fi
