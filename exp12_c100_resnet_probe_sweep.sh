#!/bin/bash
# exp12：CIFAR-100 / ResNet 的探测一致率参数扫描，共 20 组。
# 数据集和骨干固定。变的是半径、上传预算、batch、专家数、Dirichlet 和从哪一轮开始计入图。
# 正在跑的 exp11 是半径 scale=1 的对照，不在这 20 组里。
#
# 用法（在登录节点执行，不要在计算节点上再 sbatch）：
#   bash exp12_c100_resnet_probe_sweep.sh
#   bash exp12_c100_resnet_probe_sweep.sh --dry-run

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

EXP_ID="exp12"
ACCOUNT="duanty"
PARTITION="debug"
GRES="gpu:rtx_4090:1"
CPUS=4
TIME="1-00:00:00"
CONDA_ENV="moe"
CONDA_ROOT="${HOME}/miniconda3"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
fi

BASE="--dataset cifar100 --backbone resnet --method ours --select ours --merge ours --device cuda --num_clients 10"

# tag|覆盖默认值的参数
# 默认：8 专家，batch 64，100 轮，K=1–3，带宽 [10, 80]，beta=0.1，scale=1，从第 1 轮计入
JOBS=(
    "s05|--radius_scale 0.5"
    "s025|--radius_scale 0.25"
    "s01|--radius_scale 0.1"
    "s005|--radius_scale 0.05"
    "d10|--radius_mode delta --delta 0.1"
    "d01|--radius_mode delta --delta 0.01"
    "d001|--radius_mode delta --delta 0.001"
    "k1-s025|--k_min 1 --k_max 1 --radius_scale 0.25"
    "k12-s025|--k_min 1 --k_max 2 --radius_scale 0.25"
    "k12-s01|--k_min 1 --k_max 2 --radius_scale 0.1"
    "late30-s025|--eval_from_round 30 --radius_scale 0.25"
    "late40-k12-s01|--k_min 1 --k_max 2 --eval_from_round 40 --radius_scale 0.1"
    "b32-s025|--batch_size 32 --rounds 80 --eval_from_round 15 --radius_scale 0.25"
    "b16-s025|--batch_size 16 --rounds 60 --eval_from_round 10 --radius_scale 0.25"
    "b16-k12-s01|--batch_size 16 --rounds 60 --k_min 1 --k_max 2 --bw_min 10 --bw_max 30 --eval_from_round 10 --radius_scale 0.1"
    "b32-k12-s01|--batch_size 32 --rounds 80 --k_min 1 --k_max 2 --eval_from_round 15 --radius_scale 0.1"
    "e16-s025|--num_experts 16 --bw_max 160 --rounds 80 --eval_from_round 10 --radius_scale 0.25"
    "e4-k12-s025|--num_experts 4 --bw_max 40 --k_min 1 --k_max 2 --rounds 80 --radius_scale 0.25"
    "beta05-s025|--beta 0.05 --rounds 80 --eval_from_round 10 --radius_scale 0.25"
    "b32-k12-s015|--batch_size 32 --rounds 80 --k_min 1 --k_max 2 --bw_min 10 --bw_max 40 --eval_from_round 20 --radius_scale 0.15 --seed 7"
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
echo "Command: python -u probe_agreement.py ${args}"
echo "========================================"

python -u probe_agreement.py ${args}
EOF

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "[dry-run] sbatch  ${name}"
        echo "          python -u probe_agreement.py ${args}"
        rm -f "$job_script"
        return
    fi

    local jobid
    jobid="$(sbatch --parsable "$job_script")"
    rm -f "$job_script"
    echo "submitted  ${jobid}  ${name}"
}

n_jobs=0
for jspec in "${JOBS[@]}"; do
    IFS='|' read -r tag extra <<< "$jspec"
    name="${EXP_ID}-c100-resnet-probe-${tag}"
    args="${BASE} ${extra} --output_dir ./results/${name}"
    submit_one "$name" "$args"
    n_jobs=$((n_jobs + 1))
done

echo
echo "共 ${n_jobs} 个任务（CIFAR-100 / ResNet 探测一致率扫描）-> 分区=${PARTITION}, GRES=${GRES}, 账号=${ACCOUNT}, 时限=${TIME}"

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo
    echo "这是 dry-run，没有真正提交。去掉 --dry-run 再执行即可。"
else
    echo
    echo "查看队列: squeue -u ${USER}"
    echo "日志目录: ${ROOT}/logs"
fi
