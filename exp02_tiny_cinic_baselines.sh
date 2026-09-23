#!/bin/bash
# exp02：主实验 baseline 补 TinyImageNet / CINIC-10。
# 方法、超参、带宽与 MainResult/实验说明.txt 完全一致，只换这两个数据集。
# 默认分区 debug 全是 RTX 4090，并用 --gres=gpu:rtx_4090:1 钉死卡型。
#
# 方法（图例名）：
#   Ours      --select ours     --merge ours
#   FedRolex  --select fedrolex --merge ours
#   Random    --select random   --merge ours
#   FedAvg    --select ours     --merge fedavg
#   FedAdam   --select ours     --merge fedadam  --fedadam_tau 0.01
# 数据集：TinyImageNet / CINIC-10。骨干：ResNet / VGG11。
# --bw_max -1：8 专家时带宽 [10, 80]。
# 共 2 数据集 × 2 骨干 × 5 方法 = 20 个任务。
#
# 用法（在登录节点执行，不要在计算节点上再 sbatch）：
#   bash exp02_tiny_cinic_baselines.sh
#   bash exp02_tiny_cinic_baselines.sh --dry-run
# 查看 / 取消：
#   squeue -u "$USER"
#   scancel <jobid>

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

EXP_ID="exp02"
ACCOUNT="duanty"
PARTITION="debug"
GRES="gpu:rtx_4090:1"
CPUS=4
TIME="2-00:00:00"
CONDA_ENV="moe"
CONDA_ROOT="${HOME}/miniconda3"
NUM_EXPERTS=8

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
fi

# tag|cli
DATASETS=(
    "tiny|tinyimagenet"
    "cinic|cinic-10"
)
BACKBONES=(resnet vgg11)
# tag|select|merge|extra
METHODS=(
    "ours|ours|ours|"
    "fedrolex|fedrolex|ours|"
    "random|random|ours|"
    "fedavg|ours|fedavg|"
    "fedadam|ours|fedadam|--fedadam_tau 0.01"
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
for dspec in "${DATASETS[@]}"; do
    IFS='|' read -r dtag dataset <<< "$dspec"
    for backbone in "${BACKBONES[@]}"; do
        for mspec in "${METHODS[@]}"; do
            IFS='|' read -r mtag select merge extra <<< "$mspec"
            name="${EXP_ID}-${dtag}-${backbone}-e${NUM_EXPERTS}-${mtag}"
            args="--method ours --dataset ${dataset} --backbone ${backbone} --select ${select} --merge ${merge} --device cuda --num_experts ${NUM_EXPERTS} --bw_max -1 --output_dir ./results/${name}"
            if [[ -n "${extra}" ]]; then
                args="${args} ${extra}"
            fi
            submit_one "$name" "$args"
            n_jobs=$((n_jobs + 1))
        done
    done
done

echo
echo "共 ${n_jobs} 个任务（2 数据集 × 2 骨干 × 5 方法）-> 分区=${PARTITION}, GRES=${GRES}, 账号=${ACCOUNT}"

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo
    echo "这是 dry-run，没有真正提交。去掉 --dry-run 再执行即可。"
else
    echo
    echo "查看队列: squeue -u ${USER}"
    echo "日志目录: ${ROOT}/logs"
fi
