#!/bin/bash
# FedAdam 超参三组，其余设置与 run_e8_baselines.sh 相同：
#   专家数 8，--select ours --merge fedadam
#   数据集只保留上次能跑完的 CIFAR-10 / CIFAR-100 / STL-10（去掉损坏的 SVHN）
#   骨干 ResNet / VGG11
#   --bw_max -1：8 专家时带宽为 [10, 80]，使 K 仍能到 M。
#
# 三组只改一个旋钮（未写出的仍是默认 lr=0.01, beta1=0.9, beta2=0.99, tau=1e-3）：
#   lr1e3   --fedadam_lr 0.001
#   tau1e2  --fedadam_tau 0.01
#   b2999   --fedadam_beta2 0.999
# 共 3 数据集 × 2 骨干 × 3 组超参 = 18 个任务。
#
# 用法（在登录节点执行，不要在计算节点上再 sbatch）：
#   bash run_e8_fedadam_sweep.sh
#   bash run_e8_fedadam_sweep.sh --dry-run
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
NUM_EXPERTS=8

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
fi

# tag|cli
DATASETS=(
    "c10|cifar10"
    "c100|cifar100"
    "stl10|stl10"
)
BACKBONES=(resnet vgg11)
# tag|extra args
ADAM_CFGS=(
    "lr1e3|--fedadam_lr 0.001"
    "tau1e2|--fedadam_tau 0.01"
    "b2999|--fedadam_beta2 0.999"
)

mkdir -p logs

submit_one() {
    local name="$1"
    local args="$2"
    local job_script
    job_script="$(mktemp "${TMPDIR:-/tmp}/e8_adam.XXXXXX.sh")"

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
        for cspec in "${ADAM_CFGS[@]}"; do
            IFS='|' read -r ctag extra <<< "$cspec"
            name="${dtag}-${backbone}-e${NUM_EXPERTS}-fedadam-${ctag}"
            args="--method ours --dataset ${dataset} --backbone ${backbone} --select ours --merge fedadam --device cuda --num_experts ${NUM_EXPERTS} --bw_max -1 ${extra} --output_dir ./results/${name}"
            submit_one "$name" "$args"
            n_jobs=$((n_jobs + 1))
        done
    done
done

echo
echo "共 ${n_jobs} 个任务（3 数据集 × 2 骨干 × 3 组超参）-> 分区=${PARTITION}, GRES=${GRES}, 账号=${ACCOUNT}"

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo
    echo "这是 dry-run，没有真正提交。去掉 --dry-run 再执行即可。"
else
    echo
    echo "查看队列: squeue -u ${USER}"
    echo "日志目录: ${ROOT}/logs"
fi
