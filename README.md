# BLA-FedMoE

三部分互相隔离：

| 位置 | 内容 | 能不能改 |
|------|------|----------|
| `original paper/` | 原论文 PDF | 不要改 |
| `original code/` | 论文配套原版脚本（全专家上传） | 不要改 |
| 仓库根目录 | 你的实验代码 | 在这里改 |

## 入口

```bash
python main.py --method bla-fedmoe
python main.py --method moe
python main.py --method moe --dataset cifar10 --num_clients 10 --num_experts 4 --rounds 100 --seed 42
```

`--method` 可选：

- `bla-fedmoe`：`BLAFedMoE`（`select_upload_experts` 只传 k 个专家 + FedLPA）
- `moe`：普通联邦 MoE（全专家上传，参数平均）

`--expert_k` 只对 `bla-fedmoe` 生效。`--expert_k 0` 表示上传全部专家。

## 代码结构

- `main.py`：读参数、数据、联邦循环
- `methods/bla_fedmoe.py`：`BLAFedMoE`
- `methods/moe.py`：`MoE`
- `select_upload_experts.py`：仅 `BLAFedMoE` 用，上传前选专家
- `model.py` / `data_utils.py` / `utils.py`：模型和公共工具

## 依赖

```bash
pip install numpy pandas torch torchvision openpyxl
```

数据默认在 `./data`，结果写到 `./results/`。
