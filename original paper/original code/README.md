# BLA-FedMoE Original Code

Unmodified paper supplement. Do not edit this folder; put all changes in the repository root.

## Files

- `BLA-FedMoE.py`: Original implementation and experiment script.
- `environment.yml` / `requirements.txt`: Conda / pip dependencies.

## Usage

Run from the repository root so `./data` and `./results` stay shared:

```bash
python "original code/BLA-FedMoE.py"
python "original code/BLA-FedMoE.py" --dataset cifar10 --num_clients 10 --num_experts 4 --rounds 100 --seed 42
```

Each selected client uploads **all** experts. There is no `--expert_k`.
