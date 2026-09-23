"""Client-cloud bandwidth budget for expert upload.

k = min(num_experts, available_bw // expert_bw)
Static: bw_min == bw_max.
Time-varying: Rayleigh block fading -> Shannon rate, clipped to [bw_min, bw_max].
"""
import hashlib

import numpy as np


EXPERT_PREFIX = 'moe_head.experts.'
DEFAULT_SNR_DB = 10.0


def _rng(seed, round_id, client_id, salt):
    raw = f'{int(seed)}-{int(round_id)}-{int(client_id)}-{salt}'.encode()
    h = int.from_bytes(hashlib.md5(raw).digest()[:4], 'little')
    return np.random.RandomState(h)


def resolve_bw_range(args):
    expert_bw = max(int(args.expert_bw), 1)
    full_bw = int(args.num_experts) * expert_bw
    bw_max = full_bw if int(args.bw_max) < 0 else int(args.bw_max)
    bw_min = bw_max if int(args.bw_min) < 0 else int(args.bw_min)
    if bw_min > bw_max:
        bw_min, bw_max = bw_max, bw_min
    return bw_min, bw_max, expert_bw


def sample_bandwidth(args, client_id, round_id):
    """Fixed budget, or Rayleigh-Shannon budget clipped to [bw_min, bw_max]."""
    bw_min, bw_max, _ = resolve_bw_range(args)
    if bw_min == bw_max:
        return bw_min
    rng = _rng(args.seed, round_id, client_id, 'bandwidth')
    # h ~ CN(0,1) => |h|^2 ~ Exp(1). Mean SNR maps |h|^2=1 to the interval midpoint.
    gain = float(rng.exponential(1.0))
    snr = 10.0 ** (float(getattr(args, 'snr_db', DEFAULT_SNR_DB)) / 10.0)
    rate = np.log2(1.0 + snr * gain)
    rate_ref = np.log2(1.0 + snr)
    scale = 0.5 * (bw_min + bw_max) / max(rate_ref, 1e-12)
    return int(np.clip(np.rint(scale * rate), bw_min, bw_max))


def upload_expert_count(bandwidth, expert_bw, num_experts):
    expert_bw = max(int(expert_bw), 1)
    return int(min(num_experts, max(0, int(bandwidth) // expert_bw)))


def apply_drop_expert(k, prob, seed, client_id, round_id):
    """With probability `prob`, upload one fewer expert. K stays at least 0."""
    k = int(max(0, k))
    prob = float(prob or 0.0)
    if k <= 0 or prob <= 0.0:
        return k
    rng = _rng(seed, round_id, client_id, 'drop-expert')
    if float(rng.rand()) < prob:
        return k - 1
    return k


def random_select_experts(num_experts, k, seed, client_id, round_id, salt):
    k = int(max(0, min(k, num_experts)))
    if k <= 0:
        return []
    if k >= num_experts:
        return list(range(num_experts))
    rng = _rng(seed, round_id, client_id, salt)
    return sorted(rng.choice(num_experts, size=k, replace=False).tolist())


def expert_id_from_key(key):
    if not key.startswith(EXPERT_PREFIX):
        return None
    rest = key[len(EXPERT_PREFIX):]
    head = rest.split('.', 1)[0]
    if not head.isdigit():
        return None
    return int(head)


def filter_expert_state(state, expert_ids):
    keep = set(expert_ids)
    return {
        key: value for key, value in state.items()
        if (eid := expert_id_from_key(key)) is None or eid in keep
    }


def filter_expert_extra(extra, expert_ids):
    if not extra:
        return extra
    keep = set(expert_ids)
    return {
        key: value for key, value in extra.items()
        if (eid := expert_id_from_key(key)) is None or eid in keep
    }


def filter_expert_payload(state, extra, expert_ids):
    return filter_expert_state(state, expert_ids), filter_expert_extra(extra, expert_ids)


def plan_upload(args, client_id, round_id, salt):
    bandwidth = sample_bandwidth(args, client_id, round_id)
    k = upload_expert_count(bandwidth, args.expert_bw, args.num_experts)
    expert_ids = random_select_experts(
        args.num_experts, k, args.seed, client_id, round_id, salt,
    )
    return bandwidth, k, expert_ids
