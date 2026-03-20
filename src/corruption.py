import torch


def _build_generator(dataset, args):
    fixed = bool(getattr(args, "fixed_corruption", 0))
    if not fixed:
        return None
    device = dataset["observations"].device
    g = torch.Generator(device=device)
    g.manual_seed(int(getattr(args, "corruption_seed", 0)))
    return g


def _randperm(n, device, generator=None):
    if generator is None:
        return torch.randperm(n, device=device)
    return torch.randperm(n, device=device, generator=generator)


def _rand(shape, device, dtype, generator=None):
    if generator is None:
        return torch.rand(shape, device=device, dtype=dtype)
    return torch.rand(shape, device=device, dtype=dtype, generator=generator)


def _randn(shape, device, dtype, generator=None):
    if generator is None:
        return torch.randn(shape, device=device, dtype=dtype)
    return torch.randn(shape, device=device, dtype=dtype, generator=generator)


def _select_corrupt_idx(n, ratio, device, generator=None):
    if ratio <= 0.0:
        return torch.empty(0, dtype=torch.long, device=device)
    corrupt_num = max(1, int(n * ratio))
    return _randperm(n, device=device, generator=generator)[:corrupt_num]


def _compute_corrupt_signature(corrupt_idx: torch.Tensor):
    """
    返回两个轻量日志字段：
    1) 前 20 个 corrupt idx
    2) 一个稳定的 hash-like 签名，用于跨 seed 对比

    不把完整 corrupt_idx 挂到 dataset 上，避免占用额外内存。
    """
    if corrupt_idx.numel() == 0:
        head = torch.empty(0, dtype=torch.long)
        idx_hash = -1
        return head, idx_hash

    corrupt_idx_cpu = corrupt_idx.detach().cpu().to(torch.long)
    head = corrupt_idx_cpu[:20].clone()

    # 一个简单稳定的整数签名，够用于验证“是否同一批索引”
    # 用 int64 规避溢出，再转 Python int
    weights = torch.arange(
        1, corrupt_idx_cpu.numel() + 1, dtype=torch.long
    )
    idx_hash = int((corrupt_idx_cpu * weights).sum().item())

    return head, idx_hash


def apply_corruption(dataset, args):
    corruption_type = args.corruption_type
    ratio = float(args.corruption_ratio)
    std = float(args.corruption_std)

    dataset = {k: v.clone() for k, v in dataset.items()}

    n = dataset["observations"].shape[0]
    device = dataset["observations"].device

    corrupt_mask = torch.zeros(n, dtype=torch.float32, device=device)

    if corruption_type == "none":
        dataset["corrupt_mask"] = corrupt_mask
        dataset["corrupt_idx_head"] = torch.empty(0, dtype=torch.long)
        dataset["corrupt_idx_hash"] = -1
        return dataset

    g = _build_generator(dataset, args)

    obs = dataset["observations"]
    next_obs = dataset["next_observations"]
    actions = dataset["actions"]
    rewards = dataset["rewards"]

    corrupt_idx = _select_corrupt_idx(n, ratio, device=device, generator=g)

    if len(corrupt_idx) > 0:
        corrupt_mask[corrupt_idx] = 1.0

    if corruption_type == "obs_noise":
        if len(corrupt_idx) > 0:
            obs_noise = std * _randn(
                obs[corrupt_idx].shape,
                device=obs.device,
                dtype=obs.dtype,
                generator=g,
            )
            next_obs_noise = std * _randn(
                next_obs[corrupt_idx].shape,
                device=next_obs.device,
                dtype=next_obs.dtype,
                generator=g,
            )
            dataset["observations"][corrupt_idx] = (
                dataset["observations"][corrupt_idx] + obs_noise
            )
            dataset["next_observations"][corrupt_idx] = (
                dataset["next_observations"][corrupt_idx] + next_obs_noise
            )

    elif corruption_type == "action_noise":
        if len(corrupt_idx) > 0:
            act_noise = std * _randn(
                actions[corrupt_idx].shape,
                device=actions.device,
                dtype=actions.dtype,
                generator=g,
            )
            dataset["actions"][corrupt_idx] = (
                dataset["actions"][corrupt_idx] + act_noise
            )
        dataset["actions"] = dataset["actions"].clamp(-1.0, 1.0)

    elif corruption_type == "reward_noise":
        if len(corrupt_idx) > 0:
            rew_noise = std * _randn(
                rewards[corrupt_idx].shape,
                device=rewards.device,
                dtype=rewards.dtype,
                generator=g,
            )
            dataset["rewards"][corrupt_idx] = (
                dataset["rewards"][corrupt_idx] + rew_noise
            )

    elif corruption_type == "mask":
        if len(corrupt_idx) > 0:
            obs_keep = (
                _rand(
                    obs[corrupt_idx].shape,
                    device=obs.device,
                    dtype=obs.dtype,
                    generator=g,
                ) > 0.5
            ).float()
            next_obs_keep = (
                _rand(
                    next_obs[corrupt_idx].shape,
                    device=next_obs.device,
                    dtype=next_obs.dtype,
                    generator=g,
                ) > 0.5
            ).float()

            dataset["observations"][corrupt_idx] = (
                dataset["observations"][corrupt_idx] * obs_keep
            )
            dataset["next_observations"][corrupt_idx] = (
                dataset["next_observations"][corrupt_idx] * next_obs_keep
            )

    elif corruption_type == "transition_shuffle":
        if len(corrupt_idx) > 0:
            shuffle_num = len(corrupt_idx)
            shuffled_idx = corrupt_idx[
                _randperm(shuffle_num, device=device, generator=g)
            ]
            dataset["next_observations"][corrupt_idx] = (
                dataset["next_observations"][shuffled_idx]
            )

    else:
        raise ValueError(f"Unknown corruption type: {corruption_type}")

    dataset["corrupt_mask"] = corrupt_mask

    # 只保存轻量验证信息，不保存完整 corrupt_idx
    corrupt_idx_head, corrupt_idx_hash = _compute_corrupt_signature(corrupt_idx)
    dataset["corrupt_idx_head"] = corrupt_idx_head
    dataset["corrupt_idx_hash"] = corrupt_idx_hash

    return dataset