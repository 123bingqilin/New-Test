import torch

def apply_corruption(dataset, args):
    corruption_type = args.corruption_type
    ratio = args.corruption_ratio
    std = args.corruption_std

    if corruption_type == "none":
        return dataset

    dataset = {k: v.clone() for k, v in dataset.items()}

    if corruption_type == "obs_noise":
        dataset["observations"] = dataset["observations"] + std * torch.randn_like(dataset["observations"])
        dataset["next_observations"] = dataset["next_observations"] + std * torch.randn_like(dataset["next_observations"])

    elif corruption_type == "action_noise":
        dataset["actions"] = dataset["actions"] + std * torch.randn_like(dataset["actions"])
        dataset["actions"] = dataset["actions"].clamp(-1.0, 1.0)

    elif corruption_type == "reward_noise":
        dataset["rewards"] = dataset["rewards"] + std * torch.randn_like(dataset["rewards"])

    elif corruption_type == "mask":
        obs_mask = (torch.rand_like(dataset["observations"]) > ratio).float()
        next_obs_mask = (torch.rand_like(dataset["next_observations"]) > ratio).float()
        dataset["observations"] = dataset["observations"] * obs_mask
        dataset["next_observations"] = dataset["next_observations"] * next_obs_mask

    elif corruption_type == "transition_shuffle":
        n = dataset["observations"].shape[0]
        shuffle_num = int(n * ratio)
        corrupt_idx = torch.randperm(n)[:shuffle_num]
        shuffled_idx = corrupt_idx[torch.randperm(shuffle_num)]
        dataset["next_observations"][corrupt_idx] = dataset["next_observations"][shuffled_idx]

    else:
        raise ValueError(f"Unknown corruption type: {corruption_type}")

    return dataset