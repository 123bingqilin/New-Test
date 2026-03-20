from pathlib import Path

import gym
import wandb
import d4rl
import numpy as np
import torch
from tqdm import trange

from src.iql import ImplicitQLearning
from src.policy import GaussianPolicy, DeterministicPolicy
from src.value_functions import TwinQ, ValueFunction
from src.util import (
    return_range,
    set_seed,
    Log,
    sample_batch,
    torchify,
    evaluate_policy,
    set_default_device,
)
from src.auxiliary import (
    ForwardModel,
    InverseModel,
    SharedEncoder,
    SharedForwardModel,
    SharedInverseModel,
)
from src.corruption import apply_corruption


def normalize_args(args):
    if args.algo_name == "iql":
        args.use_forward = 0
        args.use_inverse = 0
        args.use_phys = 0

    if args.corruption_type == "none":
        args.data_tag = "clean"
    else:
        fix_tag = f"fix{args.fixed_corruption}_cseed{args.corruption_seed}"
        args.data_tag = (
            f"{args.corruption_type}"
            f"_r{args.corruption_ratio}"
            f"_s{args.corruption_std}"
            f"_{fix_tag}"
        )

    args.module_tag = f"f{args.use_forward}_i{args.use_inverse}_p{args.use_phys}"
    return args

def get_env_and_dataset(log, env_name, max_episode_steps, args):
    env = gym.make(env_name)
    dataset = d4rl.qlearning_dataset(env)

    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_range(dataset, max_episode_steps)
        log(f"Dataset returns have range [{min_ret}, {max_ret}]")
        dataset["rewards"] /= (max_ret - min_ret)
        dataset["rewards"] *= max_episode_steps
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0

    for k, v in dataset.items():
        dataset[k] = torchify(v)

    dataset, corruption_info = apply_corruption(dataset, args)
    return env, dataset, corruption_info


def build_aux_models(args, obs_dim, act_dim):
    encoder = None
    forward_model = None
    inverse_model = None

    if args.algo_name == "iql":
        return encoder, forward_model, inverse_model

    if args.model_mode == "separate":
        if args.use_forward:
            forward_model = ForwardModel(
                obs_dim,
                act_dim,
                hidden_dim=args.hidden_dim,
                n_hidden=args.n_hidden,
            )
        if args.use_inverse:
            inverse_model = InverseModel(
                obs_dim,
                act_dim,
                hidden_dim=args.hidden_dim,
                n_hidden=args.n_hidden,
            )
    elif args.model_mode == "shared":
        encoder = SharedEncoder(
            obs_dim,
            hidden_dim=args.hidden_dim,
            n_hidden=args.n_hidden,
        )
        feat_dim = args.hidden_dim
        if args.use_forward:
            forward_model = SharedForwardModel(
                feat_dim,
                act_dim,
                obs_dim,
                hidden_dim=args.hidden_dim,
                n_hidden=args.n_hidden,
            )
        if args.use_inverse:
            inverse_model = SharedInverseModel(
                feat_dim,
                act_dim,
                hidden_dim=args.hidden_dim,
                n_hidden=args.n_hidden,
            )
    else:
        raise ValueError(f"Unknown model_mode: {args.model_mode}")

    return encoder, forward_model, inverse_model


def main(args):
    args = normalize_args(args)

    if torch.cuda.is_available():
        set_default_device(f"cuda:{args.gpu_id}")
    else:
        set_default_device("cpu")

    torch.set_num_threads(1)

    log = Log(Path(args.log_dir) / args.env_name, vars(args))

    group_name = (
        f"{args.algo_name}_{args.data_tag}_{args.env_name}_{args.model_mode}_{args.module_tag}"
    )
    run_name = f"{group_name}_seed{args.seed}_gpu{args.gpu_id}"

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=group_name,
        name=run_name,
        config={**vars(args), "group_name": group_name, "run_name": run_name},
    )

    log(f"Log dir: {log.dir}")
    log(
        f"Using device: cuda:{args.gpu_id}"
        if torch.cuda.is_available()
        else "Using device: cpu"
    )

    env, dataset, corruption_info = get_env_and_dataset(
        log, args.env_name, args.max_episode_steps, args,
    )

    # ---- corruption 日志：用于验证 fixed corruption 是否真的复用同一批样本 ----
    corrupt_ratio_actual = float(dataset["corrupt_mask"].float().mean().item())
    corrupt_count = int(dataset["corrupt_mask"].sum().item())

    log(
        f"Actual corrupt ratio: {corrupt_ratio_actual:.6f} "
        f"({corrupt_count} / {len(dataset['corrupt_mask'])})"
    )

    log(f"Corrupt idx hash: {corruption_info['corrupt_idx_hash']}")
    log(f"Corrupt idx head(20): {corruption_info['corrupt_idx_head']}")

    wandb.log(
        {
            "data/corrupt_ratio_actual": corrupt_ratio_actual,
            "data/corrupt_count": corrupt_count,
            "data/corrupt_idx_hash": float(corruption_info["corrupt_idx_hash"]),
        },
        step=0,
    )

    obs_dim = dataset["observations"].shape[1]
    act_dim = dataset["actions"].shape[1]

    # training seed 只控制训练，不控制 corruption
    set_seed(args.seed, env=env)

    if args.deterministic_policy:
        policy = DeterministicPolicy(
            obs_dim,
            act_dim,
            hidden_dim=args.hidden_dim,
            n_hidden=args.n_hidden,
        )
    else:
        policy = GaussianPolicy(
            obs_dim,
            act_dim,
            hidden_dim=args.hidden_dim,
            n_hidden=args.n_hidden,
        )

    encoder, forward_model, inverse_model = build_aux_models(args, obs_dim, act_dim)

    iql = ImplicitQLearning(
        qf=TwinQ(
            obs_dim,
            act_dim,
            hidden_dim=args.hidden_dim,
            n_hidden=args.n_hidden,
        ),
        vf=ValueFunction(
            obs_dim,
            hidden_dim=args.hidden_dim,
            n_hidden=args.n_hidden,
        ),
        policy=policy,
        encoder=encoder,
        forward_model=forward_model,
        inverse_model=inverse_model,
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=args.learning_rate),
        max_steps=args.n_steps,
        tau=args.tau,
        beta=args.beta,
        alpha=args.alpha,
        discount=args.discount,
        lambda_inv=args.lambda_inv,
        alpha_phys=args.alpha_phys,
        aux_weight=args.aux_weight,
        algo_name=args.algo_name,
        model_mode=args.model_mode,
        use_forward=args.use_forward,
        use_inverse=args.use_inverse,
        use_phys=args.use_phys,
    )

    best_normalized_return = -1e9

    for step in trange(args.n_steps):
        metrics = iql.update(**sample_batch(dataset, args.batch_size))
        metrics["step"] = step
        wandb.log(metrics, step=step)

        if (step + 1) % args.eval_period == 0:
            eval_returns = np.array(
                [
                    evaluate_policy(env, policy, args.max_episode_steps)
                    for _ in range(args.n_eval_episodes)
                ]
            )
            normalized_returns = d4rl.get_normalized_score(args.env_name, eval_returns) * 100.0
            eval_metrics = {
                "eval/return_mean": eval_returns.mean(),
                "eval/return_std": eval_returns.std(),
                "eval/normalized_return_mean": normalized_returns.mean(),
                "eval/normalized_return_std": normalized_returns.std(),
                "step": step,
            }
            wandb.log(eval_metrics, step=step)
            log.row(eval_metrics)

            if normalized_returns.mean() > best_normalized_return:
                best_normalized_return = normalized_returns.mean()
                if args.save_best:
                    torch.save(iql.state_dict(), log.dir / "best.pt")

    torch.save(iql.state_dict(), log.dir / "final.pt")
    log.close()
    wandb.finish()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--env-name", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-id", type=int, default=0)

    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-hidden", type=int, default=2)
    parser.add_argument("--n-steps", type=int, default=10 ** 6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--alpha", type=float, default=0.005)
    parser.add_argument("--tau", type=float, default=0.7)
    parser.add_argument("--beta", type=float, default=3.0)
    parser.add_argument("--deterministic-policy", action="store_true")
    parser.add_argument("--eval-period", type=int, default=5000)
    parser.add_argument("--n-eval-episodes", type=int, default=10)
    parser.add_argument("--max-episode-steps", type=int, default=1000)

    parser.add_argument("--lambda-inv", type=float, default=1.0)
    parser.add_argument("--alpha-phys", type=float, default=0.1)
    parser.add_argument("--aux-weight", type=float, default=1.0)

    parser.add_argument("--wandb-entity", type=str, default="dlut-pqj")
    parser.add_argument("--wandb-project", type=str, default="iql-phys")

    parser.add_argument("--algo-name", type=str, default="physiql", choices=["iql", "physiql"])
    parser.add_argument("--model-mode", type=str, default="separate", choices=["separate", "shared"])
    parser.add_argument("--use-forward", type=int, default=1)
    parser.add_argument("--use-inverse", type=int, default=1)
    parser.add_argument("--use-phys", type=int, default=1)

    parser.add_argument(
        "--corruption-type",
        type=str,
        default="none",
        choices=[
            "none",
            "obs_noise",
            "action_noise",
            "reward_noise",
            "mask",
            "transition_shuffle",
        ],
    )
    parser.add_argument("--corruption-ratio", type=float, default=0.0)
    parser.add_argument("--corruption-std", type=float, default=0.0)

    # corruption 独立随机控制
    parser.add_argument("--fixed-corruption", type=int, default=1)
    parser.add_argument("--corruption-seed", type=int, default=0)

    parser.add_argument("--data-tag", type=str, default="clean")
    parser.add_argument("--save-best", type=int, default=1)

    main(parser.parse_args())