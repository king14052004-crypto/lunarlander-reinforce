"""Train a REINFORCE agent (vanilla or with-baseline) on LunarLander.

Usage:
    python -m src.train --algo reinforce          --episodes 2000
    python -m src.train --algo reinforce_baseline --episodes 2000
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict

import gymnasium as gym
import torch
import torch.optim as optim

from .config import DEFAULT_HPARAMS, ENV_ID, get_device
from .evaluate import evaluate_agent
from .policy import Policy, PolicyWithValue
from .reinforce import reinforce, reinforce_with_baseline

CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints")


def train(algo: str, episodes: int, lr: float, gamma: float, h_size: int,
          max_steps: int, seed: int = 0) -> str:
    device = get_device()
    torch.manual_seed(seed)
    env = gym.make(ENV_ID)
    s_size = env.observation_space.shape[0]
    a_size = env.action_space.n

    if algo == "reinforce":
        policy = Policy(s_size, a_size, h_size).to(device)
        optimizer = optim.Adam(policy.parameters(), lr=lr)
        train_fn = reinforce
    elif algo == "reinforce_baseline":
        policy = PolicyWithValue(s_size, a_size, h_size).to(device)
        optimizer = optim.Adam(policy.parameters(), lr=lr)
        train_fn = reinforce_with_baseline
    else:
        raise ValueError(f"Unknown algo: {algo}")

    t0 = time.time()
    result = train_fn(
        policy, optimizer, env,
        n_training_episodes=episodes,
        max_steps=max_steps,
        gamma=gamma,
        device=device,
        print_every=DEFAULT_HPARAMS["print_every"],
    )
    train_time = time.time() - t0

    # Final evaluation with fixed seeds for reproducibility.
    eval_env = gym.make(ENV_ID)
    eval_seeds = list(range(50))
    mean_r, std_r, all_r = evaluate_agent(
        eval_env, policy, device, max_steps, n_eval_episodes=50,
        greedy=False, seed=eval_seeds,
    )

    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CKPT_DIR, f"{algo}.pt")
    torch.save({
        "algo": algo,
        "state_dict": policy.state_dict(),
        "s_size": int(s_size), "a_size": int(a_size), "h_size": int(h_size),
    }, ckpt_path)

    history = {
        **asdict(result),
        "eval_mean": mean_r, "eval_std": std_r, "eval_rewards": all_r,
        "train_time_sec": train_time,
        "hparams": {"episodes": episodes, "lr": lr, "gamma": gamma,
                    "h_size": h_size, "max_steps": max_steps},
    }
    hist_path = os.path.join(CKPT_DIR, f"{algo}_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f)

    print(f"\n{algo}: eval mean_reward={mean_r:.2f} +/- {std_r:.2f} "
          f"(trained {episodes} eps in {train_time:.0f}s)")
    print(f"Saved checkpoint -> {ckpt_path}")
    print(f"Saved history    -> {hist_path}")
    return ckpt_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["reinforce", "reinforce_baseline"],
                   default="reinforce")
    p.add_argument("--episodes", type=int, default=DEFAULT_HPARAMS["n_training_episodes"])
    p.add_argument("--lr", type=float, default=DEFAULT_HPARAMS["lr"])
    p.add_argument("--gamma", type=float, default=DEFAULT_HPARAMS["gamma"])
    p.add_argument("--h_size", type=int, default=DEFAULT_HPARAMS["h_size"])
    p.add_argument("--max_steps", type=int, default=DEFAULT_HPARAMS["max_steps"])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    train(args.algo, args.episodes, args.lr, args.gamma, args.h_size,
          args.max_steps, args.seed)


if __name__ == "__main__":
    main()
