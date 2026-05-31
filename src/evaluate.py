"""Evaluation utilities: score an agent and render an episode to a GIF."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch


def evaluate_agent(
    env,
    policy,
    device: torch.device,
    max_steps: int,
    n_eval_episodes: int,
    greedy: bool = False,
    seed: Optional[List[int]] = None,
) -> Tuple[float, float, List[float]]:
    """Run the policy for ``n_eval_episodes`` and return mean/std/all rewards."""
    episode_rewards: List[float] = []
    for episode in range(n_eval_episodes):
        if seed is not None:
            state, _ = env.reset(seed=seed[episode])
        else:
            state, _ = env.reset()
        total_reward = 0.0
        for _ in range(max_steps):
            if greedy:
                action = policy.act_greedy(state, device)
            else:
                out = policy.act(state, device)
                action = out[0]
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break
        episode_rewards.append(total_reward)

    mean_reward = float(np.mean(episode_rewards))
    std_reward = float(np.std(episode_rewards))
    return mean_reward, std_reward, episode_rewards


def record_episode_gif(
    policy,
    device: torch.device,
    out_path: str,
    max_steps: int = 1000,
    seed: Optional[int] = None,
    greedy: bool = True,
    fps: int = 30,
) -> Tuple[str, float]:
    """Play one episode with rgb_array rendering and save it as a GIF.

    Returns the GIF path and the total episode reward.
    """
    import gymnasium as gym
    import imageio

    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    frames = []
    if seed is not None:
        state, _ = env.reset(seed=seed)
    else:
        state, _ = env.reset()
    total_reward = 0.0
    for _ in range(max_steps):
        frames.append(env.render())
        if greedy:
            action = policy.act_greedy(state, device)
        else:
            action = policy.act(state, device)[0]
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        if terminated or truncated:
            frames.append(env.render())
            break
    env.close()

    # Downsample frames a little to keep the GIF light.
    frames = [f for i, f in enumerate(frames) if i % 2 == 0]
    imageio.mimsave(out_path, frames, fps=fps // 2, loop=0)
    return out_path, total_reward
