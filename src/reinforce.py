"""REINFORCE (Monte-Carlo Policy Gradient) training algorithms.

Two variants are implemented:

1. ``reinforce`` — the vanilla algorithm exactly as presented in the slides
   (Exercise 3). For every episode we collect a trajectory, compute the
   discounted returns G_t, normalise them, and minimise
   ``loss = - sum_t log pi(a_t|s_t) * G_t``.

2. ``reinforce_with_baseline`` — the same Monte-Carlo policy gradient but the
   return is replaced by the *advantage* ``A_t = G_t - V(s_t)``, where V is a
   learned value baseline. This keeps the estimator unbiased while reducing its
   variance, which usually means faster and more stable learning.

Both functions share the same signature/return type so the Gradio app and the
training script can treat them interchangeably.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from .policy import Policy, PolicyWithValue


@dataclass
class TrainResult:
    """Container for everything produced during a training run."""

    scores: List[float] = field(default_factory=list)          # return per episode
    avg_scores: List[float] = field(default_factory=list)      # rolling mean (100)
    episode_lengths: List[int] = field(default_factory=list)   # steps per episode
    policy_losses: List[float] = field(default_factory=list)
    value_losses: List[float] = field(default_factory=list)
    algo: str = "reinforce"


def _discounted_returns(rewards: List[float], gamma: float, max_steps: int) -> deque:
    """Compute G_t for every timestep using the slide's appendleft trick."""
    returns: deque = deque(maxlen=max_steps)
    n_steps = len(rewards)
    for t in range(n_steps)[::-1]:
        disc_return_t = returns[0] if len(returns) > 0 else 0.0
        returns.appendleft(gamma * disc_return_t + rewards[t])
    return returns


def reinforce(
    policy: Policy,
    optimizer: optim.Optimizer,
    env,
    n_training_episodes: int,
    max_steps: int,
    gamma: float,
    device: torch.device,
    print_every: int = 100,
    progress_cb: Optional[Callable[[int, int, float], None]] = None,
) -> TrainResult:
    """Vanilla REINFORCE, faithful to the slides (with gymnasium step API)."""
    result = TrainResult(algo="reinforce")
    scores_deque: deque = deque(maxlen=100)

    for i_episode in range(1, n_training_episodes + 1):
        saved_log_probs: List[torch.Tensor] = []
        rewards: List[float] = []
        state, _ = env.reset()

        for _ in range(max_steps):
            action, log_prob = policy.act(state, device)
            saved_log_probs.append(log_prob)
            state, reward, terminated, truncated, _ = env.step(action)
            rewards.append(float(reward))
            if terminated or truncated:
                break

        total_reward = float(sum(rewards))
        scores_deque.append(total_reward)
        result.scores.append(total_reward)
        result.avg_scores.append(float(np.mean(scores_deque)))
        result.episode_lengths.append(len(rewards))

        returns = _discounted_returns(rewards, gamma, max_steps)
        eps = np.finfo(np.float32).eps.item()
        returns_t = torch.tensor(list(returns), dtype=torch.float32)
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + eps)

        policy_loss = [-log_prob * R for log_prob, R in zip(saved_log_probs, returns_t)]
        policy_loss = torch.cat(policy_loss).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        result.policy_losses.append(float(policy_loss.item()))
        result.value_losses.append(0.0)

        if progress_cb is not None:
            progress_cb(i_episode, n_training_episodes, result.avg_scores[-1])
        if print_every and i_episode % print_every == 0:
            print(f"[REINFORCE] Episode {i_episode}\tAverage Score: {result.avg_scores[-1]:.2f}")

    return result


def reinforce_with_baseline(
    policy: PolicyWithValue,
    optimizer: optim.Optimizer,
    env,
    n_training_episodes: int,
    max_steps: int,
    gamma: float,
    device: torch.device,
    print_every: int = 100,
    value_coef: float = 0.5,
    progress_cb: Optional[Callable[[int, int, float], None]] = None,
) -> TrainResult:
    """REINFORCE with a learned value baseline (advantage = G_t - V(s_t))."""
    result = TrainResult(algo="reinforce_baseline")
    scores_deque: deque = deque(maxlen=100)

    for i_episode in range(1, n_training_episodes + 1):
        saved_log_probs: List[torch.Tensor] = []
        saved_values: List[torch.Tensor] = []
        rewards: List[float] = []
        state, _ = env.reset()

        for _ in range(max_steps):
            action, log_prob, value = policy.act(state, device)
            saved_log_probs.append(log_prob)
            saved_values.append(value)
            state, reward, terminated, truncated, _ = env.step(action)
            rewards.append(float(reward))
            if terminated or truncated:
                break

        total_reward = float(sum(rewards))
        scores_deque.append(total_reward)
        result.scores.append(total_reward)
        result.avg_scores.append(float(np.mean(scores_deque)))
        result.episode_lengths.append(len(rewards))

        returns = _discounted_returns(rewards, gamma, max_steps)
        eps = np.finfo(np.float32).eps.item()
        returns_t = torch.tensor(list(returns), dtype=torch.float32)
        returns_norm = (returns_t - returns_t.mean()) / (returns_t.std() + eps)

        values_t = torch.cat(saved_values).to("cpu")
        # Advantage uses the (normalised) return minus the value estimate. We
        # detach the baseline so it only reduces variance, not introduces bias.
        advantages = returns_norm - values_t.detach()

        policy_loss = torch.cat(
            [-log_prob * A for log_prob, A in zip(saved_log_probs, advantages)]
        ).sum()
        value_loss = F.mse_loss(values_t, returns_norm, reduction="sum")
        loss = policy_loss + value_coef * value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        result.policy_losses.append(float(policy_loss.item()))
        result.value_losses.append(float(value_loss.item()))

        if progress_cb is not None:
            progress_cb(i_episode, n_training_episodes, result.avg_scores[-1])
        if print_every and i_episode % print_every == 0:
            print(
                f"[REINFORCE+Baseline] Episode {i_episode}\t"
                f"Average Score: {result.avg_scores[-1]:.2f}"
            )

    return result
