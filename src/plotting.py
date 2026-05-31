"""Matplotlib helpers that produce the figures shown in the Gradio demo."""

from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# A clean, modern look for all figures.
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})

COLORS = {
    "reinforce": "#2563eb",          # blue
    "reinforce_baseline": "#16a34a",  # green
}
LABELS = {
    "reinforce": "REINFORCE",
    "reinforce_baseline": "REINFORCE + Baseline",
}
SOLVED_THRESHOLD = 200  # LunarLander is considered solved at avg reward >= 200


def _moving_average(x: List[float], w: int = 50) -> np.ndarray:
    if len(x) < 1:
        return np.array([])
    w = min(w, len(x))
    return np.convolve(x, np.ones(w) / w, mode="valid")


def learning_curve(histories: Dict[str, dict], title: str = "Learning curve"):
    """Plot per-episode reward (faint) + rolling average (bold) for each algo."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=120)
    for algo, hist in histories.items():
        scores = hist.get("scores", [])
        avg = hist.get("avg_scores", [])
        if not scores:
            continue
        c = COLORS.get(algo, "#888")
        ax.plot(scores, color=c, alpha=0.15, linewidth=0.8)
        ax.plot(avg, color=c, linewidth=2.2, label=f"{LABELS.get(algo, algo)} (avg-100)")
    ax.axhline(SOLVED_THRESHOLD, color="#dc2626", linestyle="--", linewidth=1.2,
               label="Solved (200)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.set_title(title)
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    return fig


def reward_distribution(histories: Dict[str, dict]):
    """Histogram of evaluation rewards for each algo."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=120)
    for algo, hist in histories.items():
        rewards = hist.get("eval_rewards", [])
        if not rewards:
            continue
        c = COLORS.get(algo, "#888")
        ax.hist(rewards, bins=15, alpha=0.55, color=c,
                label=f"{LABELS.get(algo, algo)} (μ={np.mean(rewards):.0f})")
    ax.axvline(SOLVED_THRESHOLD, color="#dc2626", linestyle="--", linewidth=1.2,
               label="Solved (200)")
    ax.set_xlabel("Episode reward (evaluation)")
    ax.set_ylabel("Count")
    ax.set_title("Evaluation reward distribution (50 episodes)")
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    return fig


def episode_length_curve(histories: Dict[str, dict]):
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=120)
    for algo, hist in histories.items():
        lengths = hist.get("episode_lengths", [])
        if not lengths:
            continue
        c = COLORS.get(algo, "#888")
        ma = _moving_average(lengths, 50)
        ax.plot(range(len(ma)), ma, color=c, linewidth=2.0,
                label=f"{LABELS.get(algo, algo)}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps per episode (avg-50)")
    ax.set_title("Episode length over training")
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()
    return fig


def comparison_bars(histories: Dict[str, dict]):
    """Bar chart comparing final eval mean reward (+/- std)."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=120)
    algos, means, stds, colors = [], [], [], []
    for algo, hist in histories.items():
        if "eval_mean" not in hist:
            continue
        algos.append(LABELS.get(algo, algo))
        means.append(hist["eval_mean"])
        stds.append(hist.get("eval_std", 0.0))
        colors.append(COLORS.get(algo, "#888"))
    x = np.arange(len(algos))
    bars = ax.bar(x, means, yerr=stds, capsize=6, color=colors, alpha=0.85)
    ax.axhline(SOLVED_THRESHOLD, color="#dc2626", linestyle="--", linewidth=1.2,
               label="Solved (200)")
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m, f"{m:.0f}",
                ha="center", va="bottom", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(algos)
    ax.set_ylabel("Mean eval reward (50 eps)")
    ax.set_title("Final performance comparison")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def live_curve(scores: List[float], avg_scores: List[float], algo: str):
    """Single-run live learning curve used during interactive training."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=110)
    c = COLORS.get(algo, "#2563eb")
    ax.plot(scores, color=c, alpha=0.2, linewidth=0.8, label="reward / episode")
    ax.plot(avg_scores, color=c, linewidth=2.2, label="avg-100")
    ax.axhline(SOLVED_THRESHOLD, color="#dc2626", linestyle="--", linewidth=1.0,
               label="Solved (200)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.set_title(f"Live training — {LABELS.get(algo, algo)}")
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    return fig
