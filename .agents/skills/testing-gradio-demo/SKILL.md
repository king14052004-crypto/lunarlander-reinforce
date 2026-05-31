---
name: testing-gradio-demo
description: Test the LunarLander REINFORCE Gradio demo end-to-end. Use when verifying the app.py UI (overview/comparison, watch-agent, interactive training) or any change to the REINFORCE training/plotting/eval code.
---

# Testing the LunarLander REINFORCE Gradio demo

## Setup & launch
- Python venv: `source /home/ubuntu/.venv-rl/bin/activate` (has gymnasium[box2d], torch, gradio, matplotlib, imageio, pandas).
- From repo root: `PYTHONPATH=$(pwd) python app.py` → serves at `http://localhost:7860`.
- Pre-trained checkpoints + history JSON live in `checkpoints/`. Static plots/GIFs live in `assets/`. If these are missing, the overview tab will be empty — regenerate by training (`python src/train.py ...`) before testing.
- Maximize the browser before recording: `wmctrl -r :ACTIVE: -b add,maximized_vert,maximized_horz`.

## What to verify (3 tabs)
1. **Tổng quan & So sánh (overview)** — auto-loads on page open. Confirm summary table has 2 rows and 4 plots render. Expected trained numbers (3000 eps, lr=3e-3): REINFORCE eval `133.6 ± 54.5`, solved=`chưa`; REINFORCE+Baseline eval `206.2 ± 90.1`, solved at ep `1405`. Bar chart: baseline crosses the Solved(200) dashed line, vanilla stays below.
2. **Xem agent chơi (watch-agent)** — select an algo + seed, click `▶️ Cho agent chơi`. Takes a few seconds to render. Confirm a GIF appears and a reward line shows (Baseline @ seed=42 ≈ `181.9`).
3. **Huấn luyện tương tác (interactive training)** — pick algo, set episodes/lr/gamma/hidden size, click `🚀 Bắt đầu huấn luyện`. Progress bar advances; on completion a live learning curve + eval text appear. A 300-episode run takes ~20s and the avg-100 curve should trend upward (won't reach +200 — that needs ~1400+ eps).

## Gotchas
- **lr=0.01 diverges.** The interactive tab defaults lr to 0.01 which makes REINFORCE diverge. Set lr to ~0.003 for a clean improving demo curve.
- **LunarLander-v3**, not v2 (slide's v2 is deprecated in modern gymnasium; drop-in compatible).
- **torch.load needs `weights_only=False`** (PyTorch 2.6+) because checkpoints store config ints alongside the state_dict.
- GIF rendering uses a headless render; if it hangs, confirm `box2d` is installed in the venv.
- Interactive training reward staying negative at low episode counts is expected, NOT a bug — judge by the upward trend, not the absolute value.

## Devin Secrets Needed
- None. Everything runs locally; no external auth required.
