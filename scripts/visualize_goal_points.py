"""
Reproduce the Goal Point distribution heatmaps from the GoalFlow README.

Produces 3 rows × N columns (one column per scene):
  Row 0 — DAC score distribution
  Row 1 — Distance (IM) score distribution
  Row 2 — Final combined score

Usage:
    python scripts/visualize_goal_points.py \
        --tokens 0000548db87959c2 0a44947ca9e85579 \   # optional: specific tokens
        --num_scenes 4 \                                # or pick N random ones
        --out_dir exp/viz_goal_points

Output:
    exp/viz_goal_points/goal_point_grid.png   — full 3×N paper figure
    exp/viz_goal_points/<token>_dac.png       — individual DAC heatmap
    exp/viz_goal_points/<token>_im.png        — individual IM heatmap
    exp/viz_goal_points/<token>_final.png     — individual final heatmap
"""

import os
import sys
import argparse
import random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec

ROOT = Path(os.environ.get("NAVSIM_DEVKIT_ROOT",
            Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))

DAC_DIR  = ROOT / "data/goal_point_scores/dac"
IM_DIR   = ROOT / "data/goal_point_scores/im"
VOC_PATH = ROOT / "data/cluster_points_8192_.npy"


# ── helpers ────────────────────────────────────────────────────────────────────

def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    x = x / temperature
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def final_score(dac: np.ndarray, im: np.ndarray,
                dac_weight: float = 0.8, im_weight: float = 0.2) -> np.ndarray:
    """Reproduce GoalFlow's scoring:  score = dac_weight*dac + im_weight*im"""
    dac_norm = (dac - dac.min()) / (dac.max() - dac.min() + 1e-8)
    im_norm  = (im  - im.min())  / (im.max()  - im.min()  + 1e-8)
    return dac_weight * dac_norm + im_weight * im_norm


def scatter_heatmap(ax: plt.Axes,
                    voc: np.ndarray,
                    scores: np.ndarray,
                    title: str = "",
                    cmap: str = "RdYlGn",
                    point_size: float = 8.0) -> None:
    """
    Draw scatter heatmap of 8192 vocabulary goal points coloured by score.
    voc: (8192, 3)  [x=forward, y=lateral, heading]
    scores: (8192,) normalised to [0,1]
    """
    ax.set_facecolor("#111111")

    # slight alpha so dense clusters are visible
    sc = ax.scatter(
        voc[:, 1],   # lateral  → x axis
        voc[:, 0],   # forward  → y axis
        c=scores,
        cmap=cmap,
        s=point_size,
        alpha=0.75,
        linewidths=0,
        vmin=0, vmax=1,
    )

    # best point (argmax)
    best = np.argmax(scores)
    ax.scatter(voc[best, 1], voc[best, 0],
               s=80, c="white", marker="*", zorder=5)

    # ego
    ax.scatter(0, 0, s=60, c="#00e5ff", marker="^", zorder=6)

    ax.set_xlim(-30, 30)
    ax.set_ylim(-10, 55)
    ax.set_aspect("equal")
    ax.set_title(title, color="white", fontsize=9, pad=3)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    return sc


def available_tokens() -> list:
    return [p.stem for p in sorted(DAC_DIR.glob("*.npy"))]


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens",     nargs="*", default=None,
                        help="Specific token(s); overrides --num_scenes")
    parser.add_argument("--num_scenes", type=int, default=4)
    parser.add_argument("--seed",       type=int, default=0)
    parser.add_argument("--dac_weight", type=float, default=0.8)
    parser.add_argument("--im_weight",  type=float, default=0.2)
    parser.add_argument("--cmap",       default="RdYlGn",
                        help="Matplotlib colormap (RdYlGn matches paper)")
    parser.add_argument("--out_dir",    default=str(ROOT / "exp/viz_goal_points"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    voc = np.load(VOC_PATH)          # (8192, 3)
    print(f"Vocabulary: {voc.shape}")

    # pick tokens
    if args.tokens:
        tokens = args.tokens
    else:
        all_tokens = available_tokens()
        random.seed(args.seed)
        tokens = random.sample(all_tokens, min(args.num_scenes, len(all_tokens)))

    print(f"Rendering {len(tokens)} scenes …")

    # ── per-scene individual PNGs ───────────────────────────────────────────
    all_data = []
    for token in tokens:
        dac_path = DAC_DIR / f"{token}.npy"
        im_path  = IM_DIR  / f"{token}.npy"
        if not dac_path.exists():
            print(f"  ⚠ no scores for {token}, skipped")
            continue

        dac_raw = np.load(dac_path).astype(np.float32)
        im_raw  = np.load(im_path ).astype(np.float32)

        # normalise each to [0,1]
        dac_norm = (dac_raw - dac_raw.min()) / (dac_raw.max() - dac_raw.min() + 1e-8)
        im_norm  = (im_raw  - im_raw.min())  / (im_raw.max()  - im_raw.min()  + 1e-8)
        fin_norm = args.dac_weight * dac_norm + args.im_weight * im_norm
        fin_norm = (fin_norm - fin_norm.min()) / (fin_norm.max() - fin_norm.min() + 1e-8)

        all_data.append((token, dac_norm, im_norm, fin_norm))

        # individual PNGs
        for tag, scores in [("dac", dac_norm), ("im", im_norm), ("final", fin_norm)]:
            fig, ax = plt.subplots(1, 1, figsize=(4, 5), facecolor="#111111")
            sc = scatter_heatmap(ax, voc, scores,
                                 title={"dac": "DAC Score",
                                        "im":  "Distance Score",
                                        "final": "Final Score"}[tag],
                                 cmap=args.cmap)
            plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color="white")
            plt.setp(plt.getp(plt.colorbar(sc, ax=ax).ax.axes, 'yticklabels'), color='white')
            fig.savefig(out_dir / f"{token}_{tag}.png",
                        dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)

        print(f"  [{tokens.index(token)+1}/{len(tokens)}] {token}")

    if not all_data:
        print("No valid tokens found.")
        return

    # ── combined 3 × N grid (paper style) ──────────────────────────────────
    N = len(all_data)
    fig = plt.figure(figsize=(N * 3.5, 11), facecolor="#0a0a0a")
    gs  = GridSpec(3, N, figure=fig, hspace=0.06, wspace=0.04,
                   left=0.02, right=0.92, top=0.94, bottom=0.02)

    row_labels = ["DAC Score", "Distance Score", "Final Score"]
    row_data_keys = [1, 2, 3]  # indices into all_data tuple

    for col, (token, dac_n, im_n, fin_n) in enumerate(all_data):
        for row, (scores, label) in enumerate(
            zip([dac_n, im_n, fin_n], row_labels)
        ):
            ax = fig.add_subplot(gs[row, col])
            sc = scatter_heatmap(ax, voc, scores, cmap=args.cmap)

            if col == 0:
                ax.set_ylabel(label, color="white", fontsize=10)
                ax.yaxis.set_visible(True)
                ax.set_yticks([])

            if row == 0:
                ax.set_title(token[:16] + "…", color="#aaa", fontsize=8)

    # shared colorbar
    cbar_ax = fig.add_axes([0.93, 0.1, 0.015, 0.8])
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    cb = matplotlib.colorbar.ColorbarBase(
        cbar_ax, cmap=cm.get_cmap(args.cmap), norm=norm)
    cb.ax.yaxis.set_tick_params(color="white", labelsize=9)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
    cb.set_label("Score", color="white", fontsize=10)

    fig.suptitle("GoalFlow — Goal Point Score Distribution",
                 color="white", fontsize=13, y=0.97)

    grid_path = out_dir / "goal_point_grid.png"
    fig.savefig(grid_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"\n✅ Grid saved → {grid_path}")
    print(f"   Individual PNGs → {out_dir}/")


if __name__ == "__main__":
    main()
