from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def run(project_dir: Path) -> Path:
    in_csv = project_dir / "reports" / "current" / "model_diagnostics" / "threshold_tuning_applied_changes.csv"
    out_png = project_dir / "reports" / "current" / "model_diagnostics" / "threshold_tuning_changes.png"

    if not in_csv.exists():
        raise FileNotFoundError(f"Missing input CSV: {in_csv}")

    df = pd.read_csv(in_csv).sort_values("dataset").reset_index(drop=True)

    x = np.arange(len(df))
    width = 0.36

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left panel: threshold change (old vs new moderate)
    bars_old = axes[0].bar(x - width / 2, df["old_moderate"], width=width, label="Old moderate", color="#4e79a7")
    bars_new = axes[0].bar(x + width / 2, df["new_moderate"], width=width, label="New moderate", color="#f28e2b")
    axes[0].set_title("Moderate Threshold Update")
    axes[0].set_ylabel("Threshold value")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df["dataset"].str.upper())
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis="y", linestyle="--", alpha=0.25)
    axes[0].legend(loc="upper right")

    for bars in (bars_old, bars_new):
        for b in bars:
            h = b.get_height()
            axes[0].text(b.get_x() + b.get_width() / 2, h + 0.02, f"{h:.2f}", ha="center", va="bottom", fontsize=9)

    # Right panel: performance gain (F1 and accuracy)
    f1_gain = df["best_f1"] - df["default_f1"]
    acc_gain = df["best_accuracy"] - df["default_accuracy"]

    bars_f1 = axes[1].bar(x - width / 2, f1_gain, width=width, label="F1 gain", color="#59a14f")
    bars_acc = axes[1].bar(x + width / 2, acc_gain, width=width, label="Accuracy gain", color="#e15759")
    axes[1].set_title("Performance Improvement After Tuning")
    axes[1].set_ylabel("Absolute gain")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df["dataset"].str.upper())
    ymin = min(-0.02, float(min(f1_gain.min(), acc_gain.min())) - 0.02)
    ymax = max(0.02, float(max(f1_gain.max(), acc_gain.max())) + 0.05)
    axes[1].set_ylim(ymin, ymax)
    axes[1].axhline(0, color="black", linewidth=1, alpha=0.6)
    axes[1].grid(axis="y", linestyle="--", alpha=0.25)
    axes[1].legend(loc="upper left")

    for bars in (bars_f1, bars_acc):
        for b in bars:
            h = b.get_height()
            axes[1].text(
                b.get_x() + b.get_width() / 2,
                h + (0.01 if h >= 0 else -0.015),
                f"{h:+.3f}",
                ha="center",
                va="bottom" if h >= 0 else "top",
                fontsize=9,
            )

    fig.suptitle("Threshold Tuning Change Summary", y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    plt.close(fig)

    return out_png


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent.parent
    out = run(project_dir)
    print(f"Saved graph: {out}")
