#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, logging
from typing import List, Tuple, Any, Dict
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(ROOT, "draw_summary_graph.log")

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.figsize": (12, 5),
})

# ---------------- I/O ----------------
def load_final_json(path: str) -> Dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logging.warning(f"Failed to load {path}: {e}")
        return None

def collect_data(root: str) -> Tuple[List[str], List[float], List[float], List[float]]:
    names, sim, explored, elapsed = [], [], [], []
    for d in sorted(os.listdir(root)):
        folder = os.path.join(root, d)
        if not os.path.isdir(folder): continue
        fpath = os.path.join(folder, "final.json")
        if not os.path.exists(fpath): continue
        obj = load_final_json(fpath)
        if not isinstance(obj, dict): continue
        try:
            sim.append(float(obj["final_sim_score"]))
            explored.append(float(obj["explored_total"]))
            elapsed.append(float(obj["elapsed_sec"]))
            names.append(d)
        except Exception:
            logging.info(f"[skip] {d}: type error in final.json")
            continue
    return names, sim, explored, elapsed

# ------------- Bins & labels ----------
def round_bins_int(values: List[float], step: int) -> np.ndarray:
    lo, hi = min(values), max(values)
    lo = np.floor(lo / step) * step
    hi = np.ceil(hi / step) * step
    return np.arange(lo, hi + step, step, dtype=float)

def labels_int(edges: np.ndarray) -> List[str]:
    return [f"{int(edges[i])}–{int(edges[i+1])}" for i in range(len(edges)-1)]

def labels_float(edges: np.ndarray, decimals: int = 2) -> List[str]:
    fmt = f"{{:.{decimals}f}}–{{:.{decimals}f}}"
    return [fmt.format(edges[i], edges[i+1]) for i in range(len(edges)-1)]

# ------------- Plot helper ------------
def plot_hist_bar(values: List[float],
                  edges: np.ndarray,
                  x_label: str,
                  title_baseline: str,
                  out_file: str,
                  float_labels: bool = False,
                  decimals: int = 2):
    counts, _ = np.histogram(values, bins=edges)
    labels = labels_float(edges, decimals) if float_labels else labels_int(edges)

    fig, ax = plt.subplots()
    bars = ax.bar(np.arange(len(counts)), counts)
    ax.set_xticks(np.arange(len(counts)))
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_ylabel("Count")
    ax.set_xlabel(x_label)
    ax.set_title(title_baseline)  # baseline_数据项
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    # annotate counts
    for i, b in enumerate(bars):
        h = b.get_height()
        if h > 0:
            ax.annotate(f"{int(h)}", (b.get_x() + b.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(ROOT, out_file), dpi=150)
    plt.close(fig)
    logging.info(f"[ok] saved: {out_file}")

# ------------- Main -------------------
if __name__ == "__main__":
    names, sim, explored, elapsed = collect_data(ROOT)
    if not names:
        logging.info("No valid final.json found. Nothing to plot.")
        raise SystemExit(0)

    # 1) Similarity score: 0–1 with 0.05 step (小数，不化整)
    score_edges = np.arange(0.0, 1.0 + 1e-9, 0.05, dtype=float)
    plot_hist_bar(
        sim, score_edges,
        x_label="Final Similarity Score (0–1)",
        title_baseline="baseline_final_sim_score",
        out_file="baseline_final_sim_score.png",
        float_labels=True, decimals=2
    )

    # 2) Explored total: 整数分箱（默认 10 一个区间）
    exp_edges = round_bins_int(explored, step=10)
    plot_hist_bar(
        explored, exp_edges,
        x_label="Explored States (#)",
        title_baseline="baseline_explored_total",
        out_file="baseline_explored_total.png",
        float_labels=False
    )

    # 3) Elapsed sec: 整数分箱（默认 100s 一个区间）
    time_edges = round_bins_int(elapsed, step=100)
    plot_hist_bar(
        elapsed, time_edges,
        x_label="Elapsed Time (seconds)",
        title_baseline="baseline_elapsed_sec",
        out_file="baseline_elapsed_sec.png",
        float_labels=False
    )

    logging.info("✅ Summary graphs generated.")



