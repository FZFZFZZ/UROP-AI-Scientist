#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import logging
import matplotlib.pyplot as plt
from collections import Counter

LOG_PATH = "draw_graph.log"
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def safe_load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            try:
                obj = json.loads(line.strip())
                if isinstance(obj, dict):
                    rows.append(obj)
            except:
                logging.warning(f"{path} line {i}: JSON decode error, skipped.")
    return rows

def most_common_id(rows):
    ids = [str(r.get("id")) for r in rows if r.get("id")]
    if not ids:
        return ""
    return Counter(ids).most_common(1)[0][0]

def extract_values(rows, fields):
    """fields: list of keys required"""
    vals = []
    for r in rows:
        try:
            if all(field in r for field in fields):
                entry = [r[f] for f in fields]
                vals.append(entry)
        except:
            continue
    return vals

def plot_and_save(folder, x, y_dict, title, filename):
    plt.figure(figsize=(8, 5))
    for label, y in y_dict.items():
        plt.plot(x, y, marker="o", label=label)
    plt.xlabel("round")
    plt.title(title)
    plt.legend()
    plt.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()
    out_path = os.path.join(folder, filename)
    plt.savefig(out_path, dpi=120)
    plt.close()

def main():
    root = os.path.dirname(os.path.abspath(__file__))
    dirs = [d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))]

    logging.info(f"Total folders found: {len(dirs)}")
    
    for d in dirs:
        folder = os.path.join(root, d)
        trace = os.path.join(folder, "trace.jsonl")

        if not os.path.exists(trace):
            logging.info(f"[skip] {d}: no trace.jsonl")
            continue
        
        try:
            rows = safe_load_jsonl(trace)
            if not rows:
                logging.info(f"[skip] {d}: no valid JSON lines in trace.jsonl")
                continue

            title = most_common_id(rows) or d

            # Plot 1: round_best_score & best_score_so_far
            data1 = extract_values(rows, ["round", "round_best_score", "best_score_so_far"])
            if data1:
                data1_sorted = sorted(data1, key=lambda x: x[0])
                x = [int(z[0]) for z in data1_sorted]
                y1 = [float(z[1]) for z in data1_sorted]
                y2 = [float(z[2]) for z in data1_sorted]
                plot_and_save(folder, x, {"round_best_score": y1, "best_score_so_far": y2},
                              f"{title} - Scores", "trace_scores.png")
            else:
                logging.info(f"[skip-plot1] {d}: missing score fields")

            # Plot 2: #explored_so_far
            data2 = extract_values(rows, ["round", "#explored_so_far"])
            if data2:
                data2_sorted = sorted(data2, key=lambda x: x[0])
                x = [int(z[0]) for z in data2_sorted]
                y = [float(z[1]) for z in data2_sorted]
                plot_and_save(folder, x, {"#explored_so_far": y},
                              f"{title} - Explored", "trace_explored.png")
            else:
                logging.info(f"[skip-plot2] {d}: missing explored fields")

            logging.info(f"[ok] {d}: graphs generated")

        except Exception as e:
            logging.error(f"[error] {d}: {e}")

    logging.info("All done. Logs in draw_graph.log")

if __name__ == "__main__":
    main()
