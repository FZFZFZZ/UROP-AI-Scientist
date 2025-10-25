# -*- coding: utf-8 -*-
"""
Run idea exploration over a JSONL dataset and log results.

Usage: go to Experiment directory:
  python -m MDP.search_mechanism.main \
      --data ./Data/MDP/mechanism_exploration/extracted.jsonl \
      --out  ./Data/MDP/mechanism_exploration/log/{experiment name} \
      --rounds 40 \
      --fanout 8 \
      --workers 10
"""



from __future__ import annotations
import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# from MDP.search_mechanism.baseline import hill_climb, generate_start
from MDP.search_mechanism.taboo import taboo_climb, generate_start

# -----------------------------
# Utilities
# -----------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except Exception:
                sys.stderr.write(f"[warn] bad jsonl line (skipped): {line[:120]}...\n")
    return data

def write_jsonl(path: str, rows: List[Dict[str, Any]], *, append: bool = True) -> None:
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def write_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

# -----------------------------
# Per-sample job
# -----------------------------

def run_one_sample(
    sample: Dict[str, Any],
    out_root: str,
    rounds: int,
    fanout: int,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run hill_climb for a single JSONL item and write per-id logs.
    Returns a summary row for summary.jsonl.
    """
    sid = str(sample.get("id", "unknown")).strip()
    context = str(sample.get("Context", "")).strip()
    target_idea = str(sample.get("Idea", "")).strip()

    if not sid or not context or not target_idea:
        raise ValueError(f"Sample missing required fields: {sample}")

    # Prepare per-id directory
    id_dir = os.path.join(out_root, sid)
    ensure_dir(id_dir)

    # Get initial idea
    try:
        initial_idea = generate_start(context).strip()
        if not initial_idea:
            raise ValueError("Empty initial idea from generate_start()")
    except Exception as e:
        # fallback: use target as starting point to keep pipeline running
        if verbose:
            sys.stderr.write(f"[{sid}] generate_start failed ({e}), fallback to target Idea as initial.\n")
        initial_idea = target_idea

    # Run search
    t0 = time.time()
    result = taboo_climb(
        initial_idea=initial_idea,
        target_idea=target_idea,
        context=context,
        rounds=rounds,
        fanout=fanout,
        verbose=False,
    )
    dt = time.time() - t0

    # Parse trace for per-round logging
    trace: List[Dict[str, Any]] = result.get("trace", [])

    explored_cum = 0
    trace_rows: List[Dict[str, Any]] = []
    for step in trace:
        # step keys: round, round_best, round_best_score, best_so_far, best_score_so_far, candidates=[{idea,score}, ...]
        cands = step.get("candidates", []) or []
        explored_cum += len(cands)

        trace_rows.append({
            "id": sid,
            "round": step.get("round", None),
            "round_best": step.get("round_best", ""),
            "round_best_score": step.get("round_best_score", None),
            "best_so_far": step.get("best_so_far", ""),
            "best_score_so_far": step.get("best_score_so_far", None),
            "taboo_list": step.get("taboo_list", None),
            "taboo_length": step.get("taboo_length", None),
            "good_list": step.get("good_list", None),
            "good_length": step.get("good_length", None),
            "#explored_so_far": explored_cum,
            "#cands_this_round": len(cands),
        })

    # Write per-id logs
    write_jsonl(os.path.join(id_dir, "trace.jsonl"), trace_rows, append=False)
    final_obj = {
        "id": sid,
        "target_idea": target_idea,
        "context": context,
        "initial_idea": initial_idea,
        "final_idea": result.get("best_idea_overall", ""),
        "final_sim_score": result.get("best_score_overall", None),
        "final_taboo_list": result.get("taboo_list", ""),
        "final_good_list": result.get("good_list", ""),
        "rounds_run": result.get("rounds_run", rounds),
        "explored_total": explored_cum,
        "elapsed_sec": dt,
    }
    write_json(os.path.join(id_dir, "final.json"), final_obj)
    write_text(
        os.path.join(id_dir, "best.txt"),
        f"Best score: {final_obj['final_sim_score']}\nBest idea:\n{final_obj['final_idea']}\n",
    )

    # Summary row for summary.jsonl
    summary_row = {
        "id": sid,
        "target idea": target_idea,
        "context": context,
        "final idea": final_obj["final_idea"],
        "final sim score": final_obj["final_sim_score"],
    }
    return summary_row

# -----------------------------
# Main
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str,
                   default="../Data/MDP/mechanism_exploration/extracted.jsonl",
                   help="Input JSONL with fields: id, Context, Idea")
    p.add_argument("--out", type=str,
                   default="../Data/MDP/mechanism_exploration/log/taboo",
                   help="Output root directory for logs")
    p.add_argument("--rounds", type=int, default=40)
    p.add_argument("--fanout", type=int, default=8)
    p.add_argument("--workers", type=int, default=10, help="Thread workers")
    p.add_argument("--limit", type=int, default=0, help="Optional cap on number of samples")
    p.add_argument("--shuffle", action="store_true", help="Shuffle dataset before run")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    ensure_dir(args.out)

    # Big summary JSONL path
    summary_path = os.path.join(args.out, "summary.jsonl")

    # Load data
    data = read_jsonl(args.data)
    if not data:
        sys.stderr.write(f"[error] No data loaded from {args.data}\n")
        sys.exit(1)

    if args.shuffle:
        import random
        random.shuffle(data)
    if args.limit and args.limit > 0:
        data = data[:args.limit]

    lock = Lock()
    futures = []
    results: List[Dict[str, Any]] = []

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for sample in data:
                futures.append(
                    ex.submit(run_one_sample, sample, args.out, args.rounds, args.fanout, False)
                )
            for fu in as_completed(futures):
                try:
                    row = fu.result()
                    results.append(row)
                    # Append to summary.jsonl immediately (thread-safe)
                    with lock:
                        write_jsonl(summary_path, [row], append=True)
                except Exception as e:
                    sys.stderr.write(f"[error] sample failed: {e}\n")
    else:
        for sample in data:
            try:
                row = run_one_sample(sample, args.out, args.rounds, args.fanout, False)
                results.append(row)
                write_jsonl(summary_path, [row], append=True)
            except Exception as e:
                sys.stderr.write(f"[error] sample failed: {e}\n")

    print(f"[done] processed {len(results)} samples; summary -> {summary_path}")

if __name__ == "__main__":
    main()
