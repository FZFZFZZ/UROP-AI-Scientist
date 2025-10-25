# Experiments/evaluator/main.py
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

# --- Helpers provided elsewhere in your repo ---
from judge_match import judge_match
from judge_search import judge_search

DEFAULT_BASE = Path("../Data/MDP/mechanism_exploration/log/baseline")
SUMMARY_FILE = DEFAULT_BASE / "summary.jsonl"
DEFAULT_OUT = DEFAULT_BASE / "result_separate.jsonl"

def read_jsonl(path: Path) -> Iterable[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                print(f"[warn] {path.name}:{ln} JSON decode error: {e}", file=sys.stderr)


def load_summary(path: Path = SUMMARY_FILE) -> Dict[str, Dict]:
    idx: Dict[str, Dict] = {}
    for obj in read_jsonl(path):
        _id = obj.get("id") or obj.get("or_id")
        if _id:
            idx[_id] = obj
    return idx


def make_result_record(or_id: str, rec: Dict) -> Optional[Dict]:
    """
    Build a single result record:
    {
      "id": "...",
      "final_sim_score": <float|None>,
      "searchable": <bool>,
      "verdict": "SAME|NEARLY_SAME|RELATED_BUT_DISTINCT|DIFFERENT",
      "confidence": <float|None>
    }
    Returns None if final idea missing.
    """
    final_idea = (rec.get("final idea") or "").strip()
    if not final_idea:
        return None

    abstract = rec.get("context", "")
    target_idea = rec.get("target idea", "")
    final_score = rec.get("final sim score", None)

    searchable = judge_search(or_id, final_idea)
    match_result = judge_match(abstract, target_idea, final_idea)

    print(or_id + " DONE")

    return {
        "id": or_id,
        "final_sim_score": final_score,
        "searchable": bool(searchable),
        "verdict": match_result.get("verdict"),
        "confidence": match_result.get("confidence"),
    }


def write_results(idx: Dict[str, Dict], out_path: Path, limit: Optional[int] = None):
    ids = list(idx.keys())
    if limit is not None:
        ids = ids[:limit]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Overwrite
    with out_path.open("w", encoding="utf-8") as fout:
        for or_id in ids:
            rec = idx[or_id]
            try:
                result = make_result_record(or_id, rec)
                if result is None:
                    continue
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"[error] id={or_id}: {e}", file=sys.stderr)


def run_single(or_id: str, base_dir: Path, out_path: Optional[Path]):
    idx = load_summary(base_dir / "summary.jsonl")
    rec = idx.get(or_id)
    if not rec:
        raise SystemExit(f"[error] id '{or_id}' not found in {base_dir / 'summary.jsonl'}")

    result = make_result_record(or_id, rec)
    if result is None:
        raise SystemExit(f"[error] id '{or_id}' missing 'final idea'")

    if out_path:
        # Write exactly one line to file (overwrite if path exists)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fout:
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
    else:
        # Quiet stdout: just the JSON line for piping
        print(json.dumps(result, ensure_ascii=False))


def run_all(base_dir: Path, out_path: Path, limit: Optional[int]):
    idx = load_summary(base_dir / "summary.jsonl")
    write_results(idx, out_path, limit=limit)


def main():
    p = argparse.ArgumentParser(description="Evaluator + writer for baseline logs (quiet).")
    p.add_argument("--base", type=str, default=str(DEFAULT_BASE),
                   help="Base dir containing summary.jsonl and <id>/trace.jsonl")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--id", type=str, help="Specific id to evaluate and emit a single JSON line")
    group.add_argument("--all", action="store_true", help="Process all ids in summary.jsonl")
    p.add_argument("--limit", type=int, default=None, help="Limit number of ids when using --all")
    p.add_argument("--out", type=str, default=None,
                   help="Output file path. Defaults to ../Data/.../result_separate.jsonl (for --all). "
                        "For --id, if omitted, prints single JSON line to stdout.")
    args = p.parse_args()

    base_dir = Path(args.base)
    out_path = Path(args.out) if args.out else (DEFAULT_OUT if args.all else None)

    if args.id:
        run_single(args.id, base_dir, out_path)
    else:
        run_all(base_dir, out_path, args.limit)


if __name__ == "__main__":
    main()



