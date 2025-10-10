import json
import argparse
import statistics
from typing import Dict

from FFLM import FFLM
from Flesch import Flesch

def load_id_to_abstract(path: str) -> Dict[str, str]:
    """Load a JSONL file where each line has at least: {"id": "...", "abstract": "..."}"""
    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                pid = rec.get("id")
                abs_ = rec.get("abstract")
                if isinstance(pid, str) and isinstance(abs_, str) and abs_.strip():
                    mapping[pid] = abs_.strip()
            except json.JSONDecodeError as e:
                print(f"⚠️  Skipping malformed JSON at {path}:{ln}: {e}")
    return mapping

def main():
    ap = argparse.ArgumentParser(
        description="Compute FFLM (Context/Idea vs abstract), Perplexity, and Flesch; print averages only."
    )
    ap.add_argument("--in", dest="in_path", required=True,
                    help="Input JSONL with fields: id, Context, Idea")
    ap.add_argument("--abstracts", dest="abstracts_path", required=True,
                    help="Target JSONL with fields: id, abstract (for source lookup)")
    ap.add_argument("--fflm_model", default="gpt-4o", help="Model for FFLM() (default: gpt-4o)")
    args = ap.parse_args()

    id2abs = load_id_to_abstract(args.abstracts_path)
    if not id2abs:
        raise RuntimeError(f"No abstracts loaded from {args.abstracts_path}")

    # collectors
    n = 0
    fflm_c_vals, fflm_i_vals = [], []
    flesch_c_vals, flesch_i_vals = [], []
    len_c_words, len_i_words = [], []

    with open(args.in_path, "r", encoding="utf-8") as fin:
        for ln, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"⚠️  Skipping malformed JSON at {args.in_path}:{ln}: {e}")
                continue

            pid = rec.get("id")
            context = (rec.get("Context") or "").strip()
            idea    = (rec.get("Idea") or "").strip()

            if not pid or not context or not idea:
                print(f"⚠️  Missing id/Context/Idea at line {ln}; skipping.")
                continue

            abstract = id2abs.get(pid)
            if not abstract:
                print(f"⚠️  No abstract found for id={pid}; skipping.")
                continue

            try:
                # --- FFLM: source = abstract; summaries = Context / Idea
                fflm_val_c = FFLM(source=abstract, summary=context, model=args.fflm_model)
                fflm_val_i = FFLM(source=abstract, summary=idea,    model=args.fflm_model)

                fflm_c_vals.append(fflm_val_c)
                fflm_i_vals.append(fflm_val_i)


                flesch_c = Flesch(context)
                flesch_i = Flesch(idea)
                flesch_c_vals.append(flesch_c)
                flesch_i_vals.append(flesch_i)

                len_c_words.append(len(context.split()))
                len_i_words.append(len(idea.split()))
                n += 1

                if n % 10 == 0:
                    print(f"Processed {n} records...")

            except Exception as e:
                print(f"⚠️  Error for id={pid}: {type(e).__name__}: {e}")

    # --- Summary (averages only) ---
    print(fflm_c_vals)
    print(fflm_i_vals)
    print(flesch_c_vals)
    print(flesch_i_vals)
    print(len_c_words)
    print(len_i_words)
    print("\n=== SUMMARY (averages) ===")
    print(f"Records processed: {n}")
    if len_c_words:
        print(f"Avg word count — Context: {statistics.mean(len_c_words):.1f}, "
              f"Idea: {statistics.mean(len_i_words):.1f}")
    if fflm_c_vals:
        print(f"Avg FFLM (abstract vs Context): {statistics.mean(fflm_c_vals):.4f}")
    if fflm_i_vals:
        print(f"Avg FFLM (abstract vs Idea):    {statistics.mean(fflm_i_vals):.4f}")
    if flesch_c_vals:
        print(f"Avg Flesch — Context: {statistics.mean(flesch_c_vals):.2f}")
    if flesch_i_vals:
        print(f"Avg Flesch — Idea:    {statistics.mean(flesch_i_vals):.2f}")

if __name__ == "__main__":
    main()

# e.g. python Experiments/context_idea_extraction/analyse.py --in Data/context_idea_extraction/short_gpt-4o-mini.jsonl --abstracts Data/context_idea_extraction/data.jsonl
