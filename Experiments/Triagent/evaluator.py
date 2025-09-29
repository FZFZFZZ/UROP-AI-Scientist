# evaluator.py
import os, json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List
from openai import OpenAI

from common import (
    DEFAULT_EVALUATOR_MODEL, read_jsonl, append_jsonl, add_common_args,
    retry, LOG_DIR
)

EVAL_SYSTEM = "You are a strict research evaluator. You must only return valid JSON."

EVAL_USER_TEMPLATE = """[Ground Truth Proposal]
{ground_truth}

[Student's Attempt]
{student}

[Task]
Is the student's proposal semantically equivalent in technical content and intent to the ground truth? The following key concept must be specifically present: "null space of preserved knowledge"

Respond ONLY in this valid JSON format:
{{
  "aligned": true or false,
  "reason": "short explanation here"
}}

DO NOT include any preamble or explanation outside the JSON.
"""

def external_judge(student_response: str, ground_truth: str, client: OpenAI, model: str = DEFAULT_EVALUATOR_MODEL) -> Dict[str, Any]:
    def _call():
        resp = client.chat.completions.create(
            model=model,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": EVAL_SYSTEM},
                {"role": "user", "content": EVAL_USER_TEMPLATE.format(ground_truth=ground_truth, student=student_response)},
            ],
        )
        content = resp.choices[0].message.content.strip()
        return json.loads(content)
    return retry(_call)

def similarity_score(text1: str, text2: str) -> float:
    try:
        from sentence_transformers import SentenceTransformer, util
        model = SentenceTransformer('all-mpnet-base-v2')
        emb = model.encode([text1, text2], convert_to_tensor=True)
        score = util.cos_sim(emb[0], emb[1]).item()
        return float(score)
    except Exception:
        return -1.0  # similarity disabled

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Evaluator (OpenAI → gpt-4.1)")
    add_common_args(ap)
    ap.add_argument("--model", default=DEFAULT_EVALUATOR_MODEL)
    ap.add_argument("--ground-truth-file", required=True)
    ap.add_argument("--with-sim", action="store_true", help="Compute similarity if sentence-transformers is available.")
    args = ap.parse_args()

    gt = Path(args.ground_truth_file).read_text(encoding="utf-8").strip()
    infile = Path(args.infile) if args.infile else None
    outfile = Path(args.outfile or (LOG_DIR / "evaluator_out.jsonl"))
    if not infile or not infile.exists():
        ap.error("Provide --in JSONL (student outputs).")

    seen = set()
    if args.resume and outfile.exists():
        for row in read_jsonl(outfile):
            if "uid" in row: seen.add(row["uid"])

    client = OpenAI()

    rows = []
    for row in read_jsonl(infile):
        if args.limit and len(rows) >= args.limit: break
        uid = row.get("uid")
        if not uid: continue
        if args.resume and uid in seen: continue
        student_text = row.get("student","").strip()
        rows.append({"uid": uid, "student": student_text})

    def work(r):
        verdict = external_judge(r["student"], gt, client, model=args.model)
        out = {"uid": r["uid"], "judge": verdict}
        if args.with_sim:
            out["similarity"] = similarity_score(r["student"], gt)
        return out

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, r) for r in rows]
        for f in as_completed(futs):
            res = f.result()
            append_jsonl(outfile, res)

    print(f"[DONE] Wrote {outfile}")

if __name__ == "__main__":
    main()
