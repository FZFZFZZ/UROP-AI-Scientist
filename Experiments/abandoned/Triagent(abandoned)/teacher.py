# teacher.py
import os, json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List

from common import (
    DEFAULT_TEACHER_MODEL, read_jsonl, append_jsonl, add_common_args,
    retry, LOG_DIR, get_openai_client
)

TEACHER_SYSTEM = "You are a Socratic research mentor. Your goal is to guide the student to arrive at the correct research proposal without ever giving any important details away."

def build_teacher_user_prompt(ground_truth: str, student_text: str) -> str:
    return f"""
[Ground Truth]
{ground_truth}

[Student Proposal]
{student_text}

[Task]
You are a research advisor. Your job is to help the student arrive at the exact ground truth proposal above — through their own reasoning.

Rules:
You MUST NOT give the student the answer or exact keywords.

Be strict. Do NOT tolerate hand-wavy or unrelated solutions. This is a serious scientific training task.
""".strip()

def ask_teacher(ground_truth: str, student_text: str, client, model: str = DEFAULT_TEACHER_MODEL) -> str:
    def _call():
        resp = client.chat.completions.create(
            model=model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": TEACHER_SYSTEM},
                {"role": "user", "content": build_teacher_user_prompt(ground_truth, student_text)},
            ],
        )
        return resp.choices[0].message.content.strip()
    return retry(_call)

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Teacher feedback generator (OpenAI → gpt-4.1)")
    add_common_args(ap)
    ap.add_argument("--model", default=DEFAULT_TEACHER_MODEL)
    ap.add_argument("--ground-truth-file", required=True, help="Text file with ground truth proposal.")
    args = ap.parse_args()

    gt = Path(args.ground_truth_file).read_text(encoding="utf-8").strip()
    infile = Path(args.infile) if args.infile else None
    outfile = Path(args.outfile or (LOG_DIR / "teacher_out.jsonl"))
    if not infile or not infile.exists():
        ap.error("Provide --in JSONL (from student stage).")

    seen = set()
    if args.resume and outfile.exists():
        for row in read_jsonl(outfile):
            if "uid" in row: seen.add(row["uid"])

    client = get_openai_client()

    rows = []
    for row in read_jsonl(infile):
        if args.limit and len(rows) >= args.limit: break
        uid = row.get("uid")
        if not uid: continue
        if args.resume and uid in seen: continue
        student_text = row.get("student","").strip()
        rows.append({"uid": uid, "student": student_text})

    def work(r):
        fb = ask_teacher(gt, r["student"], client, model=args.model)
        return {"uid": r["uid"], "teacher_feedback": fb}

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, r) for r in rows]
        for f in as_completed(futs):
            res = f.result()
            append_jsonl(outfile, res)

    print(f"[DONE] Wrote {outfile}")

if __name__ == "__main__":
    main()
