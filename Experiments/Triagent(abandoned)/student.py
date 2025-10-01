# student.py
import os, json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List
import ollama

from common import (
    DEFAULT_STUDENT_MODEL, read_jsonl, append_jsonl, write_jsonl,
    add_common_args, retry, sha_uid, LOG_DIR
)

STUDENT_PROMPT_TEMPLATE = """You are a PhD student working in the field of AI and NLP.

You must propose a concise and novel research idea (1-3 sentences), given the problem context and advisor feedback. Directly output the idea.

# Few-shot example
Problem: There is a lack of principled understanding and effective automatic methods for selecting high-quality instruction tuning data to align large language models efficiently.,
Idea: Develop a framework that systematically measures instruction data along dimensions such as complexity, quality, and diversity, and introduces novel techniques for enhanced data assessment, enabling automatic selection of data samples for efficient instruction tuning.

# Your task
Problem: {problem}
Advisor feedback: {advisor}
"""

def ask_student(problem: str, teacher_feedback: str = None, model: str = DEFAULT_STUDENT_MODEL, temperature: float = 0.5) -> str:
    advisor_block = f"{teacher_feedback}" if teacher_feedback else ""
    prompt = STUDENT_PROMPT_TEMPLATE.format(problem=problem, advisor=advisor_block)

    def _call():
        resp = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature}
        )
        return resp["message"]["content"].strip()

    return retry(_call)

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Student generator (Ollama → llama3.3)")
    add_common_args(ap)
    ap.add_argument("--model", default=DEFAULT_STUDENT_MODEL)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--problem", default=None, help="Single problem string. If not set, read from --in.")
    ap.add_argument("--history", default=None, help="Optional JSONL with prior teacher feedback by uid.")
    args = ap.parse_args()

    outfile = Path(args.outfile or (LOG_DIR / "student_out.jsonl"))
    seen = set()
    if args.resume and outfile.exists():
        for row in read_jsonl(outfile):
            if "uid" in row: seen.add(row["uid"])

    items: List[Dict[str, Any]] = []
    if args.problem:
        uid = sha_uid({"problem": args.problem}, prefix="p_")
        items = [{"uid": uid, "problem": args.problem, "teacher_feedback": None}]
    else:
        infile = Path(args.infile) if args.infile else None
        if not infile or not infile.exists():
            ap.error("Provide --problem or --in JSONL.")
        for row in read_jsonl(infile):
            if args.limit and len(items) >= args.limit: break
            uid = row.get("uid") or sha_uid({"problem": row.get("problem","")}, prefix="p_")
            if args.resume and uid in seen:
                continue
            items.append({"uid": uid, "problem": row.get("problem",""), "teacher_feedback": row.get("teacher_feedback")})

    for it in items:
        out = ask_student(it["problem"], it.get("teacher_feedback"), model=args.model, temperature=args.temperature)
        append_jsonl(outfile, {"uid": it["uid"], "problem": it["problem"], "student": out})

    print(f"[DONE] Wrote {outfile}")

if __name__ == "__main__":
    main()
