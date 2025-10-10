#!/usr/bin/env python3
import os, json, argparse, traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional

# local modules (from previous step)
import student as student_mod
import teacher as teacher_mod
import evaluator as evaluator_mod
from common import (
    LOG_DIR, read_jsonl, append_jsonl, add_common_args, sha_uid, retry,
    DEFAULT_STUDENT_MODEL, DEFAULT_TEACHER_MODEL, DEFAULT_EVALUATOR_MODEL, get_openai_client
)

def run_one(
    uid: str,
    problem: str,
    ground_truth: str,
    max_iters: int,
    student_model: str,
    teacher_model: str,
    evaluator_model: str,
    student_temp: float,
    with_sim: bool,
) -> Dict[str, Any]:
    """
    Orchestrate student→evaluator→(teacher feedback) loop for a single uid/problem.
    Returns a final record with fields:
      uid, problem, aligned, iters, student, judge, (similarity?)
    """
    client = get_openai_client()
    last_feedback: Optional[str] = None
    final_student = ""
    final_judge: Dict[str, Any] = {}
    similarity = None

    for it in range(1, max_iters + 1):
        # --- Student
        final_student = student_mod.ask_student(
            problem=problem,
            teacher_feedback=last_feedback,
            model=student_model,
            temperature=student_temp,
        )

        # --- Evaluate
        final_judge = evaluator_mod.external_judge(
            student_response=final_student,
            ground_truth=ground_truth,
            client=client,
            model=evaluator_model,
        )
        aligned = bool(final_judge.get("aligned", False))

        # Optional similarity
        if with_sim and similarity is None:
            try:
                similarity = evaluator_mod.similarity_score(final_student, ground_truth)
            except Exception:
                similarity = -1.0

        # Log each iteration (optional: write per-iter trace)
        append_jsonl(LOG_DIR / "orchestrator_iters.jsonl", {
            "uid": uid,
            "iter": it,
            "aligned": aligned,
            "judge": final_judge,
            "student": final_student,
            "problem": problem,
        })

        if aligned:
            return {
                "uid": uid,
                "problem": problem,
                "aligned": True,
                "iters": it,
                "student": final_student,
                "judge": final_judge,
                **({"similarity": similarity} if similarity is not None else {}),
            }

        # --- Teacher feedback (only if not aligned)
        last_feedback = teacher_mod.ask_teacher(
            ground_truth=ground_truth,
            student_text=final_student,
            client=client,
            model=teacher_model,
        )

    # If we exit loop without alignment
    return {
        "uid": uid,
        "problem": problem,
        "aligned": False,
        "iters": max_iters,
        "student": final_student,
        "judge": final_judge,
        **({"similarity": similarity} if similarity is not None else {}),
    }

def main():
    ap = argparse.ArgumentParser(description="Student–Teacher–Evaluator Orchestrator")
    # basic IO / resume / workers
    add_common_args(ap)
    # orchestration-specific
    ap.add_argument("--ground-truth-file", required=True, help="Path to ground_truth.txt")
    ap.add_argument("--max-iters", type=int, default=6, help="Max refinement iterations per problem")
    ap.add_argument("--student-model", default=DEFAULT_STUDENT_MODEL)
    ap.add_argument("--teacher-model", default=DEFAULT_TEACHER_MODEL)
    ap.add_argument("--evaluator-model", default=DEFAULT_EVALUATOR_MODEL)
    ap.add_argument("--student-temp", type=float, default=1.0)
    ap.add_argument("--with-sim", action="store_true", help="Compute semantic similarity (if library available)")
    ap.add_argument("--problem", default=None, help="Run a single problem string; otherwise use --in JSONL.")
    args = ap.parse_args()

    gt = Path(args.ground_truth_file).read_text(encoding="utf-8").strip()
    outfile = Path(args.outfile or (LOG_DIR / "orchestrator_out.jsonl"))
    seen = set()
    if args.resume and outfile.exists():
        for row in read_jsonl(outfile):
            if "uid" in row:
                seen.add(row["uid"])

    # Build workload
    tasks: List[Dict[str, Any]] = []
    if args.problem:
        uid = sha_uid({"problem": args.problem}, prefix="p_")
        if not (args.resume and uid in seen):
            tasks.append({"uid": uid, "problem": args.problem})
    else:
        infile = Path(args.infile) if args.infile else None
        if not infile or not infile.exists():
            ap.error("Provide --problem or --in JSONL.")
        for row in read_jsonl(infile):
            if args.limit and len(tasks) >= args.limit:
                break
            prob = row.get("problem", "").strip()
            if not prob:
                continue
            uid = row.get("uid") or sha_uid({"problem": prob}, prefix="p_")
            if args.resume and uid in seen:
                continue
            tasks.append({"uid": uid, "problem": prob})

    # Process in parallel across problems
    def work(t):
        try:
            res = run_one(
                uid=t["uid"],
                problem=t["problem"],
                ground_truth=gt,
                max_iters=args.max_iters,
                student_model=args.student_model,
                teacher_model=args.teacher_model,
                evaluator_model=args.evaluator_model,
                student_temp=args.student_temp,
                with_sim=args.with_sim,
            )
        except Exception as e:
            res = {"uid": t["uid"], "problem": t["problem"], "error": str(e), "trace": traceback.format_exc()}
        append_jsonl(outfile, res)
        return res

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, t) for t in tasks]
        for _ in as_completed(futs):
            pass

    print(f"[DONE] Wrote {outfile}")

if __name__ == "__main__":
    main()
