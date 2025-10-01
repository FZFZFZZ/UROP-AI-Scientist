#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch hill-climbing over a JSONL of {custom_id, problem, idea}.
For each line, we treat:
  - problem -> PROBLEM shown to generator
  - idea    -> TARGET_IDEA given to judge
We return: {custom_id, problem, target_idea, best_idea, best_score, iters, error?}
"""

import os, sys, json, argparse, time
from typing import List, Tuple, Optional
from openai import OpenAI
import ollama

# -------------------- Config (defaults; override via CLI) --------------------
GEN_MODEL   = os.getenv("GEN_MODEL", "llama3.2")   # generator (no target exposure)
SCORE_MODEL = os.getenv("SCORE_MODEL", "o4-mini") # list-wise judge (sees target)
K           = int(os.getenv("K", "5"))
MAX_ITERS   = int(os.getenv("MAX_ITERS", "5"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "1.0"))

# Universal baseline (fixed similarity anchor)
BASELINE_IDEA  = "Placeholder Idea. [!Important instruction] For this first round, do not confidently set high scores for any ideas."
BASELINE_SCORE = 0.0

# Client
def _client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise SystemExit("Please set OPENAI_API_KEY")
    return OpenAI(api_key=key)

client = _client()

# -------------------- LLM helpers (Responses→Chat fallback) ------------------
def _supports_reasoning(model: str) -> bool:
    _REASONING_MODELS = {"o4-mini", "gpt-5", "o3", "o4-mini-high"}
    return any(model.lower().startswith(r) for r in _REASONING_MODELS)

def _resp_text(messages: list, model: str, temperature: Optional[float] = None, try_reasoning: bool = True) -> str:
    if model.startswith("llama") or ":" in model:
        r = ollama.chat(
            model=model,
            messages=messages,
            options={"temperature": float(temperature) if temperature else 1.0}
        )
        return r["message"]["content"]

    from openai import OpenAI
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise SystemExit("Please set OPENAI_API_KEY for OpenAI models")
    client = OpenAI(api_key=key)

    if try_reasoning and _supports_reasoning(model):
        try:
            r = client.responses.create(model=model, input=messages, reasoning={"effort": "medium"})
            return r.output_text
        except Exception:
            pass

    # fallback → chat.completions
    chat_temp = 1.0 if temperature is None else float(temperature)
    r = client.chat.completions.create(model=model, temperature=chat_temp, messages=messages)
    return r.choices[0].message.content or ""

# -------------------- Generation & Scoring --------------------
GEN_PROMPT = ""

def gen_candidates(model: str, problem: str, current: str, score: float, k: int = 5, temperature: float = 1.0) -> List[str]:
    sys_msg = "You propose concise, technically coherent research ideas for the given problem. Return ONLY JSON."
    user = (
        (GEN_PROMPT or "").strip()
        + "\nProblem:\n" + problem
        + f"\nCurrent similarity towards ideal answer (>0.95 is marked close; otherwise please think of something creative):\n{score}"
        + "\n\nCurrent idea:\n" + current
        + f"\n\nTask: Propose {k} alternative ideas that could better solve the problem. "
          "Vary mechanisms (data, model, training signal, retrieval structure, evaluation) and avoid trivial rewording. "
          f"Output ONLY a JSON list of {k} strings."
    )
    txt = _resp_text(
        messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": user}],
        model=model,
        temperature=temperature,
        try_reasoning=True,
    ).strip()

    # Parse JSON robustly
    try:
        data = json.loads(txt)
    except json.JSONDecodeError:
        i, j = txt.find("["), txt.rfind("]")
        data = json.loads(txt[i:j+1]) if i != -1 and j != -1 else []

    if isinstance(data, dict):
        for key in ("candidates", "list", "ideas", "sentences"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
        else:
            data = next((v for v in data.values() if isinstance(v, list)), [])

    out = [str(s).strip() for s in data if isinstance(s, (str, int, float))]
    return [str(s) for s in out][:k]

def score_listwise(model: str, target: str, sentences: List[str], fixed_first_score: float | None = None) -> List[float]:
    if not sentences:
        return []
    # Enforce baseline/current fixed score if supplied
    enforced_first = None
    if sentences[0].strip() == BASELINE_IDEA.strip():
        enforced_first = float(BASELINE_SCORE)
    elif fixed_first_score is not None:
        enforced_first = float(fixed_first_score)

    # If only one sentence and it's enforced
    if len(sentences) == 1 and enforced_first is not None:
        return [enforced_first]

    rest = sentences[1:] if enforced_first is not None else sentences[:]
    if not rest:
        return [enforced_first] if enforced_first is not None else [0.0]

    sys_msg = "You think and then return only valid JSON (array of numbers). No extra text."
    s0_clause = ""
    if enforced_first is not None:
        s0_clause = ( f"\nComparison basis (DO NOT re-score it): There exists a CURRENT idea with a FIXED similarity S0 = {enforced_first:.3f}." 
                       "\nUse S0 purely as a calibration anchor while scoring the following candidates against T:" 
                       "\n- If a candidate is clearly weaker or more generic relative to T than a typical baseline, score it <= S0." 
                       "\n- If a candidate introduces mechanisms/assumptions that move it closer in meaning to T, score it > S0." 
                       "\nYou must still output absolute similarities in [0,1] to T (not margins), but ensure they are calibrated with respect to S0." )
    instr = (
        "You are a careful, consistent idea similarity judge.\n"
        "Given a hidden target idea T and a list of candidate ideas, score EACH idea on a continuous 0-1 scale,"
        "\nwhere 0 = completely unrelated and 1 = semantically equivalent to T. Use semantics (ignore wording/style)."
        f"{s0_clause}\nReturn ONLY a compact JSON list of numbers in input order."
    )
    user = instr + "\n\nT (target idea):\n" + target + "\n\nIdeas to score:\n" + json.dumps(rest, ensure_ascii=False)

    txt = _resp_text(
        messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": user}],
        model=model,
        temperature=None,
        try_reasoning=True,
    ).strip()

    try:
        scores_rest = json.loads(txt)
    except json.JSONDecodeError:
        i, j = txt.find("["), txt.rfind("]")
        scores_rest = json.loads(txt[i:j+1]) if i != -1 and j != -1 else []

    cleaned: List[float] = []
    for v in scores_rest:
        try:
            f = float(v)
        except Exception:
            f = 0.0
        cleaned.append(0.0 if f < 0 else (1.0 if f > 1 else f))

    if len(cleaned) != len(rest):
        cleaned = (cleaned + [0.0] * len(rest))[:len(rest)]

    return ([enforced_first] + cleaned) if enforced_first is not None else cleaned

def hill_climb(problem: str, target: str, start: str, gen_model: str, score_model: str,
               k: int = 5, max_iters: int = 10, temperature: float = 1.0) -> Tuple[str, float, int]:
    current = start
    current_score = float(BASELINE_SCORE) if current.strip() == BASELINE_IDEA.strip() else score_listwise(score_model, target, [current])[0]
    for _ in range(1, max_iters + 1):
        cands = gen_candidates(gen_model, problem, current, current_score, k=k, temperature=temperature)
        pool = [current] + cands
        scores = score_listwise(score_model, target, pool, fixed_first_score=current_score)
        best_idx = max(range(len(pool)), key=lambda i: scores[i])
        best_sent, best_score = pool[best_idx], scores[best_idx]
        if best_score > current_score + 1e-6:
            current, current_score = best_sent, best_score
    return current, current_score, max_iters

# -------------------- I/O --------------------
def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except Exception:
                    continue

def append_jsonl(path: str, obj: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser(description="Batch hill-climbing over problem/idea JSONL.")
    ap.add_argument("--in", dest="infile", required=True, help="Input JSONL with {custom_id, problem, idea}")
    ap.add_argument("--out", dest="outfile", required=True, help="Output JSONL path")
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--iters", type=int, default=MAX_ITERS)
    ap.add_argument("--gen-model", default=GEN_MODEL)
    ap.add_argument("--score-model", default=SCORE_MODEL)
    ap.add_argument("--temp", type=float, default=TEMPERATURE)
    args = ap.parse_args()

    # create/clear out
    open(args.outfile, "w", encoding="utf-8").close()

    n, ok = 0, 0
    for row in read_jsonl(args.infile):
        n += 1
        cid = row.get("custom_id")
        problem = (row.get("problem") or "").strip()
        target = (row.get("idea") or "").strip()

        if not (problem and target):
            append_jsonl(args.outfile, {"custom_id": cid, "error": "missing problem or idea"})
            continue

        try:
            best_idea, best_score, steps = hill_climb(
                problem=problem,
                target=target,
                start=BASELINE_IDEA,
                gen_model=args.gen_model,
                score_model=args.score_model,
                k=args.k,
                max_iters=args.iters,
                temperature=args.temp,
            )
            append_jsonl(args.outfile, {
                "custom_id": cid,
                "problem": problem,
                "target_idea": target,
                "best_idea": best_idea,
                "best_score": round(float(best_score), 3),
                "iters": steps,
                "gen_model": args.gen_model,
                "score_model": args.score_model,
            })
            ok += 1
        except Exception as e:
            append_jsonl(args.outfile, {
                "custom_id": cid,
                "problem": problem,
                "target_idea": target,
                "error": f"{type(e).__name__}: {e}",
            })

    print(f"[DONE] processed={n} ok={ok} wrote={args.outfile}")

if __name__ == "__main__":
    main()
