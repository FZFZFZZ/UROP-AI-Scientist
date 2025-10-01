#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch hill-climbing / beam-search over a JSONL of {custom_id, problem, idea}.
Now also logs EVERY newly computed (idea, score, problem) pair to --pairs-out JSONL.

For each input line:
  - problem -> PROBLEM shown to generator
  - idea    -> TARGET_IDEA given to judge

Outputs (per-row) to --out and logs (per-candidate) to --pairs-out.
"""

import os, sys, json, argparse, time, datetime
from typing import List, Tuple, Optional
from openai import OpenAI
import ollama

# -------------------- Config (defaults; override via CLI) --------------------
GEN_MODEL   = os.getenv("GEN_MODEL", "llama3.2")   # generator (no target exposure)
SCORE_MODEL = os.getenv("SCORE_MODEL", "o4-mini")  # list-wise judge (sees target)
K           = int(os.getenv("K", "3"))             # per-beam fanout (k=3)
MAX_ITERS   = int(os.getenv("MAX_ITERS", "3"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "1.0"))
BEAMS       = int(os.getenv("BEAMS", "3"))         # beam width (3)

# Universal baseline (fixed similarity anchor)
BASELINE_IDEA  = "Placeholder Idea. [!Important instruction] For this first round, do not confidently set high scores for any ideas."
BASELINE_SCORE = 0.0

# -------------------- Client --------------------
def _client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise SystemExit("Please set OPENAI_API_KEY")
    return OpenAI(api_key=key)

client = _client()

# -------------------- Utility I/O --------------------
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

def iso_now() -> str:
    return datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).isoformat()

# -------------------- Pair logging --------------------
def log_pairs(pairs_out: Optional[str],
              rows: List[dict]):
    """Append candidate scoring rows to pairs_out if provided."""
    if not pairs_out or not rows:
        return
    with open(pairs_out, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# -------------------- LLM helpers (Responses→Chat fallback) ------------------
def _supports_reasoning(model: str) -> bool:
    _REASONING_MODELS = {"o4-mini", "gpt-5", "o3", "o4-mini-high"}
    return any(model.lower().startswith(r) for r in _REASONING_MODELS)

def _resp_text(messages: list, model: str, temperature: Optional[float] = None, try_reasoning: bool = True) -> str:
    if model.startswith("llama") or ":" in model:
        r = ollama.chat(
            model=model,
            messages=messages,
            options={"temperature": float(temperature) if temperature is not None else 1.0}
        )
        return r["message"]["content"]

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

    chat_temp = 1.0 if temperature is None else float(temperature)
    r = client.chat.completions.create(model=model, temperature=chat_temp, messages=messages)
    return (r.choices[0].message.content or "").strip()

# -------------------- Generation & Scoring --------------------
GEN_PROMPT = ""

def gen_candidates(model: str, problem: str, current: str, score: float, k: int = 3, temperature: float = 1.0) -> List[str]:
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
    data: List[str] = []
    try:
        parsed = json.loads(txt)
    except json.JSONDecodeError:
        i, j = txt.find("["), txt.rfind("]")
        parsed = json.loads(txt[i:j+1]) if i != -1 and j != -1 else []

    if isinstance(parsed, dict):
        for key in ("candidates", "list", "ideas", "sentences"):
            if key in parsed and isinstance(parsed[key], list):
                parsed = parsed[key]
                break
        else:
            parsed = next((v for v in parsed.values() if isinstance(v, list)), [])

    if isinstance(parsed, list):
        for s in parsed:
            if isinstance(s, (str, int, float)):
                data.append(str(s).strip())

    return [s for s in data if s][:k]

def score_listwise_multi(model: str,
                         target: str,
                         sentences: List[str],
                         fixed_scores: List[float]) -> List[float]:
    """
    Score a batch of ideas against target with multiple anchors.
    fixed_scores: list of known scores for the first len(fixed_scores) sentences.
    Returns scores aligned with sentences order.
    """
    anchors_clause = ""
    for i, s in enumerate(fixed_scores):
        anchors_clause += (
            f"\nAnchor idea A{i} has a FIXED similarity S{i} = {s:.3f}. "
            "Do not rescore anchors; use them only for calibration."
        )
    instr = (
        "You are a careful idea similarity judge.\n"
        "Given hidden target idea T and candidate ideas, assign a similarity in [0,1]."
        + anchors_clause +
        "\nReturn ONLY a JSON array of numbers, aligned with the input list order."
    )
    sys_msg = "You must output only valid JSON, nothing else."
    user = instr + "\n\nT:\n" + target + "\n\nIdeas:\n" + json.dumps(sentences, ensure_ascii=False)

    txt = _resp_text(
        messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": user}],
        model=model,
        temperature=None,
        try_reasoning=True,
    ).strip()

    try:
        scores = json.loads(txt)
    except Exception:
        i, j = txt.find("["), txt.rfind("]")
        scores = json.loads(txt[i:j+1]) if i != -1 and j != -1 else []

    # enforce anchors
    for i, s in enumerate(fixed_scores):
        scores[i] = float(s)

    return [max(0, min(1, float(x))) for x in scores]

# -------------------- Search algorithms --------------------
def beam_search(problem: str, target: str,
                width: int,
                k: int,
                gen_model: str,
                score_model: str,
                max_iters: int,
                temperature: float,
                custom_id: Optional[str],
                pairs_out: Optional[str]) -> Tuple[List[Tuple[str, float]], int]:
    """
    Returns: (final_beams, iters)
      final_beams: list of (idea, score) sorted desc by score, length=width
    Also logs every new candidate's (idea, score, problem) to pairs_out.
    """
    beams: List[Tuple[str, float]] = [(BASELINE_IDEA, BASELINE_SCORE) for _ in range(width)]
    tstamp = iso_now()

    for it in range(1, max_iters + 1):
        all_pool = [idea for idea, _ in beams]
        fixed_scores = [score for _, score in beams]
        cand_records = []  # rows to log

        # generate candidates per beam
        origin_index: List[int] = []  # parallel to appended candidates
        for b_idx, (idea_i, score_i) in enumerate(beams):
            cands = gen_candidates(gen_model, problem, idea_i, score_i, k=k, temperature=temperature)
            for c in cands:
                all_pool.append(c)
                origin_index.append(b_idx)

        # one scoring pass
        scores = score_listwise_multi(score_model, target, all_pool, fixed_scores)
        # anchors first, then candidates
        anchor_scores = scores[:len(beams)]
        cand_scores   = scores[len(beams):]

        # Log only candidates (newly scored items)
        for c, sc, ob in zip(all_pool[len(beams):], cand_scores, origin_index):
            cand_records.append({
                "timestamp": tstamp,
                "custom_id": custom_id,
                "problem": problem,
                "target_idea": target,
                "idea": c,
                "score": round(float(sc), 6),
                "iter": it,
                "mode": "candidate",
                "origin_beam": ob,
            })

        log_pairs(pairs_out, cand_records)

        # rebuild candidate list with scores (anchors + candidates)
        all_candidates = list(zip(all_pool, scores))
        # sort + dedup (keep best)
        all_candidates.sort(key=lambda t: t[1], reverse=True)
        dedup, seen = [], set()
        for idea, sc in all_candidates:
            key = idea.strip()
            if key not in seen:
                seen.add(key)
                dedup.append((idea, sc))
                if len(dedup) >= width:
                    break
        beams = dedup

    beams.sort(key=lambda t: t[1], reverse=True)
    return beams, max_iters

def hill_climb(problem: str, target: str,
               start: str,
               gen_model: str,
               score_model: str,
               k: int,
               max_iters: int,
               temperature: float,
               custom_id: Optional[str],
               pairs_out: Optional[str]) -> Tuple[str, float, int]:
    """
    Simple hill-climbing: keep current best; at each iter, sample k, score (current + k),
    move if any candidate beats current. Log all candidates each iter.
    """
    current, current_score = start, BASELINE_SCORE
    tstamp = iso_now()

    for it in range(1, max_iters + 1):
        cands = gen_candidates(gen_model, problem, current, current_score, k=k, temperature=temperature)
        pool = [current] + cands
        scores = score_listwise_multi(score_model, target, pool, fixed_scores=[current_score])
        cand_scores = scores[1:]

        # log candidates
        rows = []
        for c, sc in zip(cands, cand_scores):
            rows.append({
                "timestamp": tstamp,
                "custom_id": custom_id,
                "problem": problem,
                "target_idea": target,
                "idea": c,
                "score": round(float(sc), 6),
                "iter": it,
                "mode": "candidate",
                "origin_beam": 0,
            })
        log_pairs(pairs_out, rows)

        # choose best
        best_idx = max(range(len(cands)), key=lambda i: cand_scores[i], default=None)
        if best_idx is None:
            break
        best_cand, best_score = cands[best_idx], cand_scores[best_idx]

        if best_score > current_score:
            current, current_score = best_cand, best_score
        else:
            break

    return current, float(current_score), it

# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser(description="Batch hill-climbing / beam-search over problem/idea JSONL.")
    ap.add_argument("--in", dest="infile", required=True, help="Input JSONL with {custom_id, problem, idea}")
    ap.add_argument("--out", dest="outfile", required=True, help="Output JSONL path")
    ap.add_argument("--pairs-out", dest="pairs_out", default=None, help="Optional JSONL to append (problem, idea, score) pairs as they are scored")
    ap.add_argument("--beams", type=int, default=BEAMS, help="Beam width (1 = hill climb)")
    ap.add_argument("--k", type=int, default=K, help="Per-beam fanout (new ideas per beam per iter)")
    ap.add_argument("--iters", type=int, default=MAX_ITERS)
    ap.add_argument("--gen-model", default=GEN_MODEL)
    ap.add_argument("--score-model", default=SCORE_MODEL)
    ap.add_argument("--temp", type=float, default=TEMPERATURE)
    args = ap.parse_args()

    # create/clear main out
    open(args.outfile, "w", encoding="utf-8").close()
    # do NOT clear pairs_out; we append over the whole batch so you can aggregate across inputs

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
            if args.beams and args.beams > 1:
                final_beams, steps = beam_search(
                    problem=problem,
                    target=target,
                    width=int(args.beams),
                    k=int(args.k),
                    gen_model=args.gen_model,
                    score_model=args.score_model,
                    max_iters=int(args.iters),
                    temperature=float(args.temp),
                    custom_id=cid,
                    pairs_out=args.pairs_out,
                )
                best_idea, best_score = final_beams[0]
                out = {
                    "custom_id": cid,
                    "problem": problem,
                    "target_idea": target,
                    "best_idea": best_idea,
                    "best_score": round(float(best_score), 3),
                    "iters": steps,
                    "beams": [{"idea": i, "score": round(float(s), 3)} for i, s in final_beams],
                    "gen_model": args.gen_model,
                    "score_model": args.score_model,
                }
            else:
                best_idea, best_score, steps = hill_climb(
                    problem=problem,
                    target=target,
                    start=BASELINE_IDEA,
                    gen_model=args.gen_model,
                    score_model=args.score_model,
                    k=int(args.k),
                    max_iters=int(args.iters),
                    temperature=float(args.temp),
                    custom_id=cid,
                    pairs_out=args.pairs_out,
                )
                out = {
                    "custom_id": cid,
                    "problem": problem,
                    "target_idea": target,
                    "best_idea": best_idea,
                    "best_score": round(float(best_score), 3),
                    "iters": steps,
                    "gen_model": args.gen_model,
                    "score_model": args.score_model,
                }

            append_jsonl(args.outfile, out)
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

# python run.py --in test.jsonl --out results.jsonl --beams 3 --pairs-out pairs.jsonl
