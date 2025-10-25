# run under ./Experiments/MDP
from MDP.signaling.baseline import point_wise_sim
import json, time, os
from typing import List, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from MDP.helper import get_response
from openai import OpenAI

import numpy as np

_client = OpenAI()
EMBED_MODEL = "text-embedding-3-large"

MODEL = "gpt-4-turbo-2024-04-09"
TABOO_MODEL = "gpt-4.1"

def _embed(text: str) -> List[float]:
    resp = _client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return resp.data[0].embedding

def _cos(a: List[float], b: List[float]) -> float:
    a = np.array(a)
    b = np.array(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / (denom + 1e-9))

# -----------------------------
# LLM helpers
# -----------------------------

def _llm(prompt: str, *, temperature: float = 0.6, max_retries: int = 2) -> str:
    """Thin wrapper around get_response with basic retries."""
    for attempt in range(max_retries + 1):
        try:
            return get_response(
                model=MODEL,
                system_prompt="Return only the content requested, no extra commentary. Role: You are an insightful NLP and AI researcher who can always propose good ideas. Good ideas favor more on fidelity than semantic novelty.",
                user_prompt=prompt,
                temperature=temperature,
                priority=False
            )
        except Exception:
            if attempt == max_retries:
                raise
            time.sleep(0.6 + 0.3 * attempt)
    return ""

def _parse_list_payload(payload: str, n: int = 5) -> List[str]:
    """
    Parse a list of candidate ideas from model output.
    Accepts JSON array or falls back to numbered lines.
    """
    payload = payload.strip()

    # JSON array
    try:
        data = json.loads(payload)
        if isinstance(data, list) and all(isinstance(x, str) for x in data):
            out = [s.strip() for s in data if s and s.strip()]
            return out[:n]
        if isinstance(data, dict):
            for k in ("ideas", "candidates", "variants"):
                if k in data and isinstance(data[k], list):
                    out = [str(s).strip() for s in data[k] if str(s).strip()]
                    return out[:n]
    except Exception:
        pass

    # Fallback: numbered/newline list
    lines = [ln.strip() for ln in payload.splitlines() if ln.strip()]
    cleaned: List[str] = []
    for ln in lines:
        ln = ln.lstrip("-•").strip()
        i = 0
        while i < len(ln) and (ln[i].isdigit() or ln[i] in ".):"):
            i += 1
        ln = ln[i:].strip()
        if ln:
            cleaned.append(ln)
        if len(cleaned) >= n:
            break
    return cleaned[:n]

# -----------------------------
# Candidate generation
# -----------------------------

GEN_PROMPT = """You will generate concise idea variants (one sentence each) that explore potentially better alignments than the current idea.
Distance score d (a float between 0 and 1) is point-wise scored directly by LLM, signaling current idea's distance to the actually best idea to address the given context given by god. If you think it is low, think about exploring something boldly different in paradigm. If it is high, consider exploit the idea at different angles or elaborate on that more.

Upon previous exploration, these keywords are taboos that lead to bad ideas. Do not use them:
{taboo_list}

And these good keywords may lead to that optimal idea. Try to explore your ideas based on them:
{good_list}

Requirements:
- Output EXACTLY a JSON array of {n} strings, no commentary. Example: ["...", "...", "...", "...", "...", ...]
- Each string must be around 2-3 sentences, self-contained, and specific.
- Make sure no repetition between the ideas generated.
- Use academic tone. Ideas must be ICLR-standard.

[Context]
{context}

[Current Idea | Distance d = {sim}]
{current}

Now output the JSON array of {n} candidate ideas only.
"""

def generate_candidates(current_idea: str, sim: float, target_idea: str, context: str, n: int, good_list: list[str], taboo_list: list[str]) -> List[str]:
    raw = _llm(GEN_PROMPT.format(context=context, current=current_idea, sim=sim, n=n, good_list=good_list, taboo_list=taboo_list), temperature=0.5)
    ideas = _parse_list_payload(raw, n=n)
    if len(ideas) < n:
        # simple padding to keep loop stable
        while len(ideas) < n:
            ideas.append(f"{current_idea} [variant {len(ideas)+1}]")
    return ideas[:n]

# -----------------------------
# Scoring (point_wise_sim returns float)
# -----------------------------

def score_idea(idea: str, target_idea: str, context: str) -> float:
    return float(point_wise_sim(idea=idea, context=context, target_idea=target_idea))

# -----------------------------
# Taboo climbing loop (no early stop)
# -----------------------------

def taboo_climb(
    initial_idea: str,
    target_idea: str,
    context: str,
    *,
    rounds: int = 50,
    fanout: int = 8,
    verbose: bool = True,
    pool: list[list[float]] | None = None,
) -> Dict[str, Any]:
    """
    Taboo-climbing with idea pool tracking AND taboo words tracking
    'pool' accumulates unique ideas for later embedding/similarity analysis.
    """

    def clear_dup(word_list: List[str]) -> List[str]:
        cleaned = []
        seen = set()

        for w in word_list:
            if not w or not isinstance(w, str):
                continue
            key = w.strip().lower()
            if key not in seen:
                cleaned.append(w)
                seen.add(key)
        return cleaned


    def clear_collision(lx1: List[str], lx2: List[str]) -> Tuple[List[str], List[str]]:
        s1 = {w.strip().lower() for w in lx1 if isinstance(w, str)}
        s2 = {w.strip().lower() for w in lx2 if isinstance(w, str)}
        inter = s1 & s2  # case-insensitive intersections

        lx1_clean = [w for w in lx1 if isinstance(w, str) and w.strip().lower() not in inter]
        lx2_clean = [w for w in lx2 if isinstance(w, str) and w.strip().lower() not in inter]

        return lx1_clean, lx2_clean

    # initialize pool if not provided
    if pool is None:
        pool = []

    def _norm(s: str) -> str:
        return " ".join(s.lower().split())

    seen = set(_norm(x) for x in pool)

    best_so_far = initial_idea.strip()
    best_score_so_far = score_idea(best_so_far, target_idea, context)

    trace: List[Dict[str, Any]] = [{
        "round": 0,
        "round_best": best_so_far,
        "round_best_score": best_score_so_far,
        "best_so_far": best_so_far,
        "best_score_so_far": best_score_so_far,
        "candidates": []
    }]

    nov = 1

    taboo_list = generate_taboo(best_so_far, target_idea, context)
    good_list = generate_good(best_so_far, target_idea, context)

    for r in range(1, rounds + 1):
        # 1) generate variants
        candidates = generate_candidates(best_so_far, best_score_so_far, target_idea, context, fanout, good_list, taboo_list)

        # 2) filter based on similarity
        filtered: List[str] = []
        filtered_embs: List[List[float]] = []

        for c in candidates:
            emb = _embed(_norm(c))
            max_sim = max((_cos(emb, p) for p in pool), default=0.0)
            if max_sim < 0.8:
                filtered.append(c)
                filtered_embs.append(emb)

        # if nothing novel — skip iteration entirely
        if not filtered:
            if verbose:
                print(f"[{r:02d}] skipped — no novel candidates (<0.8 sim)")
            continue

        # 3) score filtered candidates
        scored: List[Tuple[float, str]] = [
            (score_idea(c, target_idea, context), c) for c in filtered
        ]
        scored.sort(key=lambda t: t[0], reverse=True)
        round_best_score, round_best_idea = scored[0]

        # 4) update taboo and good list
        for c in filtered:
            new_taboo_words = generate_taboo(c, target_idea, context)
            new_good_words = generate_good(c, target_idea, context)
            taboo_list.extend(new_taboo_words)
            good_list.extend(new_good_words)
        taboo_list = clear_dup(taboo_list)
        good_list = clear_dup(good_list)
        taboo_list, good_list = clear_collision(taboo_list, good_list)

        print(f"Round {r}\nNay: ")
        print(taboo_list)
        print(f"Yay: ")
        print(good_list)

        # 5) update best
        if round_best_score > best_score_so_far:
            best_so_far, best_score_so_far = round_best_idea, round_best_score

        # 6) add idea into the pool if it is repetitive AND lowly-scored 
        for score_value, idea_text in scored:
            if score_value >= best_score_so_far * 0.9:
                try:
                    idx = filtered.index(idea_text)
                    filtered_embs.pop(idx)
                    filtered.pop(idx)
                    nov += 1
                except ValueError:
                    print("failed to preserve a potentially good candidate")
                    pass
        pool.extend(filtered_embs)

        # 6) trace + log
        if verbose:
            print("filtered: \n")
            print(filtered)
            print("\nbsf: \n")
            print(best_so_far)
            print(f"\n[{r:02d}] best={round_best_score:.6f} | global={best_score_so_far:.6f} | judged = {len(pool) + nov}")
            

        trace.append({
            "round": r,
            "round_best": round_best_idea,
            "round_best_score": round_best_score,
            "best_so_far": best_so_far,
            "best_score_so_far": best_score_so_far,
            "taboo_list": taboo_list,
            "taboo_length": len(taboo_list),
            "good_list": good_list,
            "good_length": len(good_list),
            "candidates": [{"idea": c, "score": s} for s, c in scored],
        })


    return {
        "best_idea_overall": best_so_far,
        "best_score_overall": best_score_so_far,
        "rounds_run": rounds,
        "trace": trace,
        "#": len(pool) + nov,
        "taboo_list": taboo_list,
        "good_list": good_list
    }

# ----------------------------
# One-shot caller (JSON object)
# ----------------------------
def _llm_json_object(system_prompt: str, user_prompt: str) -> dict:
    try:
        resp = get_response(
            model=MODEL,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.2,
            priority=False
        ).strip()
        data = json.loads(resp)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}

# ----------------------------
# Combined prompts (one round)
# ----------------------------
COMBINED_SYSTEM = """You are an expert writing coach for LLM learners.

Your task: from the learner's CURRENT_IDEA, extract up to TWO "taboo" terms that MISLEAD the learner away from the TARGET_IDEA, and up to TWO "good" terms that HELP the learner move toward the TARGET_IDEA, given the CONTEXT.

Think first. Decision Rules:
0) If you are unsure, or there are no valid items for a category, return an empty array for that category.
1) Every returned term MUST appear verbatim in CURRENT_IDEA (contiguous phrase).
2) Taboo terms tend to be: vague buzzwords, novelty-for-novelty, scope creep, or items lacking direct mechanistic connection to the TARGET_IDEA.
3) Good terms tend to be: concrete mechanisms, actionable links, or tractable refinements that directly narrow the gap toward the TARGET_IDEA.
4) Enforce DISJOINTNESS: no term may appear in BOTH categories. If a candidate fits both, prefer placing it in "good" and exclude it from "taboo".
5) Output ONLY a single JSON object with this exact shape (no explanations):
{
  "taboo": ["...", "..."],
  "good": ["...", "..."]
}
"""

COMBINED_USER_TMPL = """CURRENT_IDEA:
{idea}

TARGET_IDEA:
{target_idea}

CONTEXT:
{context}

Return ONLY the JSON object with keys "taboo" and "good" as specified.
"""

# ----------------------------
# Post-processing helpers
# ----------------------------
def _dedup_case_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for w in items:
        if isinstance(w, str):
            k = w.strip().lower()
            if k and k not in seen:
                seen.add(k)
                out.append(w.strip())
    return out

def _filter_in_idea(items: List[str], idea: str) -> List[str]:
    idea_l = idea.lower()
    return [w for w in items if w.strip().lower() in idea_l]

def _enforce_disjoint(taboo: List[str], good: List[str]) -> Tuple[List[str], List[str]]:
    # Good wins ties
    taboo_set = {w.lower() for w in taboo}
    good_set = {w.lower() for w in good}
    inter = taboo_set & good_set
    if inter:
        taboo = [w for w in taboo if w.lower() not in inter]
    return taboo, good

def _cap2(items: List[str]) -> List[str]:
    return items[:2]

# ----------------------------
# Public API: one-round generator
# ----------------------------
def generate_taboo_and_good(idea: str, target_idea: str, context: str) -> Tuple[List[str], List[str]]:
    user_prompt = COMBINED_USER_TMPL.format(idea=idea, target_idea=target_idea, context=context)
    data = _llm_json_object(COMBINED_SYSTEM, user_prompt)

    taboo = data.get("taboo", []) if isinstance(data.get("taboo", []), list) else []
    good  = data.get("good", [])  if isinstance(data.get("good", []), list)  else []

    # 1) normalize & dedup
    taboo = _dedup_case_preserve_order(taboo)
    good  = _dedup_case_preserve_order(good)

    # 2) must appear verbatim (case-insensitive substring) in idea
    taboo = _filter_in_idea(taboo, idea)
    good  = _filter_in_idea(good, idea)

    # 3) disjointness (good wins)
    taboo, good = _enforce_disjoint(taboo, good)

    # 4) cap at 2 each
    taboo = _cap2(taboo)
    good  = _cap2(good)

    return taboo, good

# ----------------------------
# Backward-compatible wrappers
# ----------------------------
def generate_taboo(idea: str, target_idea: str, context: str) -> List[str]:
    taboo, _ = generate_taboo_and_good(idea, target_idea, context)
    return taboo

def generate_good(idea: str, target_idea: str, context: str) -> List[str]:
    _, good = generate_taboo_and_good(idea, target_idea, context)
    return good


# ----------------------------
# Legendary start
# ----------------------------

def generate_start(context: str) -> str:
    """
    Generate a strong, novel starting idea (2–3 sentences) from a given research context.
    Uses the same LLM interface (get_response).
    """
    SYS = """
    You are a creative research scientist with deep understanding of NLP, AI, and computational methods.
    Your job: invent genuinely novel, technically interesting research ideas that fit within the given context.
    Always respond concisely (2–3 sentences) in plain English.
    Search in your knowledge base thoroughly to expand the relevant contexts. Then output only the idea text, no commentary or formatting.
    """

    user_prompt = f"""
Context:
{context}

Generate the best novel idea you can think of in this domain. 
Be creative but realistic, and describe it in 2–3 sentences.
"""

    idea = get_response(
        model=MODEL,
        system_prompt=SYS,
        user_prompt=user_prompt,
        temperature=1,
        priority=False
    )

    return idea.strip()

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    ctx = "A common method for developing agents with Language Models involves iteratively prompting the model, reflecting on its outputs, and updating the prompts until the task is completed. However, this approach faces challenges such as limited exploration of the decision space due to repetitive reflections and an inability to utilize insights from previously solved tasks."
    start = generate_start(ctx)
    target = "Introduce DoT (Diversity of Thoughts), a framework that reduces redundant reflections to improve decision-space exploration and incorporates a task-agnostic memory component for knowledge retrieval from previously solved tasks, enhancing the agent's performance across various reasoning tasks."
    result = taboo_climb(
        initial_idea=start,
        target_idea=target,
        context=ctx,
        rounds=40,
        fanout=8,
        verbose=True
    )
    print("\n=== FINAL ===")
    print("best :", result["best_score_overall"], "::", result["best_idea_overall"])
    print("Total unique ideas collected:", result["#"])
