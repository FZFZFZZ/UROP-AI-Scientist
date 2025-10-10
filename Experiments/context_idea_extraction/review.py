import os
import json
import time
import re
from typing import Tuple
from helper import get_response

MAX_ATTEMPTS = 10
SLEEP_BETWEEN_ATTEMPTS = 0.1

MODEL = "gpt-5"  # choose from gpt-4o-mini, gpt-4o, gpt-5

IN_PATH  = "../../Data/context_idea_extraction/data.jsonl"
OUT_DIR  = "../../Data/context_idea_extraction"
OUT_PATH = os.path.join(OUT_DIR, f"review_{re.sub(r'[^A-Za-z0-9_.-]+', '_', MODEL)}.jsonl")
TEMPERATURE = 0

# --- System & Tasks -----------------------------------------------------------

SYS = "You are a highly-experienced expert in NLP and AI. Do the following task. Output strictly in JSON format only."

TASK = """
You are an NLP expert.

Summarise the given abstract into two parts — “Context” and “Idea”.
Requirements:
1. The “Context” must describe only the background or problem setting.
2. The “Idea” must describe only the proposed method or innovation in around two or three sentences — EXCLUDING experiments, results, or impacts.
3. The “Context” must not reveal or hint at the idea.
4. Only output the following JSON Format:
{
  "Context": "...",
  "Idea": "..."
}

Example:
[Abstract]
The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train.

[Output]
{
  "Context": "Previous sequence transduction models—used for tasks like machine translation—mainly relied on recurrent or convolutional neural networks. These models typically consisted of an encoder and a decoder connected through an attention mechanism. However, such architectures were computationally expensive, difficult to parallelize, and inefficient to train on large datasets.",
  "Idea": "Introduce a new, simpler architecture that relies entirely on attention mechanisms, eliminating recurrence and convolution while supporting efficient parallel training."
}

Actual input:
[Abstract] 
{abstract}

[output]
"""

# --- NEW: Review/Refine pass --------------------------------------------------

REVIEW_SYS = (
    "You are a meticulous NLP editor. You strictly validate content and formatting, "
    "and you only output valid JSON with exactly the keys 'Context' and 'Idea'."
)

REVIEW_TASK = """
You will receive:
1) The original abstract.
2) A draft JSON with "Context" and "Idea".

Your job is to REVIEW and REFINE the draft to improve:
- **Formatting**: Output MUST be a single JSON object with exactly "Context" and "Idea" (both strings). No code fences, no commentary.
- **Consistency**: 
  - "Context" = only background/problem setup; DO NOT leak or hint at the method.
  - "Idea" = only the proposed method/innovation in ~2–3 sentences; EXCLUDE experiments, results, impacts, performance claims, or datasets.
  - Keep terminology consistent with the abstract; avoid adding facts not present in the abstract.
- **Readability & Signal-to-Noise**:
  - Remove redundancy, hedging, filler phrases.
  - Prefer precise, specific phrasing over generic claims.
  - Keep both fields concise but informative.

If the draft violates constraints, FIX it. If content is missing, reconstruct minimally from the abstract without inventing unsupported details.

Return only:
{
  "Context": "...",
  "Idea": "..."
}
"""

# -----------------------------------------------------------------------------

def make_user_prompt(abstract: str) -> str:
    """
    Fill the abstract into TASK. We only replace the {abstract} placeholder
    to avoid interfering with other braces in the prompt.
    """
    if "{abstract}" in TASK:
        return TASK.replace("{abstract}", abstract)
    return f"{TASK.strip()}\n\n[Abstract] {abstract}\n"

def strict_json_extract(text: str) -> dict:
    """
    Extract and parse a single JSON object from model text output.
    - Finds the first '{' to the last '}' and tries json.loads
    - Validates keys "Context" and "Idea" exist and are strings
    Raises ValueError if invalid.
    """
    start = text.find("{")
    end   = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in output.")
    candidate = text[start:end+1].strip()
    obj = json.loads(candidate)

    if not isinstance(obj, dict):
        raise ValueError("Parsed JSON is not an object.")
    if "Context" not in obj or "Idea" not in obj:
        raise ValueError('JSON must contain keys "Context" and "Idea".')
    if not isinstance(obj["Context"], str) or not isinstance(obj["Idea"], str):
        raise ValueError('"Context" and "Idea" must be strings.')
    return obj

def enforce_json_only_prompt(base_user_prompt: str) -> str:
    """Append a strict instruction to output only JSON with the two keys."""
    suffix = (
        "\n\nIMPORTANT: Output ONLY a single JSON object with exactly two fields:\n"
        '{ "Context": "...", "Idea": "..." }\n'
        "No extra text, no code fences, no comments."
    )
    return base_user_prompt + suffix

def get_valid_response(system_prompt: str, user_prompt: str, model: str, temperature: float) -> dict:
    """
    Calls the LLM up to MAX_ATTEMPTS times until a valid JSON is returned.
    Returns the parsed dict.
    """
    last_err = None
    prompt = enforce_json_only_prompt(user_prompt)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            raw = get_response(model, system_prompt, prompt, temperature=temperature)
            obj = strict_json_extract(raw)
            return obj
        except Exception as e:
            last_err = e
            prompt = (
                enforce_json_only_prompt(user_prompt)
                + "\nIf your previous output was not valid JSON, regenerate strictly as specified."
            )
            time.sleep(SLEEP_BETWEEN_ATTEMPTS)
    raise RuntimeError(f"Failed to obtain valid JSON after {MAX_ATTEMPTS} attempts: {last_err}")

# --- NEW: second-pass reviewer -----------------------------------------------

def make_review_prompt(abstract: str, draft: dict) -> str:
    """Builds the review/refine user prompt with abstract + draft JSON."""
    # Keep the draft compact to avoid encouraging commentary
    draft_json = json.dumps(
        {"Context": draft.get("Context", ""), "Idea": draft.get("Idea", "")},
        ensure_ascii=False,
    )
    return (
        f"{REVIEW_TASK}\n\n"
        f"[Abstract]\n{abstract.strip()}\n\n"
        f"[Draft]\n{draft_json}\n\n"
        "[Output]"
    )

def review_and_refine(abstract: str, draft_obj: dict, model: str, temperature: float = 0) -> dict:
    """
    Second-pass LLM call to review and refine the draft strictly into JSON.
    """
    user_prompt = make_review_prompt(abstract, draft_obj)
    return get_valid_response(REVIEW_SYS, user_prompt, model, temperature)

# --- Optional local heuristic guard on "Idea" --------------------------------

RESULTY_WORDS = re.compile(
    r"\b(accuracy|accuracies|f1|f-?score|auc|auroc|bleu|rouge|improv(e|es|ed|ement)|"
    r"outperform|state[- ]of[- ]the[- ]art|sota|experiment(s|al)|result(s)?|"
    r"empirical|benchmark(s)?|dataset(s)?|evaluation(s)?)\b",
    flags=re.IGNORECASE
)

def strip_resulty_language(text: str) -> str:
    """
    Soft cleanup that neutralizes common results/impact phrasing if it slipped in.
    """
    # Minimal neutralizations without inventing content
    text = re.sub(r"\b(outperforms?|state[- ]of[- ]the[- ]art|SOTA)\b", "the proposed approach", text, flags=re.IGNORECASE)
    return text

# --- Main --------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    total, written = 0, 0

    if not os.path.exists(IN_PATH):
        raise FileNotFoundError(f"Input file not found: {IN_PATH}")

    with open(IN_PATH, "r", encoding="utf-8") as fin, \
         open(OUT_PATH, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1

            rec = None
            try:
                rec = json.loads(line)
                pid = rec.get("id")
                abstract = rec.get("abstract", "")

                if not pid or not isinstance(abstract, str) or not abstract.strip():
                    raise ValueError("Invalid record: missing id or abstract.")

                # --- Pass 1: draft extraction
                user_prompt = make_user_prompt(abstract)
                draft = get_valid_response(SYS, user_prompt, MODEL, TEMPERATURE)

                # --- Optional light guard before review
                if RESULTY_WORDS.search(draft["Idea"]):
                    draft["Idea"] = strip_resulty_language(draft["Idea"])

                # --- Pass 2: review & refine
                refined = review_and_refine(abstract, draft, MODEL, TEMPERATURE)

                # --- Final guard (never invent; only strip if needed)
                if RESULTY_WORDS.search(refined["Idea"]):
                    refined["Idea"] = strip_resulty_language(refined["Idea"])

                out_record = {
                    "id": pid,
                    "Context": refined["Context"].strip(),
                    "Idea": refined["Idea"].strip(),
                }
                fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                written += 1

                # --- small progress log every 10 successes
                if written % 10 == 0:
                    print(f"{written}/100", flush=True)

            except Exception as e:
                rec_id = rec.get("id") if isinstance(rec, dict) else None
                print(f"⚠️ Error on id={rec_id}: {type(e).__name__}: {e}", flush=True)

    print(f"Done. Processed: {total}, wrote: {written}, output: {OUT_PATH}")


if __name__ == "__main__":
    main()
