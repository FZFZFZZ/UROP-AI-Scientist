from helper import get_response
from typing import Any, Dict
import re
import json

SYS = """
You are an expert scientific-idea equivalence judge. 
Task: compare a TARGET (abstract + concise idea) to a CANDIDATE idea and output a single JSON object ONLY, with no extra text.

Rules:
- Use ONLY the provided text. Do NOT use outside knowledge.
- Quote evidence as exact substrings and give 0-based [start, end) character offsets.
  • For TARGET quotes, offsets refer to the string: target_abstract + "\n" + target_idea
  • For CANDIDATE quotes, offsets refer to the string: candidate_idea
  • Max 3 quotes per side, each quote ≤ 40 characters.
  • If no suitable quote exists, use an empty list for that side.
- Verdict rubric:
  • SAME: Core mechanism AND objective match; problem/setting close; differences are superficial.
  • NEARLY_SAME: Mechanism & objective match, but scope/assumptions materially differ.
  • RELATED_BUT_DISTINCT: Problem/setting similar, but mechanism OR objective differs.
  • DIFFERENT: None of the above or insufficient overlap.
- Confidence ∈ [0,1], reflecting strength/coverage of matches and evidence quality.
- Be concise in "why" (≤ 60 words). No markdown, no commentary, no trailing commas.
- Output must validate as strict JSON with keys: why, evidence{target[],candidate[]}, verdict, confidence.
"""

def make_user_prompt(target_abstract: str, target_idea: str, mimic_idea: str) -> str:
    return (
        "TARGET_ABSTRACT:\n" + target_abstract.strip() + "\n\n"
        "TARGET_IDEA:\n" + target_idea.strip() + "\n\n"
        "CANDIDATE_IDEA:\n" + mimic_idea.strip() + "\n\n"
        'Return exactly this JSON:\n'
        '{\n'
        '  "why": "short natural-language rationale",\n'
        '  "evidence": {\n'
        '    "target": [{"span": [start, end], "quote": "..."}],\n'
        '    "candidate":  [{"span": [start, end], "quote": "..."}]\n'
        '  },\n'
        '  "verdict": "SAME | NEARLY_SAME | RELATED_BUT_DISTINCT | DIFFERENT",\n'
        '  "confidence": 0.0\n'
        '}\n\n'
        "Where:\n"
        "- For TARGET spans, compute offsets on the concatenated string:\n"
        "  target_full = TARGET_ABSTRACT + \"\\n\" + TARGET_IDEA\n"
        "- For CANDIDATE spans, compute offsets on the string:\n"
        "  candidate_full = CANDIDATE_IDEA\n"
        "- Use at most 3 quotes per side; quotes ≤ 40 chars each.\n"
        "- If a field lacks supportable text, use an empty list for that side.\n"
        "- Choose ONE verdict per rubric and a confidence in [0,1].\n"
    )

def _coerce_json(s: str) -> Dict[str, Any]:
    """
    Be forgiving to minor formatting issues:
    - strip code fences
    - grab the first {...} block
    - load as JSON
    """
    if not s:
        raise ValueError("Empty model response.")
    # remove code fences if any
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s.strip(), flags=re.IGNORECASE)
    # extract first JSON object
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON object found in response.")
    block = m.group(0)
    # remove trailing commas (common LLM mistake)
    block = re.sub(r",\s*([\]}])", r"\1", block)
    return json.loads(block)

def judge_match(
    target_abstract: str,
    target_idea: str,
    mimic_idea: str,
    *,
    model: str = "gpt-5",
    temperature: float = 0.0
) -> Dict[str, Any]:
    """
    Call the LLM judge and return a strict dict:
    {
      "why": str,
      "evidence": {"target": [...], "mimic": [...]},
      "verdict": "SAME|NEARLY_SAME|RELATED_BUT_DISTINCT|DIFFERENT",
      "confidence": float
    }
    """
    user_prompt = make_user_prompt(target_abstract, target_idea, mimic_idea)
    raw = get_response(model=model, system_prompt=SYS, user_prompt=user_prompt, temperature=temperature, priority = True)
    out = _coerce_json(raw)

    # Minimal schema guard-rails
    out.setdefault("why", "")
    out.setdefault("evidence", {})
    out["evidence"].setdefault("target", [])
    out["evidence"].setdefault("mimic", [])
    out.setdefault("verdict", "DIFFERENT")
    # force confidence into [0,1] float if possible
    try:
        c = float(out.get("confidence", 0.0))
        out["confidence"] = max(0.0, min(1.0, c))
    except Exception:
        out["confidence"] = 0.0

    return out

if __name__ == "__main__":
    abstract = """A common method for developing agents with Language Models involves iteratively prompting the model, reflecting on its outputs, and updating the prompts until the task is completed. However, this approach faces challenges such as limited exploration of the decision space due to repetitive reflections and an inability to utilize insights from previously solved tasks."""
    target_idea = "Introduce DoT (Diversity of Thoughts), a framework that reduces redundant reflections to improve decision-space exploration and incorporates a task-agnostic memory component for knowledge retrieval from previously solved tasks, enhancing the agent's performance across various reasoning tasks."
    mimic_idea = "Develop a meta-learning framework for language model agents that incorporates a dynamic memory module. This module stores successful strategies and key insights from past interactions, enabling the agent to recall and apply previous solutions to new, but similar tasks. Additionally, integrate a diversity-driven prompting mechanism that generates varied and explorative prompts based on past inefficiencies in task-solving, thus widening the decision space and fostering innovative problem-solving approaches."

    result = judge_match(abstract, target_idea, mimic_idea)
    print(json.dumps(result, indent=2))
