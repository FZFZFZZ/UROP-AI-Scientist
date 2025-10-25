# Baseline is classical explain-then-answer approach, point-wise
# run under ./Experiments/MDP

import sys, os, json
from MDP.helper import get_response

MODEL = "gpt-4.1"

SYS = """
You are an expert research analyst evaluating idea similarity point-wise in the NLP and AI field.
Your task: reason carefully before giving a similarity score (a float from 0 to 1, precision 2dp).
You must only output a JSON: {"Explanation": "...", "Score": "..."}
"""

def point_wise_sim(idea: str, context: str, target_idea: str) -> float:

    user_prompt = f"""
You are evaluating idea similarity POINT-WISE for two short ideas (curr vs target) within the context below.

[Domain context]
{context}

Explain at the angle of approach, scope, and degree-of-detail, then compute a similarity score between these two ideas.

Current Idea:
<<<
{idea}
>>>

Target Idea:
<<<
{target_idea}
>>>
"""
    resp = get_response(
        model=MODEL,
        system_prompt=SYS,
        user_prompt=user_prompt,
        temperature=0,
        priority=False
    )

    try:
        data = json.loads(resp)
        return float(data.get("Score", 0))
    except Exception:
        print("Malformed JSON:", resp)
        return 0.0


