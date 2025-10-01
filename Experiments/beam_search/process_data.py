#!/usr/bin/env python3
import sys, json, argparse, re

def parse_inner_json(s: str) -> dict:
    """
    The model's content is itself a JSON string. This function:
    - strips code fences if present
    - tries json.loads
    - falls back to a light regex if needed
    """
    if s is None:
        return {}
    s = s.strip()
    # remove code fences like ```json ... ```
    if s.startswith("```"):
        # keep only inside the first/last fence
        parts = re.split(r"^```[a-zA-Z]*\s*|\s*```$", s, flags=re.MULTILINE)
        s = "".join(p for p in parts if p is not None).strip()

    # try strict JSON first
    try:
        return json.loads(s)
    except Exception:
        pass

    # fallback: try to pull "problem" and "idea" via regex (very forgiving)
    problem = None
    idea = None
    m = re.search(r'"problem"\s*:\s*"(?P<val>(?:[^"\\]|\\.)*)"', s, flags=re.IGNORECASE)
    if m: problem = bytes(m.group("val"), "utf-8").decode("unicode_escape")
    m = re.search(r'"idea"\s*:\s*"(?P<val>(?:[^"\\]|\\.)*)"', s, flags=re.IGNORECASE)
    if m: idea = bytes(m.group("val"), "utf-8").decode("unicode_escape")
    out = {}
    if problem is not None: out["problem"] = problem
    if idea is not None: out["idea"] = idea
    return out

def extract_line(line: str):
    """
    Returns dict with keys: custom_id, problem, idea (if available), else None to skip.
    """
    try:
        obj = json.loads(line)
    except Exception:
        return None

    custom_id = obj.get("custom_id") or obj.get("Custom_ID") or obj.get("customId")
    # Find the nested message content
    content = None
    try:
        content = obj["response"]["body"]["choices"][0]["message"]["content"]
    except Exception:
        # Sometimes different layout; try a few alternates
        content = (
            obj.get("response", {})
               .get("body", {})
               .get("choices", [{}])[0]
               .get("message", {})
               .get("content")
        )
    inner = parse_inner_json(content) if content else {}

    problem = inner.get("problem") or inner.get("Problem")
    idea = inner.get("idea") or inner.get("Idea")

    if not (custom_id and (problem or idea)):
        return None

    # Normalize whitespace
    if isinstance(problem, str): problem = " ".join(problem.split())
    if isinstance(idea, str): idea = " ".join(idea.split())

    return {"custom_id": custom_id, "problem": problem, "idea": idea}

def main():
    ap = argparse.ArgumentParser(description="Extract custom_id, problem, idea from batch JSONL into a flat JSONL.")
    ap.add_argument("input", help="Input JSONL file")
    ap.add_argument("output", help="Output JSONL file")
    ap.add_argument("--keep-empty", action="store_true",
                    help="Write lines even if problem or idea is missing (defaults to skipping incomplete rows)")
    args = ap.parse_args()

    out_count = 0
    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = extract_line(line)
            if rec is None:
                if args.keep-empty:
                    # try to at least pass through custom_id if present
                    try:
                        obj = json.loads(line)
                        cid = obj.get("custom_id") or obj.get("Custom_ID") or obj.get("customId")
                    except Exception:
                        cid = None
                    rec = {"custom_id": cid, "problem": None, "idea": None}
                else:
                    continue
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_count += 1

    print(f"wrote {out_count} lines to {args.output}")

if __name__ == "__main__":
    main()

# USAGE: python process_data.py ../../Data/batch_output_sample.jsonl test.jsonl
