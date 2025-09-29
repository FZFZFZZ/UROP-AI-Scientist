#!/usr/bin/env python3
import os, sys, json, argparse, hashlib, re
from pathlib import Path
from typing import Any, Dict, List, Optional
from openai import OpenAI

DEF_MODEL = "gpt-4.1"
DEF_INPUT_JSONL = "batch_input.jsonl"
DEF_OUTPUT_JSONL = "batch_output.jsonl"
DEF_COLLECTED_JSON = "accepted_paper_problem_idea.json"

# ---------- Helpers ----------
def read_json_any(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    return v
            return [data]
    except json.JSONDecodeError:
        items = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                try:
                    items.append(json.loads(line))
                except:
                    pass
        return items

def stable_id(rec: Dict[str, Any], source: str, idx: int) -> str:
    key = json.dumps({"source": source, "idx": idx, "abstract": rec.get("abstract")}, ensure_ascii=False)
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]

def parse_model_json(s: str) -> Dict[str, Any]:
    s = s.strip()
    # strip triple backtick fences if present
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", s)
        s = re.sub(r"\n```$", "", s)
    # direct JSON
    try:
        return json.loads(s)
    except Exception:
        # extract first {...} block
        m = re.search(r"\{[\s\S]*\}", s)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {"problem": "", "idea": "", "_raw": s}

def load_records(root: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    datasets = [
        ("ICLR", root/"ICLR"/"accepted_papers.json"),
        ("ICML", root/"ICML"/"accepted_papers.json"),
        ("NeurIPS", root/"NeurIPS"/"accepted_papers.json"),
    ]
    records: List[Dict[str, Any]] = []
    for name, path in datasets:
        if not path.exists():
            print(f"[WARN] Missing: {path} — skipping {name}", file=sys.stderr)
            continue
        items = read_json_any(path)
        for i, rec in enumerate(items):
            rec["_source"] = name
            rec["_idx"] = i
            records.append(rec)
    if limit is not None:
        records = records[:limit]
    return records

# ---------- Subcommands ----------
def cmd_make_input(args):
    root = Path(args.root)
    prompt_path = root / "prompt.txt"
    if not prompt_path.exists():
        sys.exit("[ERROR] prompt.txt not found")
    # normalize prompt newlines to \n; json.dumps will escape them as \\n in-file
    system_prompt = prompt_path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not system_prompt:
        sys.exit("[ERROR] prompt.txt is empty.")

    records = load_records(root, args.limit)

    out_path = Path(args.input)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: skip existing custom_id if requested
    seen = set()
    if args.resume and out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if "custom_id" in obj:
                        seen.add(obj["custom_id"])
                except Exception:
                    pass
        print(f"[INFO] Resume: {len(seen)} lines already exist.", file=sys.stderr)

    count = 0
    mode = "a" if args.resume and out_path.exists() else "w"
    # UTF-8 (no BOM): default 'utf-8' is correct; do NOT use 'utf-8-sig'
    with out_path.open(mode, encoding="utf-8") as fh:
        for rec in records:
            uid = stable_id(rec, rec["_source"], rec["_idx"])
            if uid in seen:
                continue

            # User content is a JSON string of the abstract/keywords.
            # json.dumps here ensures inner quotes are escaped (\"), and no literal newlines.
            user_payload = json.dumps(
                {
                    "abstract": rec.get("abstract", ""),
                    "keywords": rec.get("keywords", []),
                },
                ensure_ascii=False,
            )

            body = {
                "model": args.model,
                "temperature": 0,  # deterministic
                "response_format": {"type": "json_object"},  # force JSON output
                "max_tokens": 256,  # safety cap
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_payload},
                ],
            }

            line_obj = {
                "custom_id": uid,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }

            # json.dumps on the whole line guarantees:
            # - one JSON object per line
            # - internal newlines in strings are escaped as \\n
            # - inner quotes are escaped as \"
            # Use compact separators to avoid accidental whitespace/newlines.
            serialized = json.dumps(line_obj, ensure_ascii=False, separators=(",", ":"))
            # sanity check: no literal newline characters in the serialized line
            if "\n" in serialized or "\r" in serialized:
                # Extremely defensive: replace any stray literal newlines (shouldn't happen)
                serialized = serialized.replace("\r", "").replace("\n", "\\n")
            fh.write(serialized + "\n")
            count += 1

    print(f"[DONE] Wrote {count} lines → {out_path}")



def cmd_submit(args):
    client = OpenAI()
    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"[ERROR] Input file not found: {input_path}")

    # Upload input JSONL as a file with purpose='batch'
    with input_path.open("rb") as f:
        up = client.files.create(file=f, purpose="batch")
    # Create batch
    batch = client.batches.create(
        input_file_id=up.id,
        endpoint="/v1/chat/completions",
        completion_window=args.window,  # e.g., "24h"
    )
    print(json.dumps({
        "batch_id": batch.id,
        "status": batch.status,
        "input_file_id": up.id,
        "dashboard": "https://platform.openai.com/batches"
    }, indent=2))

def cmd_status(args):
    client = OpenAI()
    batch = client.batches.retrieve(args.batch_id)
    print(json.dumps({
        "batch_id": batch.id,
        "status": batch.status,
        "created_at": batch.created_at,
        "in_progress": getattr(batch, "in_progress", None),
        "request_counts": getattr(batch, "request_counts", None),
        "output_file_id": getattr(batch, "output_file_id", None),
        "error_file_id": getattr(batch, "error_file_id", None),
    }, indent=2, ensure_ascii=False))

def _download_file(client: OpenAI, file_id: str, out_path: Path):
    # Stream file content to disk
    resp = client.files.content(file_id)
    content = resp.read().decode("utf-8")
    out_path.write_text(content, encoding="utf-8")
    return out_path

def cmd_responses(args):
    client = OpenAI()
    batch = client.batches.retrieve(args.batch_id)
    out_file = getattr(batch, "output_file_id", None)
    if not out_file:
        sys.exit("[ERROR] Batch has no output_file_id yet. Try again later.")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _download_file(client, out_file, out_path)
    print(f"[DONE] Downloaded responses → {out_path}")

def cmd_errors(args):
    client = OpenAI()
    batch = client.batches.retrieve(args.batch_id)
    err_file = getattr(batch, "error_file_id", None)
    if not err_file:
        sys.exit("[INFO] Batch has no error_file_id (great!)")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _download_file(client, err_file, out_path)
    print(f"[DONE] Downloaded errors → {out_path}")

def cmd_collect(args):
    # Parse batch output JSONL -> final JSON of {uid, problem, idea}
    in_path = Path(args.responses)
    if not in_path.exists():
        sys.exit(f"[ERROR] Responses file not found: {in_path}")

    results: List[Dict[str, Any]] = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
            except:
                continue
            # Expected structure: {"custom_id": "...", "response": {"body": {...}}}
            uid = obj.get("custom_id")
            body = (obj.get("response") or {}).get("body", {})
            try:
                content = body["choices"][0]["message"]["content"]
            except Exception:
                content = ""
            parsed = parse_model_json(content or "")
            results.append({
                "uid": uid,
                "problem": parsed.get("problem",""),
                "idea": parsed.get("idea",""),
            })

    out_path = Path(args.out)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] Wrote {len(results)} items → {out_path}")

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="OpenAI Batch Pipeline for Problem–Idea Extraction")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_make = sub.add_parser("make-input", help="Build batch input JSONL from ICLR/ICML/NeurIPS using prompt.txt")
    ap_make.add_argument("--root", default=".", help="Project root containing prompt.txt and datasets")
    ap_make.add_argument("--input", default=DEF_INPUT_JSONL, help="Output batch input JSONL path")
    ap_make.add_argument("--model", default=DEF_MODEL, help="OpenAI model (default: %(default)s)")
    ap_make.add_argument("--limit", type=int, default=None, help="Limit total items (debug)")
    ap_make.add_argument("--resume", action="store_true", help="Append & skip existing custom_id lines if input exists")
    ap_make.set_defaults(func=cmd_make_input)

    ap_submit = sub.add_parser("submit", help="Upload input JSONL and create a batch job")
    ap_submit.add_argument("--input", default=DEF_INPUT_JSONL, help="Batch input JSONL path")
    ap_submit.add_argument("--window", default="24h", help="Completion window, e.g., 24h")
    ap_submit.set_defaults(func=cmd_submit)

    ap_status = sub.add_parser("status", help="Retrieve batch status")
    ap_status.add_argument("batch_id", help="Batch ID")
    ap_status.set_defaults(func=cmd_status)

    ap_resp = sub.add_parser("responses", help="Download batch responses JSONL")
    ap_resp.add_argument("batch_id", help="Batch ID")
    ap_resp.add_argument("--output", default=DEF_OUTPUT_JSONL, help="Where to save responses JSONL")
    ap_resp.set_defaults(func=cmd_responses)

    ap_err = sub.add_parser("errors", help="Download batch error JSONL (if any)")
    ap_err.add_argument("batch_id", help="Batch ID")
    ap_err.add_argument("--output", default="batch_errors.jsonl", help="Where to save errors JSONL")
    ap_err.set_defaults(func=cmd_errors)

    ap_collect = sub.add_parser("collect", help="Collect responses JSONL → final JSON ({uid,problem,idea})")
    ap_collect.add_argument("--responses", default=DEF_OUTPUT_JSONL, help="Downloaded responses JSONL")
    ap_collect.add_argument("--out", default=DEF_COLLECTED_JSON, help="Final JSON output path")
    ap_collect.set_defaults(func=cmd_collect)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()

