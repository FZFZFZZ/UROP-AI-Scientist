# common.py
import os, json, time, hashlib, argparse
from pathlib import Path
from typing import Iterable, Dict, Any, List, Optional, Callable

# ---- Config
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
DEFAULT_STUDENT_MODEL = os.environ.get("STUDENT_MODEL", "llama3.3")  # ollama local
DEFAULT_TEACHER_MODEL = os.environ.get("TEACHER_MODEL", "gpt-4.1")
DEFAULT_EVALUATOR_MODEL = os.environ.get("EVALUATOR_MODEL", "gpt-4.1")

LOG_DIR = Path(os.environ.get("LOG_DIR", "logs"))
LOG_DIR.mkdir(exist_ok=True, parents=True)

def sha_uid(payload: Dict[str, Any], prefix: str = "") -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    h = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}{h}" if prefix else h

# ---- JSONL IO
def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def append_jsonl(path: Path, obj: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def write_jsonl(path: Path, objects: Iterable[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for obj in objects:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# ---- Retry
def retry(fn: Callable[[], Any], attempts: int = 6, base: float = 0.5):
    err = None
    for k in range(attempts):
        try:
            return fn()
        except Exception as e:
            err = e
            if k == attempts - 1:
                raise
            time.sleep(base * (2 ** k))
    raise err

# ---- CLI common
def add_common_args(ap: argparse.ArgumentParser):
    ap.add_argument("--in", dest="infile", required=False, help="Input JSONL (items).")
    ap.add_argument("--out", dest="outfile", required=False, help="Output JSONL.")
    ap.add_argument("--limit", type=int, default=None, help="Limit items.")
    ap.add_argument("--resume", action="store_true", help="Skip items already in --out by uid.")
    ap.add_argument("--workers", type=int, default=4, help="Parallel workers (teacher/evaluator).")
    ap.add_argument("--log", default=None, help="Optional log JSONL (append).")

