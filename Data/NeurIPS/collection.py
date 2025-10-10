#!/usr/bin/env python3
# neurips_accepts.py
"""
Fetch NeurIPS accepted papers from OpenReview (accepted-only, robust).

Strategy order:
  1) /notes?venue=NeurIPS.cc/{year}/Conference (API v2)
  2) /notes?venue=NeurIPS.cc/{year}/Conference (API v1 fallback)
  3) Fallback: fetch Submission/Blind_Submission, then locally filter
     note.venueid or content.venueid == NeurIPS.cc/{year}/Conference

Outputs:
  neurips{year}_accepted.parquet
  neurips{year}_accepted.csv (if --csv)
"""

import os, sys, time, random, logging, argparse
from typing import List, Dict, Any
import requests
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO),
                    format="[%(asctime)s] %(levelname)s %(message)s")
logger = logging.getLogger("neurips-accepts")

API2 = "https://api2.openreview.net"
API1 = "https://api.openreview.net"

# ---------------- Session ----------------
def session_with_retries() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=5, connect=5, read=5, backoff_factor=0.8,
                  status_forcelist=[429,500,502,503,504],
                  allowed_methods=frozenset(["GET"]),
                  raise_on_status=False)
    ad = HTTPAdapter(max_retries=retry)
    s.mount("https://", ad); s.mount("http://", ad)
    s.headers.update({
        "User-Agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
        "Accept":"application/json", "Accept-Language":"en-US,en;q=0.9"
    })
    return s

def _sleep(attempt: int, base=1.2, cap=60.0):
    delay = min(base * (2 ** attempt), cap)
    time.sleep(delay * (1 + random.uniform(0.1, 0.3)))

# ---------------- Fetch ----------------
def fetch_notes(session: requests.Session, base: str, params: Dict[str,Any],
                step: int = 1000, max_pages: int = 80, timeout: int = 45) -> List[Dict[str,Any]]:
    out: List[Dict[str,Any]] = []
    for page in range(max_pages):
        offset = page * step
        q = "&".join(f"{k}={requests.utils.quote(str(v), safe='')}" for k,v in params.items())
        url = f"{base}/notes?{q}&offset={offset}"
        for attempt in range(5):
            try:
                r = session.get(url, timeout=timeout)
                if r.status_code == 429:
                    _sleep(attempt); continue
                r.raise_for_status()
                js = r.json()
                notes = js.get("notes", js.get("results", []))  # api2 vs api1
                if not notes:
                    logger.info("No more notes for %s offset=%d", params, offset)
                    return out
                out.extend(notes)
                logger.info("Fetched %d (total=%d) via %s", len(notes), len(out), url)
                break
            except requests.exceptions.RequestException as e:
                if attempt < 4: _sleep(attempt); continue
                logger.warning("Fetch failed for %s (%s)", url, e)
                return out
    return out

# ---------------- Extract helpers ----------------
def cval(c: Dict[str,Any], key: str) -> str:
    return ((c.get(key, {}) or {}).get("value", "") or "").strip()

def clist(c: Dict[str,Any], key: str) -> list:
    return (c.get(key, {}) or {}).get("value", []) or []

def to_row(note: Dict[str,Any], year: int) -> Dict[str,Any]:
    content = note.get("content", {}) or {}
    return {
        "year": year,
        "id": note.get("forum", "") or "",
        "title": cval(content, "title"),
        "abstract": cval(content, "abstract"),
        "authors": ", ".join(clist(content, "authors")),
        "authorids": ", ".join(clist(content, "authorids")) if clist(content, "authorids") else "",
        "keywords": [str(k).lower() for k in clist(content, "keywords")],
        "pdate": note.get("pdate", None),
        "venue": note.get("venue", "") or "",
        "venueid": note.get("venueid", "") or "",
        "content_venue": cval(content, "venue"),
        "content_venueid": cval(content, "venueid"),
        "pdf": cval(content, "pdf"),
        "source_invitation": note.get("invitation","") or "",
    }

def infer_decision(venue_str: str) -> str:
    v = (venue_str or "").lower()
    if "oral" in v: return "Accept (Oral)"
    if "spotlight" in v: return "Accept (Spotlight)"
    if "poster" in v: return "Accept (Poster)"
    return "Accept"

# ---------------- Core ----------------
def collect_neurips_accepted(year: int, max_pages: int = 80) -> pd.DataFrame:
    sess = session_with_retries()
    conference = f"NeurIPS.cc/{year}/Conference"

    # Plan A: API v2, by venue
    notes = fetch_notes(sess, API2, {"venue": conference}, max_pages=max_pages)
    if not notes:
        logger.warning("API v2 with venue= returned 0; trying API v1…")
        # Plan B: API v1, by venue
        notes = fetch_notes(sess, API1, {"venue": conference}, max_pages=max_pages)

    # Plan C: fallback via submission + local filter
    if not notes:
        logger.warning("No accepted pool visible; falling back to Submission pool for local filtering.")
        subs: List[Dict[str,Any]] = []
        for inv in (f"{conference}/-/Submission", f"{conference}/-/Blind_Submission"):
            subs.extend(fetch_notes(sess, API2, {"invitation": inv}, max_pages=max_pages))
        filtered = []
        for n in subs:
            vid  = n.get("venueid","") or ""
            cvid = cval(n.get("content",{}) or {}, "venueid")
            if vid == conference or cvid == conference:
                filtered.append(n)
        notes = filtered

    sess.close()

    if not notes:
        logger.warning("No accepted notes visible yet for NeurIPS %d.", year)
        return pd.DataFrame(columns=[
            "year","id","title","abstract","authors","authorids","keywords",
            "decision","pdate","venue","venueid","content_venue","content_venueid","pdf","source_invitation"
        ])

    rows = []
    for n in notes:
        r = to_row(n, year)
        # hard gate to Conference
        if r["venueid"] != conference and r["content_venueid"] != conference:
            continue
        merged_v = (r["venue"] or "") + "|" + (r["content_venue"] or "")
        r["decision"] = infer_decision(merged_v)
        rows.append(r)

    df = pd.DataFrame(rows).drop_duplicates(subset=["id"]).reset_index(drop=True)

    # Hygiene
    if not df.empty:
        keep = df["title"].fillna("").str.len() > 0
        df = df[keep].reset_index(drop=True)

    return df

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(description="Fetch NeurIPS accepted-only from OpenReview.")
    ap.add_argument("--year", type=int, default=int(os.getenv("NEURIPS_YEAR","2024")))
    ap.add_argument("--max-pages", type=int, default=80)
    ap.add_argument("--out-prefix", type=str, default="neurips")
    ap.add_argument("--csv", action="store_true")
    args = ap.parse_args()

    df = collect_neurips_accepted(args.year, max_pages=args.max_pages)
    outpq = f"{args.out_prefix}{args.year}_accepted.parquet"
    df.to_parquet(outpq)
    logging.info("Saved → %s (rows=%d)", outpq, len(df))
    if args.csv:
        outcsv = f"{args.out_prefix}{args.year}_accepted.csv"
        df.to_csv(outcsv, index=False)
        logging.info("Saved CSV → %s", outcsv)

    if not df.empty:
        logging.info("Presentation breakdown:")
        for k,v in df["decision"].value_counts().items():
            logging.info("  %-18s %5d", k, v)
        print("\nSample accepted (title | decision):")
        with pd.option_context("display.max_colwidth", 120):
            print(df.loc[:,["title","decision"]].head(10).to_string(index=False))
    else:
        logging.info("No accepted notes visible yet.")

if __name__ == "__main__":
    sys.exit(main())

