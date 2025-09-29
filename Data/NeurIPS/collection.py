#!/usr/bin/env python3
# neurips_scrape.py
import os
import sys
import time
import random
import logging
import argparse
from typing import List, Dict, Any, Tuple, Iterable

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO),
                    format="[%(asctime)s] %(levelname)s %(message)s")
logger = logging.getLogger("neurips-scraper")

API2 = "https://api2.openreview.net"

# ---------------- HTTP session w/ retries ----------------
def create_session_with_retries() -> requests.Session:
    s = requests.Session()
    retry_strategy = Retry(
        total=5, connect=5, read=5,
        backoff_factor=0.8,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    })
    return s

def _sleep_backoff(attempt: int, base: float = 1.0, max_delay: float = 60.0) -> None:
    delay = min(base * (2 ** attempt), max_delay)
    time.sleep(delay + random.uniform(0.1, 0.5) * delay)

# ---------------- Fetch helpers (API v2) ----------------
def fetch_notes_by_invitation(session: requests.Session, invitation: str,
                              start_offset: int = 0, step: int = 1000,
                              max_pages: int = 50, timeout: int = 45) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for page in range(max_pages):
        offset = start_offset + page * step
        url = f"{API2}/notes?invitation={requests.utils.quote(invitation, safe='')}&offset={offset}"
        for attempt in range(5):
            try:
                r = session.get(url, timeout=timeout)
                if r.status_code == 429:
                    _sleep_backoff(attempt + 1, base=1.5); continue
                r.raise_for_status()
                notes = r.json().get("notes", [])
                if not notes:
                    logger.info("No more notes invitation=%s offset=%d", invitation, offset)
                    return out
                out.extend(notes)
                logger.info("Fetched %d (total=%d) from %s", len(notes), len(out), invitation)
                break
            except requests.exceptions.RequestException:
                if attempt < 4: _sleep_backoff(attempt + 1, base=1.2)
                else: raise
    return out

def fetch_notes_by_venueid(session: requests.Session, venueid: str, extra_suffix: str = "",
                           start_offset: int = 0, step: int = 1000,
                           max_pages: int = 50, timeout: int = 45) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    base = f"{API2}/notes?content.venueid={requests.utils.quote(venueid, safe='')}{extra_suffix}"
    for page in range(max_pages):
        offset = start_offset + page * step
        url = f"{base}&offset={offset}"
        for attempt in range(5):
            try:
                r = session.get(url, timeout=timeout)
                if r.status_code == 429:
                    _sleep_backoff(attempt + 1, base=1.5); continue
                r.raise_for_status()
                notes = r.json().get("notes", [])
                if not notes:
                    logger.info("No more notes venueid=%s offset=%d", venueid, offset)
                    return out
                out.extend(notes)
                logger.info("Fetched %d (total=%d) from %s", len(notes), len(out), base)
                break
            except requests.exceptions.RequestException:
                if attempt < 4: _sleep_backoff(attempt + 1, base=1.2)
                else: raise
    return out

# -------------- Decision invitations (optional) --------------
LIKELY_DECISION_INVITES = [
    "Decision", "Paper_Decision", "Final_Decision", "Meta_Review", "Recommendation"
]

def fetch_decision_notes_any(session: requests.Session, year: int,
                             invites: Iterable[str], max_pages: int = 20) -> List[Dict[str, Any]]:
    all_decisions: List[Dict[str, Any]] = []
    for name in invites:
        inv = f"NeurIPS.cc/{year}/Conference/-/{name}"
        try:
            notes = fetch_notes_by_invitation(session, inv, start_offset=0, step=1000, max_pages=max_pages)
            if notes:
                logger.info("Found %d decision-like notes under %s", len(notes), inv)
                all_decisions.extend(notes)
        except requests.exceptions.RequestException:
            logger.info("Invitation %s not available/public", inv)
    return all_decisions

def _text_value(d: Dict[str, Any], key: str) -> str:
    return ((d.get(key, {}) or {}).get("value", "") or "").strip()

def decision_string_from_content(c: Dict[str, Any]) -> str:
    for k in ["decision", "Decision", "final_decision", "Final_Decision",
              "recommendation", "Recommendation", "meta_review", "summary_of_recommendation"]:
        v = _text_value(c, k)
        if v:
            return v
    return ""

def forums_with_accept(decision_notes: List[Dict[str, Any]]) -> set:
    acc = set()
    for n in decision_notes:
        dtext = decision_string_from_content(n.get("content", {}) or {})
        if dtext and "accept" in dtext.lower():
            fid = n.get("forum", "") or ""
            if fid: acc.add(fid)
    return acc

# ---------------- Extraction & labeling ----------------
def normalize_text_field(content: Dict[str, Any], key: str) -> str:
    return ((content.get(key, {}) or {}).get("value", "") or "").strip()

def list_field(content: Dict[str, Any], key: str) -> List[str]:
    return (content.get(key, {}) or {}).get("value", []) or []

def extract_rows(notes: List[Dict[str, Any]], year: int, decision_bucket: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for note in notes:
        try:
            content  = note.get("content", {}) or {}
            title    = normalize_text_field(content, "title")
            abstract = normalize_text_field(content, "abstract")
            kws      = [kw.lower() for kw in list_field(content, "keywords")]
            authors  = ", ".join(list_field(content, "authors"))
            forum    = note.get("forum", "") or ""

            # PRIMARY acceptance signal on API v2:
            pdate = note.get("pdate", None)  # non-null => accepted/published

            # Optional extras if visible:
            venue   = note.get("venue", "") or ""
            venueid = note.get("venueid", "") or ""
            c_venue   = ((content.get("venue", {}) or {}).get("value", "") or "")
            c_venueid = ((content.get("venueid", {}) or {}).get("value", "") or "")

            decision = decision_bucket or ""
            if not decision and pdate:
                decision = "Accept (Published)"

            # Mild backstop (NeurIPS often uses Poster/Spotlight/Oral as well)
            if not decision and (venueid == f"NeurIPS.cc/{year}/Conference" or c_venueid == f"NeurIPS.cc/{year}/Conference"):
                if ("Poster" in venue) or ("Poster" in c_venue):
                    decision = "Accept (Poster)"
                elif ("Spotlight" in venue) or ("Spotlight" in c_venue):
                    decision = "Accept (Spotlight)"
                elif ("Oral" in venue) or ("Oral" in c_venue):
                    decision = "Accept (Oral)"
                elif venue or c_venue:
                    decision = "Accept"

            rows.append({
                "year": year,
                "id": forum,
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "decision": decision,
                "keywords": kws,
                "pdate": pdate,
                "venue": venue,
                "venueid": venueid,
                "content_venue": c_venue,
                "content_venueid": c_venueid,
            })
        except Exception as e:
            logger.warning("Skipping note due to parsing error (id=%s): %s", note.get("id"), e)
    return rows

# ---------------- Orchestration ----------------
def collect_neurips_year(year: int, max_pages: int = 50, include_decisions: bool = True) -> pd.DataFrame:
    session = create_session_with_retries()
    venueid = f"NeurIPS.cc/{year}/Conference"

    buckets: List[Tuple[str, str]] = [
        ("", ""),  # main pool (accepted detected via pdate)
        ("/Withdrawn_Submission", "Withdrawn"),
        ("/Rejected_Submission", "Reject"),
        ("/Desk_Rejected_Submission", "Desk rejected"),
    ]

    best_by_forum: Dict[str, Dict[str, Any]] = {}

    def rank(dec: str) -> int:
        if isinstance(dec, str) and dec.startswith("Accept"): return 3
        if dec in ("Reject", "Desk rejected", "Withdrawn"):  return 2
        return 1

    for suffix, label in buckets:
        logger.info("Fetching %s submissions...", label or "main")
        notes = fetch_notes_by_venueid(session, venueid, extra_suffix=suffix, max_pages=max_pages)
        for r in extract_rows(notes, year, decision_bucket=label):
            fid = r["id"]
            if not fid: continue
            old = best_by_forum.get(fid)
            if old is None or rank(r["decision"]) > rank(old["decision"]):
                best_by_forum[fid] = r

    if include_decisions:
        logger.info("Probing decision invitations…")
        dec_notes   = fetch_decision_notes_any(session, year, LIKELY_DECISION_INVITES, max_pages=20)
        acc_forums  = forums_with_accept(dec_notes)
        if acc_forums:
            for fid in acc_forums:
                if fid in best_by_forum and not best_by_forum[fid]["decision"].startswith("Accept"):
                    best_by_forum[fid]["decision"] = "Accept (DecisionNote)"

    session.close()

    df = pd.DataFrame(list(best_by_forum.values()))
    # Optional: filter very-short abstracts
    if not df.empty:
        mask = df["abstract"].fillna("").str.len() >= 100
        removed = (~mask).sum()
        if removed:
            logger.info("Removing %d rows with abstract length < 100", removed)
        df = df[mask].reset_index(drop=True)
    return df

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(description="Fetch NeurIPS notes from OpenReview API v2.")
    ap.add_argument("--year", type=int, default=int(os.getenv("NEURIPS_YEAR", "2024")),
                    help="NeurIPS year to fetch (default: 2024)")
    ap.add_argument("--max-pages", type=int, default=50,
                    help="Max pages per bucket (1000 notes per page).")
    ap.add_argument("--no-decisions", action="store_true",
                    help="Skip probing decision invitations.")
    ap.add_argument("--out-prefix", type=str, default="neurips",
                    help="Output file prefix (default: neurips)")
    args = ap.parse_args()

    year = args.year
    logger.info("Starting NeurIPS %d collection", year)
    df = collect_neurips_year(year, max_pages=args.max_pages, include_decisions=not args.no_decisions)

    if df.empty:
        logger.warning("No data collected for NeurIPS %d.", year)
        return 0

    full_path = f"{args.out_prefix}{year}.parquet"
    df.to_parquet(full_path)
    logger.info("Saved full dataset → %s (rows=%d)", full_path, len(df))

    # Accepted = pdate set OR decision says Accept
    accepted = df[(df["pdate"].notna()) | (df["decision"].str.startswith("Accept", na=False))].copy()
    acc_path = f"{args.out_prefix}{year}_accepted.parquet"
    accepted.to_parquet(acc_path)
    logger.info("Saved accepted-only → %s (rows=%d)", acc_path, len(accepted))

    # Console summaries
    logger.info("Decision breakdown:")
    for k, v in df["decision"].value_counts(dropna=False).items():
        logger.info("  %-18s %5d", (k or "Unknown"), v)

    logger.info("Accepted breakdown:")
    for k, v in accepted["decision"].value_counts(dropna=False).items():
        logger.info("  %-18s %5d", (k or "Unknown"), v)

    if not accepted.empty:
        print("\nSample accepted:")
        print(accepted.loc[:, ["title", "decision"]].head(10).to_string(index=False))
    else:
        logger.info("No accepted rows detected — check if `pdate` is present for this year.")

    return 0

if __name__ == "__main__":
    sys.exit(main())
