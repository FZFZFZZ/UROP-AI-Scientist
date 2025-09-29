#!/usr/bin/env python3
# icml_scrape.py
"""
ICML OpenReview (API v2) scraper.

Key points:
- Submissions are fetched via invitation(s) (e.g., .../-/Submission or .../-/Blind_Submission).
- Withdrawals and desk rejections are fetched via their own invitations (e.g., .../-/Withdrawal, .../-/Desk_Rejection).
- Rejected papers are usually not public on OpenReview for ICML, so acceptance share will look high by design.
- 'pdate' is the strongest acceptance signal for API v2 (published/accepted papers).
"""

import os
import sys
import time
import random
import logging
import argparse
from typing import List, Dict, Any, Tuple, Iterable, Optional

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="[%(asctime)s] %(levelname)s %(message)s"
)
logger = logging.getLogger("icml-scraper")

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
                    _sleep_backoff(attempt + 1, base=1.5)
                    continue
                r.raise_for_status()
                notes = r.json().get("notes", [])
                if not notes:
                    logger.info("No more notes invitation=%s offset=%d", invitation, offset)
                    return out
                out.extend(notes)
                logger.info("Fetched %d (total=%d) from invitation=%s", len(notes), len(out), invitation)
                break
            except requests.exceptions.RequestException as e:
                if attempt < 4:
                    _sleep_backoff(attempt + 1, base=1.2)
                else:
                    logger.info("Invitation %s not available/public (%s)", invitation, e)
                    return out
    return out

def fetch_notes_by_invitations(session: requests.Session, invitations: List[str],
                               max_pages: int = 50, timeout: int = 45) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for inv in invitations:
        out.extend(fetch_notes_by_invitation(session, inv, start_offset=0,
                                             step=1000, max_pages=max_pages, timeout=timeout))
    return out

# -------------- Decision invitations (optional) --------------
LIKELY_DECISION_INVITES = [
    "Decision", "Paper_Decision", "Final_Decision", "Meta_Review", "Recommendation"
]

def fetch_decision_notes_any(session: requests.Session, year: int,
                             invites: Iterable[str], max_pages: int = 20) -> List[Dict[str, Any]]:
    all_decisions: List[Dict[str, Any]] = []
    for name in invites:
        inv = f"ICML.cc/{year}/Conference/-/{name}"
        all_decisions.extend(fetch_notes_by_invitation(
            session, inv, start_offset=0, step=1000, max_pages=max_pages))
    if all_decisions:
        logger.info("Found %d decision-like notes across candidate invitations", len(all_decisions))
    return all_decisions

# ---------------- Extraction & labeling ----------------
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
            if fid:
                acc.add(fid)
    return acc

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

            # In API v2, pdate present => published (accepted)
            pdate = note.get("pdate", None)

            # Optional extras
            venue   = note.get("venue", "") or ""
            venueid = note.get("venueid", "") or ""
            c_venue   = ((content.get("venue", {}) or {}).get("value", "") or "")
            c_venueid = ((content.get("venueid", {}) or {}).get("value", "") or "")

            decision = decision_bucket or ""
            if not decision and pdate:
                decision = "Accept (Published)"

            # Backstop for track labels in venue strings
            if not decision and (venueid == f"ICML.cc/{year}/Conference" or c_venueid == f"ICML.cc/{year}/Conference"):
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
                "source_invitation": note.get("invitation", ""),
            })
        except Exception as e:
            logger.warning("Skipping note due to parsing error (id=%s): %s", note.get("id"), e)
    return rows

# ---------------- Venue metadata helpers ----------------
def get_group_content_value(session: requests.Session, group_id: str, key: str) -> Optional[str]:
    """Fetch a group and return content[key].value if present."""
    url = f"{API2}/groups?id={requests.utils.quote(group_id, safe='')}"
    try:
        r = session.get(url, timeout=30)
        if not r.ok:
            return None
        groups = r.json().get("groups", [])
        if not groups:
            return None
        content = groups[0].get("content", {}) or {}
        val = ((content.get(key, {}) or {}).get("value", "") or "").strip()
        return val or None
    except requests.exceptions.RequestException:
        return None

def possible_submission_invitations(venueid: str) -> List[str]:
    # ICML commonly uses single-blind 'Submission'; keep Blind_Submission for robustness.
    return [
        f"{venueid}/-/Submission",
        f"{venueid}/-/Blind_Submission",
    ]

def possible_withdrawal_invitations(session: requests.Session, venueid: str) -> List[str]:
    custom = get_group_content_value(session, venueid, "withdrawal_name")
    name = custom or "Withdrawal"
    # Some venues historically used "Withdrawn_Submission"
    return [
        f"{venueid}/-/{name}",
        f"{venueid}/-/Withdrawn_Submission",
    ]

def possible_desk_reject_invitations(session: requests.Session, venueid: str) -> List[str]:
    custom = get_group_content_value(session, venueid, "desk_rejection_name")
    name = custom or "Desk_Rejection"
    # Some venues historically used "Desk_Rejected_Submission"
    return [
        f"{venueid}/-/{name}",
        f"{venueid}/-/Desk_Rejected_Submission",
    ]

def possible_rejected_invitations(venueid: str) -> List[str]:
    # Rejections are often not public; try anyway.
    return [
        f"{venueid}/-/Rejected_Submission",
        f"{venueid}/-/Paper_Reject",  # rare / fallback
    ]

# ---------------- Orchestration ----------------
def collect_icml_year(year: int, max_pages: int = 50, include_decisions: bool = True) -> pd.DataFrame:
    session = create_session_with_retries()
    venueid = f"ICML.cc/{year}/Conference"

    # 1) Submissions (main pool)
    subm_notes = fetch_notes_by_invitations(
        session, possible_submission_invitations(venueid), max_pages=max_pages)

    best_by_forum: Dict[str, Dict[str, Any]] = {}

    def rank(dec: str) -> int:
        if isinstance(dec, str) and dec.startswith("Accept"):
            return 4
        if dec in ("Withdrawn", "Desk rejected"):
            return 3
        if dec in ("Reject",):
            return 2
        return 1

    for r in extract_rows(subm_notes, year, decision_bucket=""):
        fid = r["id"]
        if not fid:
            continue
        old = best_by_forum.get(fid)
        if old is None or rank(r["decision"]) > rank(old["decision"]):
            best_by_forum[fid] = r

    # 2) Withdrawals
    w_invs = possible_withdrawal_invitations(session, venueid)
    w_notes = fetch_notes_by_invitations(session, w_invs, max_pages=max_pages)
    for n in w_notes:
        fid = n.get("forum", "")
        if fid and fid in best_by_forum:
            best_by_forum[fid]["decision"] = "Withdrawn"
            best_by_forum[fid]["source_invitation"] = n.get("invitation", "")

    # 3) Desk Rejections
    dr_invs = possible_desk_reject_invitations(session, venueid)
    dr_notes = fetch_notes_by_invitations(session, dr_invs, max_pages=max_pages)
    for n in dr_notes:
        fid = n.get("forum", "")
        if fid and fid in best_by_forum:
            best_by_forum[fid]["decision"] = "Desk rejected"
            best_by_forum[fid]["source_invitation"] = n.get("invitation", "")

    # 4) Explicit Rejections (usually private; try common names)
    rej_invs = possible_rejected_invitations(venueid)
    rej_notes = fetch_notes_by_invitations(session, rej_invs, max_pages=max_pages)
    for n in rej_notes:
        fid = n.get("forum", "")
        if fid and fid in best_by_forum:
            best_by_forum[fid]["decision"] = "Reject"
            best_by_forum[fid]["source_invitation"] = n.get("invitation", "")

    # 5) Decision-like notes (many may be private)
    if include_decisions:
        logger.info("Probing decision invitations…")
        dec_notes  = fetch_decision_notes_any(session, year, LIKELY_DECISION_INVITES, max_pages=20)
        acc_forums = forums_with_accept(dec_notes)
        if acc_forums:
            for fid in acc_forums:
                if fid in best_by_forum and not best_by_forum[fid]["decision"].startswith("Accept"):
                    best_by_forum[fid]["decision"] = "Accept (DecisionNote)"

    session.close()

    df = pd.DataFrame(list(best_by_forum.values()))
    if df.empty:
        return df

    # Backfill acceptance by pdate (primary signal on API v2)
    has_pdate = df["pdate"].notna()
    df.loc[has_pdate & ~df["decision"].str.startswith("Accept", na=False), "decision"] = "Accept (Published)"

    # Optional: infer track labels from venue strings
    mask_venue = (df["venue"].fillna("") + "|" + df["content_venue"].fillna(""))
    df.loc[mask_venue.str.contains("Poster"), "decision"] = "Accept (Poster)"
    df.loc[mask_venue.str.contains("Spotlight"), "decision"] = "Accept (Spotlight)"
    df.loc[mask_venue.str.contains("Oral"), "decision"] = "Accept (Oral)"

    # Filter very-short abstracts
    keep = df["abstract"].fillna("").str.len() >= 100
    removed = (~keep).sum()
    if removed:
        logger.info("Removing %d rows with abstract length < 100", removed)
    df = df[keep].reset_index(drop=True)
    return df

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(description="Fetch ICML notes from OpenReview API v2.")
    ap.add_argument("--year", type=int, default=int(os.getenv("ICML_YEAR", "2025")),
                    help="ICML year to fetch (default: 2025)")
    ap.add_argument("--max-pages", type=int, default=50,
                    help="Max pages per invitation (1000 notes per page).")
    ap.add_argument("--no-decisions", action="store_true",
                    help="Skip probing decision invitations.")
    ap.add_argument("--out-prefix", type=str, default="icml",
                    help="Output file prefix (default: icml)")
    args = ap.parse_args()

    year = args.year
    logger.info("Starting ICML %d collection", year)
    df = collect_icml_year(year, max_pages=args.max_pages, include_decisions=not args.no_decisions)

    if df.empty:
        logger.warning("No data collected for ICML %d.", year)
        return 0

    # Quick sanity summary by source invitation
    src_counts = df["source_invitation"].fillna("").value_counts()
    if not src_counts.empty:
        logger.info("Source invitation breakdown:")
        for k, v in src_counts.items():
            logger.info("  %-50s %5d", (k or "unknown"), v)

    full_path = f"{args.out_prefix}{year}.parquet"
    df.to_parquet(full_path)
    logger.info("Saved full dataset → %s (rows=%d)", full_path, len(df))

    # Accepted: pdate set OR decision startswith Accept
    accepted = df[(df["pdate"].notna()) | (df["decision"].str.startswith("Accept", na=False))].copy()
    acc_path = f"{args.out_prefix}{year}_accepted.parquet"
    accepted.to_parquet(acc_path)
    logger.info("Saved accepted-only → %s (rows=%d)", acc_path, len(accepted))

    # Console summaries
    logger.info("Decision breakdown (all rows):")
    for k, v in df["decision"].fillna("Unknown").value_counts(dropna=False).items():
        logger.info("  %-18s %5d", k, v)

    logger.info("Accepted breakdown (accepted-only slice):")
    for k, v in accepted["decision"].fillna("Unknown").value_counts(dropna=False).items():
        logger.info("  %-18s %5d", k, v)

    if not accepted.empty:
        print("\nSample accepted:")
        print(accepted.loc[:, ["title", "decision"]].head(10).to_string(index=False))
    else:
        logger.info("No accepted rows detected — check if `pdate` is present for this year.")

    return 0

if __name__ == "__main__":
    sys.exit(main())

