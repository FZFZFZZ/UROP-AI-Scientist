from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import time
import html
import re
import requests
import xml.etree.ElementTree as ET
from typing import List, Dict
from or_to_arxiv import get_arxiv_id

from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

from helper import get_response

EMBED_MODEL = "text-embedding-3-large"
ARXIV_API = "http://export.arxiv.org/api/query"
ARXIV_MAX_RESULTS = 40
TOP_K = 3
REQUEST_TIMEOUT = 20

Settings.embed_model = OpenAIEmbedding(model=EMBED_MODEL)

def search_arxiv(query: str, max_results: int) -> List[Dict]:
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }

    for attempt in range(3):
        try:
            resp = requests.get(ARXIV_API, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            xml = resp.content
            break
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(1.2 * (attempt + 1))
    root = ET.fromstring(xml)
    ns = {"atom": "http://www.w3.org/2005/Atom"}

    results = []
    for entry in root.findall("atom:entry", ns):
        title = entry.findtext("atom:title", default="", namespaces=ns).strip()
        summary = entry.findtext("atom:summary", default="", namespaces=ns).strip()
        
        link = entry.findtext("atom:id", default="", namespaces=ns).strip()
        if not link:
            for lk in entry.findall("atom:link", ns):
                href = lk.attrib.get("href", "")
                if "arxiv.org" in href:
                    link = href
                    break

        authors = [
            a.findtext("atom:name", default="", namespaces=ns).strip()
            for a in entry.findall("atom:author", ns)
        ]

        title = html.unescape(title)
        summary = html.unescape(summary)

        if not summary:
            summary = "(No abstract available)"

        results.append(
            {
                "title": title,
                "summary": summary,
                "authors": authors,
                "link": link,
            }
        )

    seen = set()
    deduped = []
    for p in results:
        key = p["title"].lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(p)
    return deduped


def build_documents(papers: List[Dict]) -> List[Document]:

    docs: List[Document] = []
    for p in papers:
        text = f"Title: {p['title']}\n\nAbstract:\n{p['summary']}"
        metadata = {"title": p["title"], "authors": p["authors"], "link": p["link"]}
        docs.append(Document(text=text, metadata=metadata))
    return docs


def format_result(response, top_k: int = TOP_K):
    res = []

    try:
        nodes = response.source_nodes[:top_k]
    except Exception:
        nodes = []

    if not nodes:
        #print(str(response).strip())
        return

    for i, n in enumerate(nodes, 1):
        md = n.metadata or {}
        title = md.get("title", "Untitled")
        link = md.get("link", "")
        score = getattr(n, "score", None)

        # Extract arXiv ID from the link (e.g., '2410.12345v2')
        match = re.search(r'arxiv\.org/(abs|pdf)/([\w\.\-]+)', link)
        arxiv_id = match.group(2) if match else "N/A"
        res.append(arxiv_id)

        # print(f"{i}. {title}")
        # print(f"   arXiv ID: {arxiv_id}")
        # if score is not None:
        #     print(f"   Score   : {score:.4f}")
        # print()
    return res

def judge_search(or_id, idea):
    return get_arxiv_id(or_id) in MRR(idea)

def search(idea):
    # 1) direct search
    papers = search_arxiv(idea, max_results=ARXIV_MAX_RESULTS)

    if not papers:
        raise SystemExit("No results from arXiv. Try another query or increase max_results.")

    # 2) embedding index
    docs = build_documents(papers)
    index = VectorStoreIndex.from_documents(docs)

    # 3) embedding match
    query_engine = index.as_query_engine(similarity_top_k=TOP_K)
    response = query_engine.query(idea)

    # 4) ids
    res = format_result(response, top_k=TOP_K)
    return res or []


def make_key_word(idea):
    """use get_response(model: str, system_prompt: str, user_prompt: str, *, temperature: float = 0.2, priority: bool = False) and extract a list of keywords from an idea. must be suitable for search. output a list of keyword lists"""
    import json

    SYS = (
        "You extract compact, search-friendly keyword sets for arXiv.\n"
        "- Return ONLY valid JSON: a list of lists of strings.\n"
        "- 8–12 lists total; each list 6–12 items.\n"
        '- Include quoted phrases where useful (e.g., "diversity of thoughts").\n'
        "- No commentary."
    )
    USR = f"Text:\n{idea}\n\nTask: Produce 8–12 DISTINCT keyword lists covering different facets. Return JSON list of lists."

    try:
        resp = get_response(
            model="gpt-5",
            system_prompt=SYS,
            user_prompt=USR,
            temperature=0.2,
        )
        obj = json.loads(resp)
        if isinstance(obj, list) and all(isinstance(x, list) for x in obj):
            # normalize strings & drop empties
            cleaned = []
            for lst in obj:
                lst_clean = [str(t).strip() for t in lst if str(t).strip()]
                if lst_clean:
                    cleaned.append(lst_clean)
            if cleaned:
                print(cleaned)
                return cleaned
    except Exception:
        pass

    # Fallback (single facet) if parsing fails
    return [[idea]]


def MRR(idea):
    """base function is search. write MRR. return top-k result in a list"""
    import os
    from collections import defaultdict

    facets = make_key_word(idea)

    # run retrieval for each facet
    facet_rankings = []
    for kw_list in facets:
        q = " ".join(kw_list)
        ranked_ids = search(q)  # uses TOP_K cutoff internally
        facet_rankings.append(ranked_ids)

    gold_or_id = os.environ.get("GOLD_OR_ID", "").strip()
    if gold_or_id:
        gold = get_arxiv_id(gold_or_id)
        rrs = []
        for ranked in facet_rankings:
            rr = 0.0
            for i, pid in enumerate(ranked, 1):
                if pid == gold:
                    rr = 1.0 / i
                    break
            rrs.append(rr)
        mrr = sum(rrs) / max(1, len(rrs))
        print(f"MRR@{TOP_K} over {len(facet_rankings)} facets: {mrr:.4f}")

    # fuse with simple Reciprocal Rank Fusion (RRF) to produce a single top list
    #    RRF(d) = sum_q 1 / (k + rank_q(d)); use k=60 (standard)
    k_rrf = 60
    scores = defaultdict(float)
    for ranked in facet_rankings:
        for r, pid in enumerate(ranked, 1):
            scores[pid] += 1.0 / (k_rrf + r)

    fused = sorted(scores.items(), key=lambda t: t[1], reverse=True)
    top_ids = [pid for pid, _ in fused[:TOP_K]]

    return top_ids

if __name__ == "__main__":
    res = MRR("Introduce MathCheck, a checklist designed to test task generalization and reasoning robustness, along with an automatic tool for efficient checklist generation. MathCheck includes various mathematical reasoning tasks and robustness tests, and is used to develop MathCheck-GSM and MathCheck-GEO for evaluating mathematical textual and multi-modal reasoning capabilities, respectively, offering an improved assessment over existing benchmarks.")
    print(get_arxiv_id("nDvgHIBRxQ") in res)
