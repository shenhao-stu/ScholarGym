"""
Minimal standalone script to validate Qdrant server-side filters for
  - before_date (lexicographic / datetime range on metadata.date)
  - exclude_arxiv_ids (must_not MatchAny on metadata.arxiv_id)

Payload schema in the collection is nested:
    payload = {
        "page_content": str,
        "metadata": {
            "arxiv_id": "2212.13261",
            "date": "2022-12-25",
            ...
        }
    }
So filter keys must be the dotted form: "metadata.arxiv_id" / "metadata.date".

Usage:
    conda activate verl
    python scripts/test_qdrant_filter.py
"""

import os
import sys

os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
import config  # noqa: E402

from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.http import models as qm  # noqa: E402
from langchain_ollama import OllamaEmbeddings  # noqa: E402


QUERY = "graph neural networks for molecular property prediction"
BEFORE_DATE = "2020-01"  # YYYY-MM, same convention as rag.search_citations_vector
TOP_K = 20


def embed(text: str):
    emb = OllamaEmbeddings(
        model=config.QDRANT_EMBEDDING_MODEL,
        base_url=config.OLLAMA_URL,
    )
    return emb.embed_query(text)


def show(tag, points):
    print(f"\n=== {tag} (got {len(points)}) ===")
    for p in points[:10]:
        m = p.payload.get("metadata", {}) if p.payload else {}
        print(f"  {m.get('arxiv_id'):<12} {m.get('date'):<12} score={p.score:.4f}")


def main():
    client = QdrantClient(url=config.QDRANT_URL)
    vec = embed(QUERY)

    # 1) Baseline: no filter
    baseline = client.query_points(
        collection_name=config.QDRANT_COLLECTION_NAME,
        query=vec,
        limit=TOP_K,
        with_payload=True,
    ).points
    show("baseline (no filter)", baseline)

    baseline_ids = [p.payload["metadata"]["arxiv_id"] for p in baseline]

    # 2) exclude_arxiv_ids via must_not + MatchAny
    exclude = set(baseline_ids[:5])
    print(f"\n[exclude set] {sorted(exclude)}")
    excluded_flt = qm.Filter(
        must_not=[
            qm.FieldCondition(
                key="metadata.arxiv_id",
                match=qm.MatchAny(any=list(exclude)),
            )
        ]
    )
    r_excl = client.query_points(
        collection_name=config.QDRANT_COLLECTION_NAME,
        query=vec,
        limit=TOP_K,
        with_payload=True,
        query_filter=excluded_flt,
    ).points
    show("excluded via must_not MatchAny", r_excl)
    violators = [p.payload["metadata"]["arxiv_id"] for p in r_excl
                 if p.payload["metadata"]["arxiv_id"] in exclude]
    print(f"  -> violators (should be []): {violators}")

    # 3) before_date using DatetimeRange (ISO string field)
    #    This requires Qdrant to parse ISO format. dates in payload are 'YYYY-MM-DD'.
    try:
        date_flt_dt = qm.Filter(
            must=[
                qm.FieldCondition(
                    key="metadata.date",
                    range=qm.DatetimeRange(lt=f"{BEFORE_DATE}-01T00:00:00Z"),
                )
            ]
        )
        r_dt = client.query_points(
            collection_name=config.QDRANT_COLLECTION_NAME,
            query=vec,
            limit=TOP_K,
            with_payload=True,
            query_filter=date_flt_dt,
        ).points
        show(f"before_date {BEFORE_DATE} via DatetimeRange(lt=...)", r_dt)
        bad = [p.payload["metadata"]["arxiv_id"]
               for p in r_dt
               if p.payload["metadata"]["date"][:7] >= BEFORE_DATE]
        print(f"  -> violators (should be []): {bad}")
    except Exception as e:
        print(f"[DatetimeRange path failed] {type(e).__name__}: {e}")

    # 4) Combined: before_date AND exclude_arxiv_ids
    try:
        combo_flt = qm.Filter(
            must=[
                qm.FieldCondition(
                    key="metadata.date",
                    range=qm.DatetimeRange(lt=f"{BEFORE_DATE}-01T00:00:00Z"),
                )
            ],
            must_not=[
                qm.FieldCondition(
                    key="metadata.arxiv_id",
                    match=qm.MatchAny(any=list(exclude)),
                )
            ],
        )
        r_combo = client.query_points(
            collection_name=config.QDRANT_COLLECTION_NAME,
            query=vec,
            limit=TOP_K,
            with_payload=True,
            query_filter=combo_flt,
        ).points
        show("combined before_date + exclude", r_combo)
        bad = []
        for p in r_combo:
            m = p.payload["metadata"]
            if m["date"][:7] >= BEFORE_DATE or m["arxiv_id"] in exclude:
                bad.append(m["arxiv_id"])
        print(f"  -> violators (should be []): {bad}")
    except Exception as e:
        print(f"[combined path failed] {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
