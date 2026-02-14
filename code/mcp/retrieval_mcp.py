from typing import List, Dict, Optional, Set
from rag import CitationRAGSystem
import config


def _run_search(
    rag_system: CitationRAGSystem,
    query: str,
    top_k: int,
    search_method: str,
    before_date: Optional[str] = None,
    offset: int = 0,
    gt_arxiv_ids: Optional[Set[str]] = None,
    selected_paper_ids_tracker: Optional[Set[str]] = None
):
    """Internal helper to dispatch to the configured search method."""
    # Enforce per-call cap
    capped_top_k = min(top_k, getattr(config, 'MAX_RESULTS_PER_QUERY', top_k))
    if search_method == "vector":
        return rag_system.search_citations_vector(query, top_k=capped_top_k, offset=offset, before_date=before_date, gt_arxiv_ids=gt_arxiv_ids, exclude_arxiv_ids=selected_paper_ids_tracker)
    if search_method == "bm25":
        return rag_system.search_citations_bm25(query, top_k=capped_top_k, offset=offset, before_date=before_date, gt_arxiv_ids=gt_arxiv_ids, exclude_arxiv_ids=selected_paper_ids_tracker)
    # default to hybrid
    return rag_system.search_citations_hybrid(query, top_k=capped_top_k, before_date=before_date, offset=offset)


def search_papers(
    rag_system: CitationRAGSystem,
    query: str,
    top_k: int = 10,
    search_method: str = "hybrid",
    before_date: Optional[str] = None,
    offset: int = 0,
    gt_arxiv_ids: Optional[Set[str]] = None,
    selected_paper_ids_tracker: Optional[Set[str]] = None
) -> List[Dict]:
    """
    Search for papers using the RAG system.

    Args:
        rag_system: An instance of the CitationRAGSystem.
        query: The search query.
        top_k: The number of results to return.
        search_method: 'vector', 'bm25', or 'hybrid'.
        before_date: Optional date to filter results.
        offset: Optional results offset for pagination.

    Returns:
        A list of paper results with metadata and scores.
    """
    results,rank_dict = _run_search(
        rag_system=rag_system,
        query=query,
        top_k=top_k,
        search_method=search_method,
        before_date=before_date,
        offset=offset,
        gt_arxiv_ids=gt_arxiv_ids,
        selected_paper_ids_tracker=selected_paper_ids_tracker
    )
    return [
        {
            "paper_id": paper_id,
            "title": info.get("title", "N/A"),
            "score": score,
            "abstract": info.get("abstract", "N/A"),
            "arxiv_id": info.get("arxiv_id", "N/A"),
            "date": info.get("date", "N/A")
        }
        for paper_id, score, info in results
    ],rank_dict


def batch_search_papers(
    rag_system: CitationRAGSystem,
    queries: List[str],
    top_k: int = 10,
    search_method: str = "hybrid",
    before_date: Optional[str] = None,
    offset: int = 0,
) -> List[List[Dict]]:
    """
    Batch search variant for efficiency.
    """
    outputs: List[List[Dict]] = []
    for q in queries:
        outputs.append(
            search_papers(
                rag_system=rag_system,
                query=q,
                top_k=top_k,
                search_method=search_method,
                before_date=before_date,
                offset=offset,
            )
        )
    return outputs
