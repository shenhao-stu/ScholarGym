"""Ground-truth extraction, result combination, and retrieval metric calculation."""
from typing import Dict, List, Set


def extract_ground_truth_arxiv_ids(ground_truth_papers: List[Dict], gt_labels: List[int]) -> set:
    """
    Extract ground truth paper arxiv_ids.

    Args:
        ground_truth_papers (List[Dict]): List of ground truth papers
        gt_labels (List[int]): List of labels (1 for relevant, 0 for not)

    Returns:
        set: Set of ground truth arXiv IDs
    """
    return {
        paper['arxiv_id']
        for paper, label in zip(ground_truth_papers, gt_labels)
        if label == 1 and 'arxiv_id' in paper and paper['arxiv_id']
    }


def combine_search_results(all_search_results: List[List[tuple]]) -> List[tuple]:
    """
    Combine and deduplicate search results from multiple queries.

    Args:
        all_search_results (List[List[tuple]]): List of search results, each is a list of (paper_id, similarity, paper_info)

    Returns:
        List[tuple]: Combined and sorted list of (paper_id, similarity, paper_info)
    """
    paper_scores = {}
    paper_info_map = {}

    for search_results in all_search_results:
        for paper_id, similarity, paper_info in search_results:
            if paper_id not in paper_scores or similarity > paper_scores[paper_id]:
                paper_scores[paper_id] = similarity
                paper_info_map[paper_id] = paper_info

    sorted_papers = sorted(paper_scores.items(), key=lambda x: x[1], reverse=True)

    return [
        (paper_id, paper_scores[paper_id], paper_info_map[paper_id])
        for paper_id, _ in sorted_papers
    ]


def calculate_retrieval_metrics(
    gt_arxiv_ids: Set[str],
    retrieved_arxiv_ids: Set[str],
    selected_arxiv_ids: Set[str] = None
) -> Dict[str, float]:
    """
    Calculate recall and precision metrics for retrieval and selection.

    Args:
        gt_arxiv_ids: Set of ground truth arXiv IDs
        retrieved_arxiv_ids: Set of retrieved arXiv IDs (before selection)
        selected_arxiv_ids: Set of selected arXiv IDs (after selection, optional)

    Returns:
        Dict containing recall and precision metrics
    """
    metrics = {}

    # Retrieval metrics
    retrieved_matches = len(gt_arxiv_ids.intersection(retrieved_arxiv_ids))
    metrics['retrieval_recall'] = retrieved_matches / len(gt_arxiv_ids) if gt_arxiv_ids else 0.0
    metrics['retrieval_precision'] = retrieved_matches / len(retrieved_arxiv_ids) if retrieved_arxiv_ids else 0.0
    metrics['retrieval_f1'] = (
        2 * metrics['retrieval_recall'] * metrics['retrieval_precision'] / (metrics['retrieval_recall'] + metrics['retrieval_precision'])
        if (metrics['retrieval_recall'] + metrics['retrieval_precision']) > 0 else 0.0
    )
    metrics['retrieval_matches'] = retrieved_matches
    metrics['retrieved_count'] = len(retrieved_arxiv_ids)

    # Selection metrics (if provided)
    if selected_arxiv_ids is not None:
        selected_matches = len(gt_arxiv_ids.intersection(selected_arxiv_ids))
        metrics['recall'] = selected_matches / len(gt_arxiv_ids) if gt_arxiv_ids else 0.0
        metrics['precision'] = selected_matches / len(selected_arxiv_ids) if selected_arxiv_ids else 0.0
        metrics['f1'] = (
            2 * metrics['recall'] * metrics['precision'] / (metrics['recall'] + metrics['precision'])
            if (metrics['recall'] + metrics['precision']) > 0 else 0.0
        )
        metrics['matches'] = selected_matches
        metrics['selected_count'] = len(selected_arxiv_ids)

    return metrics
