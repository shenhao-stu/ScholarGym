#!/usr/bin/env python3
"""
Metrics and Performance Utilities for Deep Research Workflow.
Contains Timer for performance tracking and MetricsCalculator for evaluation.
"""
import time
from typing import Dict, Set
from structures import SubQuery
import config

class Timer:
    """Context manager for measuring code block execution time."""
    
    def __enter__(self):
        self._start = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc, tb):
        self.elapsed = time.perf_counter() - self._start


class MetricsCalculator:
    """Calculate precision and recall metrics for retrieval and selection."""
    
    @staticmethod
    def calculate_subquery_metrics(
        retrieved_arxiv_ids: Set[str],
        selected_arxiv_ids: Set[str],
        gt_arxiv_ids: Set[str]
    ) -> Dict:
        """
        Calculate precision and recall for a single subquery.
        
        Args:
            retrieved_arxiv_ids: ArXiv IDs from retrieval
            selected_arxiv_ids: ArXiv IDs after Selector filtering
            gt_arxiv_ids: Ground truth ArXiv IDs
            
        Returns:
            Dict with retrieval metrics, selection metrics, and discarded GT count
        """
        # Retrieval metrics (before Selector)
        retrieved_matched = retrieved_arxiv_ids.intersection(gt_arxiv_ids)
        retrieval_precision = (
            len(retrieved_matched) / len(retrieved_arxiv_ids)
            if retrieved_arxiv_ids else 0.0
        )
        retrieval_recall = (
            len(retrieved_matched) / len(gt_arxiv_ids)
            if gt_arxiv_ids else 0.0
        )
        
        # Selection metrics (after Selector)
        selected_matched = selected_arxiv_ids.intersection(gt_arxiv_ids)
        selection_precision = (
            len(selected_matched) / len(selected_arxiv_ids)
            if selected_arxiv_ids else 0.0
        )
        selection_recall = (
            len(selected_matched) / len(gt_arxiv_ids)
            if gt_arxiv_ids else 0.0
        )
        
        # Ground truth papers discarded by Selector
        discarded_gt = retrieved_matched - selected_matched
        
        return {
            "retrieval": {
                "precision": retrieval_precision,
                "recall": retrieval_recall,
                "matched": len(retrieved_matched),
                "total": len(retrieved_arxiv_ids)
            },
            "selection": {
                "precision": selection_precision,
                "recall": selection_recall,
                "matched": len(selected_matched),
                "total": len(selected_arxiv_ids)
            },
            "discarded_gt_count": len(discarded_gt),
            "discarded_gt_ids": list(discarded_gt)
        }
    
    @staticmethod
    def calculate_iteration_metrics(
        subquery_results: Dict[int, Dict],
        gt_arxiv_ids: Set[str]
    ) -> Dict:
        """
        Calculate aggregated precision and recall for an entire iteration.
        
        Args:
            subquery_results: Dict mapping subquery_id to {retrieved_arxiv_ids, selected_arxiv_ids}
            gt_arxiv_ids: Ground truth ArXiv IDs
            
        Returns:
            Dict with iteration-level retrieval and selection metrics
        """
        # Collect all unique papers across all subqueries (deduplicated)
        all_retrieved = set()
        all_selected = set()
        
        for sq_id, result in subquery_results.items():
            all_retrieved.update(result.get('retrieved_arxiv_ids', set()))
            all_selected.update(result.get('selected_arxiv_ids', set()))
        
        # Calculate metrics
        retrieved_matched = all_retrieved.intersection(gt_arxiv_ids)
        selected_matched = all_selected.intersection(gt_arxiv_ids)
        
        retrieval_precision = (
            len(retrieved_matched) / len(all_retrieved)
            if all_retrieved else 0.0
        )
        retrieval_recall = (
            len(retrieved_matched) / len(gt_arxiv_ids)
            if gt_arxiv_ids else 0.0
        )
        
        selection_precision = (
            len(selected_matched) / len(all_selected)
            if all_selected else 0.0
        )
        selection_recall = (
            len(selected_matched) / len(gt_arxiv_ids)
            if gt_arxiv_ids else 0.0
        )
        
        # Total ground truth papers discarded by Selector in this iteration
        total_discarded_gt = retrieved_matched - selected_matched
        
        # Total discarded papers (all retrieved but not selected)
        total_discarded = all_retrieved - all_selected
        
        # Discarded ratio: discarded_count / selected_count
        discarded_ratio = (
            len(total_discarded_gt) / len(total_discarded)
            if total_discarded else 0.0
        )
        
        return {
            "retrieval": {
                "precision": retrieval_precision,
                "recall": retrieval_recall,
                "matched": len(retrieved_matched),
                "total": len(all_retrieved)
            },
            "selection": {
                "precision": selection_precision,
                "recall": selection_recall,
                "matched": len(selected_matched),
                "total": len(all_selected)
            },
            "discarded_gt_count": len(total_discarded_gt),
            "discarded_gt_ids": list(total_discarded_gt),
            "discarded_total_count": len(total_discarded),
            "discarded_ratio": discarded_ratio
        }
    
    @staticmethod
    def calculate_gt_rank_and_distance(
        subqueries: Dict[int, SubQuery],
        rank_dicts: Dict[int, Dict[str, Dict]],
        gt_arxiv_ids: Set[str],
        selected_paper_ids_tracker: Set[str],
        selected_min_rank_tracker: Dict[str, int],
        gt_rank_cutoff: int
    ) -> Dict:
        """
        Calculate ground truth ranking and distance metrics for the current iteration.
        
        This function computes:
        1. Current iteration's minimum rank for each GT paper across all subqueries
        2. Adjusted minimum rank considering historical best ranks for selected papers
        3. Distance metric based on rank positions
        4. Average distance for iteration quality assessment
        
        Args:
            subqueries: Dict mapping subquery_id to SubQuery object
            rank_dicts: Dict mapping subquery_id to {arxiv_id: {rank, total}}
            gt_arxiv_ids: Set of ground truth ArXiv IDs
            selected_paper_ids_tracker: Set of paper IDs selected across iterations
            selected_min_rank_tracker: Dict tracking historical minimum ranks for selected papers
            gt_rank_cutoff: Cutoff value for distance calculation
            
        Returns:
            Dict containing:
                - gt_rank: List of rank details per subquery
                - cur_iter_distances: Dict of distance scores per GT paper
                - avg_distance: Average distance across all GT papers (-1 if invalid)
                - updated_selected_min_rank_tracker: Updated historical minimum ranks
        """
        cur_iter_min_rank = {}  # Current iteration's minimum rank per GT paper
        gt_rank_details = []
        
        # Step 1: Calculate minimum rank within current iteration across all subqueries
        for sq_id, sq in subqueries.items():
            ranks = []
            rank_dict = rank_dicts.get(sq_id, {})
            for arxiv_id in (gt_arxiv_ids or []):
                rank_info = rank_dict.get(arxiv_id, None)
                # Only to consider the case where rank_info is None, in which case it is treated as not found, and each value is valid
                if rank_info is None:
                    # Invalid rank data
                    ranks.append({
                        "arxiv_id": arxiv_id,
                        "rank": config.TOTAL_PAPER_NUM,
                        "total_rank": -1
                    })
                    
                    # Track minimum rank within current iteration
                    if arxiv_id not in cur_iter_min_rank:
                        cur_iter_min_rank[arxiv_id] = config.TOTAL_PAPER_NUM
                    else:
                        cur_iter_min_rank[arxiv_id] = min(
                            cur_iter_min_rank[arxiv_id], 
                            config.TOTAL_PAPER_NUM
                        )
                        
                else:
                    ranks.append({
                        "arxiv_id": arxiv_id,
                        "rank": rank_info["rank"],
                        "total_rank": rank_info["total"]
                    })
                    
                    # Track minimum rank within current iteration
                    if arxiv_id not in cur_iter_min_rank:
                        cur_iter_min_rank[arxiv_id] = rank_info["rank"]
                    else:
                        cur_iter_min_rank[arxiv_id] = min(
                            cur_iter_min_rank[arxiv_id], 
                            rank_info["rank"]
                        )
            
            gt_rank_details.append({
                "sq_id": sq_id,
                "sub_query": sq.text,
                "ranks": ranks
            })
        
        # Step 2: Apply historical best ranks for selected papers
        adjusted_min_rank = cur_iter_min_rank.copy()
        for arxiv_id in selected_paper_ids_tracker:
            if arxiv_id in (gt_arxiv_ids or []) and arxiv_id in selected_min_rank_tracker:
                if arxiv_id in adjusted_min_rank:
                    # Use the better of current rank or historical rank
                    adjusted_min_rank[arxiv_id] = min(
                        adjusted_min_rank[arxiv_id],
                        selected_min_rank_tracker[arxiv_id]
                    )
                else:
                    # Paper not found in current iteration, use historical rank
                    adjusted_min_rank[arxiv_id] = selected_min_rank_tracker[arxiv_id]
        
        # Step 3: Update selected_min_rank_tracker for next iteration
        updated_selected_min_rank_tracker = selected_min_rank_tracker.copy()
        for arxiv_id in (gt_arxiv_ids or []):
            if arxiv_id in adjusted_min_rank:
                updated_selected_min_rank_tracker[arxiv_id] = min(
                    updated_selected_min_rank_tracker.get(arxiv_id, float('inf')),
                    adjusted_min_rank[arxiv_id]
                )
        
        # Step 4: Calculate distance metrics based on adjusted ranks
        cur_iter_distances = {
            arxiv_id: max(1 - (rank / gt_rank_cutoff), 0)
            for arxiv_id, rank in adjusted_min_rank.items()
        }
        
        # Step 5: Calculate average distance
        avg_distance = sum(cur_iter_distances.values()) / len(cur_iter_distances)
        
        return {
            "gt_rank": gt_rank_details,
            "cur_iter_distances": cur_iter_distances,
            "avg_distance": avg_distance,
            "updated_selected_min_rank_tracker": updated_selected_min_rank_tracker
        }
    
    @staticmethod
    def calculate_simple_avg_distance(
        rank_dicts: list,
        gt_arxiv_ids: Set[str],
        gt_rank_cutoff: int
    ) -> float:
        """
        Calculate average distance metric for simple workflow.
        
        Args:
            rank_dicts: List of rank dictionaries from multiple queries
            gt_arxiv_ids: Set of ground truth ArXiv IDs
            gt_rank_cutoff: Cutoff rank for distance calculation
            
        Returns:
            float: Average distance metric (0 to 1, higher is better)
        """
        if not rank_dicts or not gt_arxiv_ids:
            return 0.0
        
        # Aggregate minimum ranks across all queries for each ground truth paper
        min_ranks = {}
        for rank_dict in rank_dicts:
            for arxiv_id, rank_info in rank_dict.items():
                rank = rank_info.get('rank', gt_rank_cutoff)
                if arxiv_id not in min_ranks or rank < min_ranks[arxiv_id]:
                    min_ranks[arxiv_id] = rank
        
        # Calculate distance for each ground truth paper using the formula: max(1 - rank/cutoff, 0)
        distances = [
            max(1 - (min_ranks.get(arxiv_id, gt_rank_cutoff) / gt_rank_cutoff), 0)
            for arxiv_id in gt_arxiv_ids
        ]
        
        # Return average distance
        return sum(distances) / len(distances)