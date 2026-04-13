"""CheckpointManager — resume-from-file support for eval runs."""
import json
import os
from typing import Dict, List, Set, Tuple

from logger import get_logger

logger = get_logger(__name__, log_file='./log/utils.log')


class CheckpointManager:
    """
    Manages checkpoint loading, saving, and statistics rebuilding for evaluation.
    """

    def __init__(self, checkpoint_file: str):
        """
        Args:
            checkpoint_file: Path to the checkpoint JSONL file
        """
        self.checkpoint_file = checkpoint_file
        self.processed_indices: Set[int] = set()
        self.cached_results: List[Dict] = []

    def load_checkpoint(self) -> Tuple[Set[int], List[Dict]]:
        """
        Load checkpoint from file and return processed indices and cached results.

        Returns:
            Tuple of (processed_indices, cached_results)
        """
        if not os.path.exists(self.checkpoint_file):
            logger.info(f"[📝] No checkpoint found, starting fresh")
            return set(), []

        logger.info(f"[📂] Loading checkpoint: {self.checkpoint_file}")
        processed_indices = set()
        cached_results = []

        with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, 1):
                if line.strip():
                    try:
                        result = json.loads(line.strip())
                    except json.JSONDecodeError:
                        logger.warning(f"[⚠️] Skipping corrupt checkpoint line {line_no}: {line[:80]!r}")
                        continue
                    idx = result.get('idx', -1)
                    if idx >= 0:
                        processed_indices.add(idx)
                        cached_results.append(result)

        self.processed_indices = processed_indices
        self.cached_results = cached_results
        logger.info(f"[✓] Loaded {len(processed_indices)} processed queries from checkpoint")

        return processed_indices, cached_results

    def rebuild_statistics(
        self,
        results: Dict,
        workflow: str,
        top_k_list: List[int] = None,
        max_iterations: int = 3
    ) -> Dict:
        """
        Rebuild statistics from cached results.

        Args:
            results: Results dictionary to update
            workflow: 'simple' or 'deep_research'
            top_k_list: Top-k values for simple workflow
            max_iterations: Max iterations for deep_research workflow

        Returns:
            Updated results dictionary
        """
        if not self.cached_results:
            return results

        logger.info(f"[📊] Rebuilding statistics from {len(self.cached_results)} cached results...")

        for query_result in self.cached_results:
            results['successful_queries'] += 1
            results['detailed_results'].append(query_result)

            if workflow == 'deep_research':
                self._rebuild_deep_research_stats(query_result, results, max_iterations)
            else:
                self._rebuild_simple_stats(query_result, results, top_k_list)

        return results

    def _rebuild_deep_research_stats(
        self,
        query_result: Dict,
        results: Dict,
        max_iterations: int
    ):
        """Rebuild statistics for deep_research workflow."""
        # Collect metrics by iteration — must match metric_names in eval.py evaluate_benchmark
        metric_names = ['recall', 'precision', 'f1', 'retrieval_recall', 'retrieval_precision', 'retrieval_f1', 'missed_gt_ratio']
        metrics_by_iter = {name: {} for name in metric_names}

        for res in query_result['iteration_results']:
            iter_idx = res['iter_idx']
            for metric_name in metric_names:
                if metric_name in res:
                    metrics_by_iter[metric_name][iter_idx] = res[metric_name]

        # Fill missing iterations with last value
        for metric_name, metric_dict in metrics_by_iter.items():
            if not metric_dict:
                continue

            max_it = max(metric_dict.keys())
            last_val = metric_dict[max_it]
            for it in range(max_it + 1, max_iterations + 1):
                metric_dict[it] = last_val

            # Add to results
            for it, val in metric_dict.items():
                key = f'{metric_name}_iter_{it}'
                if key not in results:
                    results[key] = []
                results[key].append(val)

        # Timing statistics
        # Keep consistent with eval.py so aggregated `avg_*` timing metrics are available
        # (e.g., browser_during for evaluation_summary.jsonl).
        for phase in ['planner', 'retrieval', 'selector', 'browser', 'overhead', 'total']:
            phase_key = f'{phase}_during'
            if phase_key not in results:
                results[phase_key] = []
            for res in query_result['iteration_results']:
                time_val = res.get(phase_key, -1)
                if time_val >= 0:
                    results[phase_key].append(time_val)

        # Distance statistics
        for res in query_result['iteration_results']:
            iter_idx = res['iter_idx']
            distance_key = f'avg_distance_iter_{iter_idx}'
            if distance_key not in results:
                results[distance_key] = []
            avg_distance = res.get('avg_distance', -1)
            if avg_distance >= 0:
                results[distance_key].append(avg_distance)

        # Discarded statistics
        for res in query_result['iteration_results']:
            iter_idx = res['iter_idx']
            iteration_metrics = res.get('iteration_metrics', {})

            ratio_key = f'discarded_ratio_iter_{iter_idx}'
            if ratio_key not in results:
                results[ratio_key] = []
            discarded_ratio = iteration_metrics.get('discarded_ratio', -1)
            if discarded_ratio >= 0:
                results[ratio_key].append(discarded_ratio)

            count_key = f'discarded_total_count_iter_{iter_idx}'
            if count_key not in results:
                results[count_key] = []
            discarded_count = iteration_metrics.get('discarded_total_count', -1)
            if discarded_count >= 0:
                results[count_key].append(discarded_count)

    def _rebuild_simple_stats(
        self,
        query_result: Dict,
        results: Dict,
        top_k_list: List[int]
    ):
        """Rebuild statistics for simple workflow."""
        gt_arxiv_ids = set(query_result['ground_truth_arxiv_ids'])
        retrieved_arxiv_ids = [res['arxiv_id'] for res in query_result['top_results']]

        for k in top_k_list:
            top_k_arxiv_ids = set(retrieved_arxiv_ids[:k])
            matches_k = len(gt_arxiv_ids.intersection(top_k_arxiv_ids))
            recall_k = matches_k / len(gt_arxiv_ids) if gt_arxiv_ids else 0.0
            precision_k = matches_k / len(top_k_arxiv_ids) if top_k_arxiv_ids else 0.0

            results[f'recall@{k}'].append(recall_k)
            results[f'precision@{k}'].append(precision_k)

    def append_result(self, result: Dict):
        """
        Append a single query result to checkpoint file.

        Args:
            result: Query result dictionary
        """
        os.makedirs(os.path.dirname(self.checkpoint_file), exist_ok=True)
        with open(self.checkpoint_file, 'a', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False)
            f.write('\n')

        # Update internal cache
        idx = result.get('idx', -1)
        if idx >= 0:
            self.processed_indices.add(idx)
            self.cached_results.append(result)

    def is_processed(self, idx: int) -> bool:
        """Check if a query index has been processed."""
        return idx in self.processed_indices
