#!/usr/bin/env python3
import os
import sys
import json
import shutil
import importlib.util
from typing import List, Dict
from tqdm import tqdm
import numpy as np
import datetime

from logger import get_logger
from rag import CitationRAGSystem
import config
from deeprag import DeepResearchWorkflow
from simplerag import SimpleWorkflow
from utils import extract_ground_truth_arxiv_ids, CheckpointManager, calculate_retrieval_metrics, AgentTraceRecorder

logger = get_logger(__name__, log_file='./log/eval.log')

def load_config_from_path(config_path: str):
    """
    Dynamically load config from a file path.
    
    Args:
        config_path: Path to config.py file
        
    Returns:
        Loaded config module
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # First, ensure the base config module is loaded so that
    # custom configs can do `from config import *` to inherit defaults.
    base_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.py')
    if os.path.abspath(config_path) != os.path.abspath(base_config_path):
        # Load the base config first under "config" so the custom file can import from it
        if "config" not in sys.modules:
            base_spec = importlib.util.spec_from_file_location("config", base_config_path)
            if base_spec and base_spec.loader:
                base_module = importlib.util.module_from_spec(base_spec)
                sys.modules["config"] = base_module
                base_spec.loader.exec_module(base_module)

    # Now load the custom config under a temporary name
    spec = importlib.util.spec_from_file_location("config_custom", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load config from: {config_path}")

    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    # Copy all public attributes from the custom config INTO the existing
    # config module object.  This is critical: by the time main() calls us,
    # every downstream module (deeprag, agent/*, simplerag, …) has already
    # executed `import config` and holds a reference to the *original* module
    # object.  Replacing sys.modules["config"] would leave those references
    # stale.  Updating in-place ensures every module sees the new values.
    existing_config = sys.modules["config"]
    for attr in dir(config_module):
        if not attr.startswith('_'):
            setattr(existing_config, attr, getattr(config_module, attr))

    logger.info(f"[📝] Loaded config from: {config_path}")
    return existing_config

class CitationEvaluator:
    def __init__(self, rag_system: CitationRAGSystem, trace_recorder=None):
        self.rag_system = rag_system
        self.llm_model = config.LLM_MODEL_NAME
        self.is_local = config.IS_LOCAL_LLM
        self.prompt_type = config.EVAL_PROMPT_TYPE
        self.search_method = config.EVAL_SEARCH_METHOD
        self.gen_params = config.LLM_GEN_PARAMS
        self.trace_recorder = trace_recorder

        # Validate search method
        available_methods = self.rag_system.get_available_search_methods()
        if self.search_method not in available_methods:
            raise ValueError(f"Search method '{self.search_method}' not available. Available methods: {available_methods}")

        # Initialize workflows
        self.simple_workflow = SimpleWorkflow(
            rag_system=self.rag_system,
            trace_recorder=trace_recorder
        )

        self.deep_research_workflow = DeepResearchWorkflow(
            rag_system=self.rag_system,
            trace_recorder=trace_recorder
        )

    def load_benchmark_data(self, benchmark_jsonl_path: str) -> List[Dict]:
        """Load benchmark data from JSONL file."""
        benchmark_data = []
        with open(benchmark_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                benchmark_data.append(json.loads(line.strip()))
        
        logger.info(f"[📂]Loaded {len(benchmark_data)} benchmark queries")
        return benchmark_data

    def evaluate_single_query_deep_research(
        self,
        query_data: Dict,
        idx: int = 1
    ) -> Dict:
        """
        Evaluate a single benchmark query using the Deep Research workflow.

        Args:
            query_data: Query data dictionary
            idx: Query index for logging
        """
        query = query_data['query']
        gt_labels = query_data['gt_label']
        gt_arxiv_ids = extract_ground_truth_arxiv_ids(query_data['cited_paper'], gt_labels)

        if not gt_arxiv_ids:
            return None

        workflow_results = self.deep_research_workflow.run(
            query_data,
            gt_arxiv_ids=gt_arxiv_ids,
            idx=idx,
        )
        
        # Handle workflow failure
        if workflow_results is None:
            logger.warning(f"[❌] Query {idx} failed - workflow returned None")
            return None
        
        iteration_results = []
        # Use 'select' stage to compute per-iteration metrics
        select_steps = [item for item in workflow_results['history'] if item.get('stage') == 'select']

        # Track cumulative retrieval and selection across iterations
        all_retrieved_arxiv_ids = set()
        all_selected_arxiv_ids = set()
        total_discarded_gt_count = 0  # Cumulative count of discarded ground truth papers

        for step in select_steps:
            iter_idx = step.get('iter_idx')
            retrieved_in_iter = step.get('retrieved_papers') or {}
            selected_in_iter = step.get('selected_papers') or {}
            gt_rank = step.get('gt_rank') or []
            browsing_arxiv_ids = step.get('browsing_arxiv_ids') or []
            avg_distance = step.get('avg_distance', 0)
            iteration_metrics = step.get('iteration_metrics') or {}
            subquery_metrics = step.get('subquery_metrics') or {}
            
            if not isinstance(retrieved_in_iter, dict):
                retrieved_in_iter = {}
            if not isinstance(selected_in_iter, dict):
                selected_in_iter = {}

            # Collect current iteration ArXiv IDs
            current_iter_retrieved = {
                p.get('arxiv_id')
                for papers in retrieved_in_iter.values()
                for p in papers if p.get('arxiv_id')
            }
            current_iter_selected = {
                p.get('arxiv_id')
                for papers in selected_in_iter.values()
                for p in papers if p.get('arxiv_id')
            }

            # Update cumulative sets
            all_retrieved_arxiv_ids.update(current_iter_retrieved)
            all_selected_arxiv_ids.update(current_iter_selected)

            # Calculate cumulative metrics using unified function
            metrics = calculate_retrieval_metrics(
                gt_arxiv_ids, 
                all_retrieved_arxiv_ids, 
                all_selected_arxiv_ids
            )
            
            # Count discarded ground truth in this iteration
            iter_discarded_gt = iteration_metrics.get('discarded_gt_count', 0)
            total_discarded_gt_count += iter_discarded_gt

            # Calculate retrieved but not selected GTs in this iteration
            iter_retrieved_not_selected_gt = (gt_arxiv_ids & current_iter_retrieved) - current_iter_selected
            
            missed_gt_ratio = len(iter_retrieved_not_selected_gt) / len(current_iter_retrieved) if current_iter_retrieved else 0.0

            iteration_results.append({
                "iter_idx": iter_idx,
                "iter_retrieved_not_selected_gt": list(iter_retrieved_not_selected_gt),
                "missed_gt_ratio": missed_gt_ratio,
                # Selection metrics (after selector filtering)
                "recall": metrics['recall'],
                "precision": metrics['precision'],
                "f1": metrics.get('f1', 0.0),
                "matches": metrics['matches'],
                "selected_count": metrics['selected_count'],
                # Retrieval metrics (before selector filtering)
                "retrieval_recall": metrics['retrieval_recall'],
                "retrieval_precision": metrics['retrieval_precision'],
                "retrieval_f1": metrics.get('retrieval_f1', 0.0),
                "retrieval_matches": metrics['retrieval_matches'],
                "retrieved_count": metrics['retrieved_count'],
                # Detailed metrics
                "iteration_metrics": iteration_metrics,
                "subquery_metrics": subquery_metrics,
                "iter_discarded_gt_count": iter_discarded_gt,
                "total_discarded_gt_count": total_discarded_gt_count,
                "gt_rank": gt_rank,
                'avg_distance': avg_distance,
                "total_gt": len(gt_arxiv_ids),
                "planner_during": step.get('planner_during', -1),
                "retrieval_during": step.get('retrieval_during', -1),
                "selector_during": step.get('selector_during', -1),
                "browser_during": step.get('browser_during', -1),
                "overhead_during": step.get('overhead_during', -1),
                "total_during": step.get('total_during', -1),
                "current_iter_retrieved": list(current_iter_retrieved),
                "current_iter_selected": list(current_iter_selected),
                "current_iter_browsing": browsing_arxiv_ids,
            })
            
            # Log cumulative metrics
            logger.info(
                f"[📊 Cumulative Metrics after Iter {iter_idx}] "
                f"Retrieved: {metrics['retrieved_count']} papers, "
                f"Retrieval Recall: {metrics['retrieval_recall']:.4f}, "
                f"Retrieval Precision: {metrics['retrieval_precision']:.4f}; "
                f"Selected: {metrics['selected_count']} papers, "
                f"Selection Recall: {metrics['recall']:.4f}, "
                f"Selection Precision: {metrics['precision']:.4f}; "
                f"Total Discarded GT: {total_discarded_gt_count}"
            )

        final_selected_papers = [
            {"title": p.title, "arxiv_id": p.arxiv_id} 
            for p in workflow_results.get('selected_papers', [])
        ]

        return {
            'idx': idx,
            'query': query,
            'ground_truth_arxiv_ids': list(gt_arxiv_ids),
            'iteration_results': iteration_results,
            'final_report': workflow_results.get('final_report', ''),
            'final_selected_papers': final_selected_papers,
            'executed_queries': workflow_results.get('executed_queries', [])
        }

    def evaluate_benchmark(
        self,
        benchmark_data: List[Dict],
        workflow: str = 'simple',
        top_k_list: List[int] = None,
        detailed_results_file: str = None,
        enable_resume: bool = True
    ) -> Dict:
        """
        Evaluate the entire benchmark dataset.

        Args:
            benchmark_data: List of benchmark queries
            workflow: 'simple' or 'deep_research'
            top_k_list: Top-k values for simple workflow (ignored for deep_research)
            detailed_results_file: Path to save detailed results incrementally (JSONL format)
            enable_resume: Enable resume from checkpoint
        """
        max_iterations = config.EVAL_MAX_ITERATIONS
        results_per_query = config.MAX_RESULTS_PER_QUERY
        logger.info(f"Evaluating {len(benchmark_data)} queries with results_per_query={results_per_query} (using top_k={top_k_list} for simple workflow), max_iterations={max_iterations}")
        logger.info(f"Using prompt type: {self.prompt_type}\nUsing search method: {self.search_method}\nUsing workflow: {workflow}")

        # Initialize checkpoint manager for resume support
        checkpoint_manager = None
        if enable_resume and detailed_results_file:
            checkpoint_manager = CheckpointManager(detailed_results_file)
            checkpoint_manager.load_checkpoint()

        results = {
            'total_queries': len(benchmark_data),
            'successful_queries': 0,
            'prompt_type': self.prompt_type,
            'search_method': self.search_method,
            'workflow': workflow,
            'detailed_results': []
        }
        
        # Initialize workflow-specific result containers
        if workflow == 'simple':
            for k in top_k_list:
                results[f'recall@{k}'] = []
                results[f'precision@{k}'] = []
        
        # Rebuild statistics from checkpoint if exists
        if checkpoint_manager and checkpoint_manager.cached_results:
            checkpoint_manager.rebuild_statistics(
                results=results,
                workflow=workflow,
                top_k_list=top_k_list,
                max_iterations=max_iterations
            )
        
        for idx, query_data in enumerate(tqdm(benchmark_data, desc="Evaluating queries")):
            # Skip already processed queries
            if checkpoint_manager and checkpoint_manager.is_processed(idx):
                logger.info(f"[⏭️] Skipping already processed query {idx}")
                continue
                
            try:
                if workflow == 'deep_research':
                    query_result = self.evaluate_single_query_deep_research(
                        query_data,
                        idx=idx,
                    )
                    if query_result:
                        results['successful_queries'] += 1
                        results['detailed_results'].append(query_result)
                        
                        # Write result to checkpoint file immediately
                        if checkpoint_manager:
                            checkpoint_manager.append_result(query_result)
                        
                        # Collect metrics by iteration (unified approach)
                        metric_names = ['recall', 'precision', 'f1', 'retrieval_recall', 'retrieval_precision', 'retrieval_f1', 'missed_gt_ratio']
                        metrics_by_iter = {name: {} for name in metric_names}
                        
                        for res in query_result['iteration_results']:
                            it = res['iter_idx']
                            for metric_name in metric_names:
                                if metric_name in res:
                                    metrics_by_iter[metric_name][it] = res[metric_name]
                        
                        # Fill missing iterations with last value and accumulate to results
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
                            
                        # 耗时统计
                        # Note: evaluation_summary.jsonl only records aggregated `avg_*` metrics.
                        # So we must accumulate all *_during fields we care about here.
                        for phase in ['planner', 'retrieval', 'selector', 'browser', 'overhead', 'total']:
                            phase_key = f'{phase}_during'
                            if phase_key not in results:
                                results[phase_key] = []
                            for res in query_result['iteration_results']:
                                time_val = res.get(phase_key, -1)
                                if time_val >= 0:
                                    results[phase_key].append(time_val)
                        
                        # 累积各轮次的 avg_distance 列表，区分轮次
                        for res in query_result['iteration_results']:
                            iter_idx = res['iter_idx']
                            distance_key = f'avg_distance_iter_{iter_idx}'
                            if distance_key not in results:
                                results[distance_key] = []
                            avg_distance = res.get('avg_distance', -1)
                            if avg_distance >= 0:
                                results[distance_key].append(avg_distance)
                        
                        # 累积各轮次的 discarded_ratio 列表，区分轮次
                        for res in query_result['iteration_results']:
                            iter_idx = res['iter_idx']
                            iteration_metrics = res.get('iteration_metrics', {})
                            
                            # Discarded ratio
                            ratio_key = f'discarded_ratio_iter_{iter_idx}'
                            if ratio_key not in results:
                                results[ratio_key] = []
                            discarded_ratio = iteration_metrics.get('discarded_ratio', -1)
                            if discarded_ratio >= 0:
                                results[ratio_key].append(discarded_ratio)
                            
                            # Discarded total count
                            count_key = f'discarded_total_count_iter_{iter_idx}'
                            if count_key not in results:
                                results[count_key] = []
                            discarded_count = iteration_metrics.get('discarded_total_count', -1)
                            if discarded_count >= 0:
                                results[count_key].append(discarded_count)

                else: # simple workflow
                    query_result = self.simple_workflow.run(
                        query_data, 
                        top_k=max(top_k_list), 
                        search_method=self.search_method,
                        idx=idx
                    )
                    if query_result:
                        results['successful_queries'] += 1
                        results['detailed_results'].append(query_result)
                        
                        # Write result to checkpoint file immediately
                        if checkpoint_manager:
                            checkpoint_manager.append_result(query_result)
                        
                        gt_arxiv_ids = set(query_result['ground_truth_arxiv_ids'])
                        retrieved_arxiv_ids = [res['arxiv_id'] for res in query_result['top_results']]
                        
                        for k in top_k_list:
                            top_k_arxiv_ids = set(retrieved_arxiv_ids[:k])
                            matches_k = len(gt_arxiv_ids.intersection(top_k_arxiv_ids))
                            recall_k = matches_k / len(gt_arxiv_ids) if gt_arxiv_ids else 0.0
                            precision_k = matches_k / len(top_k_arxiv_ids) if top_k_arxiv_ids else 0.0
                            
                            results[f'recall@{k}'].append(recall_k)
                            results[f'precision@{k}'].append(precision_k)
                        
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                logger.warning(f"[💢]Failed to evaluate query {query_data.get('query', 'N/A')[:30]}: {type(e).__name__}: {e}")
                continue
            
        total_queries = results['total_queries']

        if workflow == 'deep_research':
            # Average all per-iteration and per-phase metric lists
            # Divide by total_queries (not just successful), treating failed queries as 0
            avg_prefixes = (
                'recall_iter_', 'precision_iter_', 'f1_iter_',
                'retrieval_recall_iter_', 'retrieval_precision_iter_', 'retrieval_f1_iter_',
                'missed_gt_ratio_iter_', 'avg_distance_iter_',
                'discarded_ratio_iter_', 'discarded_total_count_iter_',
            )
            for key in list(results.keys()):
                if any(key.startswith(p) for p in avg_prefixes) or key.endswith('_during'):
                    if isinstance(results[key], list):
                        results[f'avg_{key}'] = sum(results[key]) / total_queries if total_queries > 0 else 0.0

            # 计算各轮次平均 missed_gt_ratio 在轮次上的平均
            missed_gt_ratio_avgs = [v for k, v in results.items() if k.startswith('avg_missed_gt_ratio_iter_')]
            if missed_gt_ratio_avgs:
                results['avg_missed_gt_ratio_macro_avg'] = np.mean(missed_gt_ratio_avgs)

        else:
            for k in top_k_list:
                results[f'avg_recall@{k}'] = sum(results[f'recall@{k}']) / total_queries if total_queries > 0 else 0.0
                results[f'avg_precision@{k}'] = sum(results[f'precision@{k}']) / total_queries if total_queries > 0 else 0.0
        
        return results

    def save_results(self, results: Dict, output_dir: str):
        """Save evaluation results to a JSON file with a descriptive name."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (
            f"eval_results_{results['workflow']}_{results['search_method']}_"
            f"{results['prompt_type']}_{timestamp}.json"
        )
        output_path = os.path.join(output_dir, filename)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        
        logger.info(f"[💾]Results saved to {output_path}")

    def save_detailed_results(self, detailed_results: List[Dict], output_file: str):
        """Save detailed per-query evaluation results to a JSONL file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in detailed_results:
                json.dump(result, f, ensure_ascii=False)
                f.write('\n')
        logger.info(f"[📊]Detailed results saved to {output_file}")

    def append_summary_record(self, results: Dict, output_file: str, detailed_results_file: str):
        """Append a summary of the evaluation record to a file in JSONL format."""
        record = {
            "model_name": self.llm_model,
            "prompt_type": results.get('prompt_type'),
            "search_method": results.get('search_method'),
            "workflow": results.get('workflow'),
            "enable_reasoning": config.ENABLE_REASONING,
            "enable_structured_output": config.ENABLE_STRUCTURED_OUTPUT,
            "EVAL_TOP_K_VALUES": config.EVAL_TOP_K_VALUES,
            "MAX_RESULTS_PER_QUERY": config.MAX_RESULTS_PER_QUERY,
            "EVAL_MAX_ITERATIONS": config.EVAL_MAX_ITERATIONS,
            "EVAL_DETAILED_RESULTS_PATH": detailed_results_file,
            "GT_RANK_CUTOFF": config.GT_RANK_CUTOFF,
            "BROWSER_MODE": config.BROWSER_MODE,
            "PLANNER_ABLATION": config.PLANNER_ABLATION,
        }
        
        # Flatten the avg_recalls dictionary and clean up the keys
        avg_metrics = {k: v for k, v in results.items() if k.startswith('avg_')}
        cleaned_metrics = {k.replace('avg_', ''): v for k, v in avg_metrics.items()}
        record.update(cleaned_metrics)
        
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')
        
        logger.info(f"Evaluation summary record appended to {output_file}")

    def print_summary(self, results: Dict):
        """Print evaluation results summary."""
        logger.info("=" * 60)
        logger.info("📊 CITATION EVALUATION RESULTS 📊")
        logger.info("=" * 60)
        logger.info(f"Total queries: {results['total_queries']}")
        logger.info(f"Successfully evaluated: {results['successful_queries']}")
        logger.info(f"Prompt type: {results.get('prompt_type', 'N/A')}")
        logger.info(f"Search method: {results.get('search_method', 'N/A')}")
        logger.info("")
        
        for key in sorted(results.keys()):
            if key.startswith('avg_recall@'):
                k_value = key.split('@')[1]
                recall = results[key]
                precision = results.get(f'avg_precision@{k_value}', 0.0)
                logger.info(f"Average Recall@{k_value}: {recall:.4f}, Precision@{k_value}: {precision:.4f}")
        
        logger.info("=" * 60)

def main():
    """Main evaluation pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Citation RAG Evaluation System')
    parser.add_argument('--config', type=str, default=None, help='Path to config.py file (if specified, overrides default config)')
    parser.add_argument('--paper_db', type=str, default=None, help='Path to paper database JSON file')
    parser.add_argument('--benchmark_jsonl', type=str, default=None, help='Path to benchmark JSONL file')
    parser.add_argument('--llm_model', type=str, default=None, help='LLM model for query generation')
    parser.add_argument('--bm25_path', type=str, default=None, help='Path for BM25 index file')
    parser.add_argument('--output_dir', type=str, default=None, help='Base directory to save evaluation results')
    parser.add_argument('--rebuild_index', action='store_true', help='Force rebuild of indices')
    parser.add_argument('--top_k', type=int, nargs='+', default=None, help='Top-k values for evaluation')
    parser.add_argument('--device', type=str, default=None, help='Device for embedding model')
    parser.add_argument('--is_local', action='store_true', default=None, help='Use local LLM API')
    parser.add_argument('--prompt_type', type=str, default=None, choices=['complex', 'simple'], help='Type of prompt to use ("complex" or "simple")')
    parser.add_argument('--search_method', type=str, default=None, choices=['vector', 'bm25'], help='Search method to use ("vector" or "bm25")')
    parser.add_argument('--workflow', type=str, default=None, choices=['simple', 'deep_research'], help='Evaluation workflow to use')
    parser.add_argument('--max_iterations', type=int, default=None, help='Maximum number of iterations for deep research workflow')
    parser.add_argument('--results_per_query', type=int, default=None, help='Results per query for deep research workflow')
    parser.add_argument('--browser_mode', type=str, default=None, choices=['PRE_ENRICH', 'REFRESH', 'INCREMENTAL', 'NONE'], help='Browser mode for deep research workflow')

    args = parser.parse_args()

    # Load config from custom path if specified (updates config module in-place)
    if args.config:
        load_config_from_path(args.config)

    # CLI args override config values (only when explicitly provided)
    if args.paper_db:
        config.PAPER_DB_PATH = args.paper_db
    if args.benchmark_jsonl:
        config.BENCHMARK_PATH = args.benchmark_jsonl
    if args.llm_model:
        config.LLM_MODEL_NAME = args.llm_model
    if args.bm25_path:
        config.BM25_PATH = args.bm25_path
    if args.output_dir:
        config.EVAL_BASE_DIR = args.output_dir
    if args.top_k:
        config.EVAL_TOP_K_VALUES = args.top_k
    if args.device:
        config.DEVICE = args.device
    if args.is_local is not None:
        config.IS_LOCAL_LLM = args.is_local
    if args.prompt_type:
        config.EVAL_PROMPT_TYPE = args.prompt_type
    if args.search_method:
        config.EVAL_SEARCH_METHOD = args.search_method
    if args.workflow:
        config.EVAL_WORKFLOW = args.workflow
    if args.max_iterations:
        config.EVAL_MAX_ITERATIONS = args.max_iterations
    if args.results_per_query:
        config.MAX_RESULTS_PER_QUERY = args.results_per_query
    if args.browser_mode:
        config.BROWSER_MODE = args.browser_mode

    # Build output directory name from config (distinguish params by workflow)
    reasoning_flag = 'reasoning' if config.ENABLE_REASONING else 'instruct'
    structured_flag = 'structured' if config.ENABLE_STRUCTURED_OUTPUT else 'non-structured'
    ablation_flag = '_ablation' if config.PLANNER_ABLATION else ''
    model_name = config.LLM_MODEL_NAME.split('/')[-1] if '/' in config.LLM_MODEL_NAME else config.LLM_MODEL_NAME
    if config.EVAL_WORKFLOW == 'simple':
        workflow_params = f"topk-{config.EVAL_TOP_K_VALUES}"
    else:
        workflow_params = f"maxq-{config.MAX_RESULTS_PER_QUERY}_iter-{config.EVAL_MAX_ITERATIONS}"
    current_output_dir = os.path.join(
        config.EVAL_BASE_DIR,
        f"{model_name}_{config.EVAL_PROMPT_TYPE}_{config.EVAL_SEARCH_METHOD}_{config.EVAL_WORKFLOW}"
        f"_{workflow_params}"
        f"_{reasoning_flag}_{structured_flag}_{config.BROWSER_MODE}{ablation_flag}"
    )
    os.makedirs(current_output_dir, exist_ok=True)

    # Save config file for reproduction
    try:
        source_config = args.config if args.config else config.__file__
        if source_config:
            target_config_path = os.path.join(current_output_dir, "config.py")
            shutil.copy(source_config, target_config_path)
            logger.info(f"[💾] Config file saved to {target_config_path}")
    except Exception as e:
        logger.warning(f"[⚠️] Failed to save config file: {e}")

    config.CASE_STUDY_OUTPUT_DIR = os.path.join(current_output_dir, "case_study")

    logger.info("[🚀]Initializing RAG system...")
    rag_system = CitationRAGSystem(
        search_method=config.EVAL_SEARCH_METHOD
    )

    rag_system.load_or_build_indices(
        paper_db_path=config.PAPER_DB_PATH,
        bm25_path=config.BM25_PATH,
        rebuild=args.rebuild_index
    )

    # Initialize trace recorder if enabled
    trace_recorder = None
    if config.SAVE_AGENT_TRACES:
        trace_recorder = AgentTraceRecorder(
            output_dir=config.EVAL_BASE_DIR,
            model_name=config.LLM_MODEL_NAME,
            prompt_type=config.EVAL_PROMPT_TYPE,
            search_method=config.EVAL_SEARCH_METHOD,
            workflow=config.EVAL_WORKFLOW,
            top_k=config.EVAL_TOP_K_VALUES,
            max_results=config.MAX_RESULTS_PER_QUERY,
            enable_reasoning=config.ENABLE_REASONING,
            enable_structured=config.ENABLE_STRUCTURED_OUTPUT
        )

    logger.info("[🚀]Initializing evaluator...")
    evaluator = CitationEvaluator(
        rag_system=rag_system,
        trace_recorder=trace_recorder
    )

    # Load and process benchmark data
    benchmark_data = evaluator.load_benchmark_data(config.BENCHMARK_PATH)

    detailed_results_file = os.path.join(current_output_dir, 'detailed_results.jsonl')
    summary_file = os.path.join(config.EVAL_BASE_DIR, 'evaluation_summary.jsonl')

    logger.info("[📈]Starting evaluation...")
    results = evaluator.evaluate_benchmark(
        benchmark_data=benchmark_data,
        workflow=config.EVAL_WORKFLOW,
        top_k_list=config.EVAL_TOP_K_VALUES,
        detailed_results_file=detailed_results_file,
        enable_resume=True
    )

    evaluator.print_summary(results)
    
    evaluator.save_results(results, current_output_dir)

    evaluator.append_summary_record(results, summary_file, detailed_results_file)

    logger.info(f"[✅] Evaluation completed! Results in: {current_output_dir}")

    logger.info(f"[📊] Summary record saved to: {summary_file}")


if __name__ == "__main__":
    main()
