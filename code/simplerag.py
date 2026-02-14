#!/usr/bin/env python3
"""
Simple Query Expansion Workflow: Single-iteration retrieval with query expansion.
- Generates diverse query keys using LLM
- Retrieves papers for each query key
- Combines and deduplicates results
"""
from typing import List, Dict, Set, Optional
from logger import get_logger
from rag import CitationRAGSystem
from prompt import COMPLEX_QUERY_GENERATION_PROMPT, SIMPLE_QUERY_GENERATION_PROMPT
from api import _call_llm
from metrics import MetricsCalculator
import config
from utils import parse_response_to_keys, extract_ground_truth_arxiv_ids, combine_search_results, calculate_retrieval_metrics

logger = get_logger(__name__, log_file='./log/simple_workflow.log')


class SimpleWorkflow:
    """
    Orchestrates a simple query expansion workflow for citation retrieval.
    """
    def __init__(self, rag_system: CitationRAGSystem, llm_model: str, gen_params: Dict, 
                 is_local: bool, prompt_type: str = 'simple', trace_recorder=None):
        """
        Initialize SimpleWorkflow.
        
        Args:
            rag_system: Citation RAG system for retrieval
            llm_model: LLM model name for query generation
            gen_params: Generation parameters for LLM
            is_local: Whether to use local LLM
            prompt_type: Type of prompt ('simple' or 'complex')
            trace_recorder: Optional trace recorder for debugging and analysis
        """
        self.rag_system = rag_system
        self.llm_model = llm_model
        self.gen_params = gen_params
        self.is_local = is_local
        self.trace_recorder = trace_recorder
        
        self.prompt_template = (
            SIMPLE_QUERY_GENERATION_PROMPT 
            if prompt_type == "simple" 
            else COMPLEX_QUERY_GENERATION_PROMPT
        )
    
    def generate_query_keys(self, query: str, idx: int = None) -> List[str]:
        """
        Generate diverse query keys using LLM.
        
        Args:
            query (str): Original user query
            idx (int): Query index for trace recording
            
        Returns:
            List[str]: List of generated query keys
        """
        try:
            prompt = self.prompt_template.format(query=query)
            
            response = _call_llm(
                prompt,
                self.llm_model, 
                self.gen_params, 
                self.is_local,
                enable_thinking=config.ENABLE_REASONING
            )
            
            # Handle reasoning output format and extract reasoning content
            reasoning_content = None
            response_content = response
            if config.ENABLE_REASONING and isinstance(response, tuple):
                reasoning_content, response_content = response
            
            # Record trace if enabled
            if self.trace_recorder and config.SAVE_AGENT_TRACES and idx is not None:
                stage_data = {
                    'prompt': prompt,
                    'response': response_content,
                    'reasoning': reasoning_content
                }
                self.trace_recorder.record_stage(idx, 'query_generator', stage_data)
            
            if response_content:
                return parse_response_to_keys(response_content)
            
            logger.warning(f"[💢]No response from LLM for query: {query[:50]}...")
            return []
                
        except Exception as e:
            logger.warning(f"[💢]Failed to generate keys for query: {e}")
            return []
    
    def run(self, query_data: Dict, top_k: int = 10, search_method: str = 'hybrid', idx: int = None) -> Dict:
        """
        Execute simple workflow for a single query.
        
        Args:
            query_data: Query data dictionary containing 'query', 'gt_label', 'cited_paper'
            top_k: Number of top results to retrieve per query key
            search_method: Search method to use ('vector', 'bm25', or 'hybrid')
            idx: Query index for logging/tracking
            
        Returns:
            Dict: Results containing query, ground truth, retrieved papers, and metrics
        """
        query = query_data['query']
        gt_labels = query_data['gt_label']
        gt_arxiv_ids = extract_ground_truth_arxiv_ids(query_data['cited_paper'], gt_labels)
        
        if not gt_arxiv_ids:
            return None
        
        # Generate query keys with trace recording
        query_keys = self.generate_query_keys(query, idx=idx)
        if not query_keys:
            logger.warning(f"[💢]No query keys generated for: {query[:50]}...")
            return None
        
        # Setup search function based on method
        search_functions = {
            'vector': self.rag_system.search_citations,
            'bm25': self.rag_system.search_citations_bm25,
            'hybrid': self.rag_system.search_citations_hybrid
        }
        
        search_function = search_functions.get(search_method)
        if not search_function:
            raise ValueError(f"Unsupported search method: {search_method}")
        
        # Retrieve papers for each query key and handle tuple unpacking for results and rank_dicts
        all_search_results = []
        all_rank_dicts = []
        
        for q in query_keys:
            # Call individual search function with gt_arxiv_ids for rank tracking
            result = search_function(q, top_k, gt_arxiv_ids=gt_arxiv_ids, debug=True)
            
            if isinstance(result, tuple) and len(result) == 2:
                results, rank_dict = result
                all_search_results.append(results)
                all_rank_dicts.append(rank_dict)
            else:
                # Fallback if no rank_dict returned
                all_search_results.append(result)
        
        # Combine and deduplicate results
        combined_results = combine_search_results(all_search_results)
        retrieved_arxiv_ids = {paper_info['arxiv_id'] for _, _, paper_info in combined_results}
        
        # Calculate metrics using unified function
        metrics = calculate_retrieval_metrics(gt_arxiv_ids, retrieved_arxiv_ids)
        
        # Calculate average distance metric using MetricsCalculator
        avg_distance = MetricsCalculator.calculate_simple_avg_distance(
            all_rank_dicts, gt_arxiv_ids, config.GT_RANK_CUTOFF
        )
        
        # Save agent traces if enabled
        if self.trace_recorder and config.SAVE_AGENT_TRACES and idx is not None:
            self.trace_recorder.save_sample(idx)
        
        return {
            'idx': idx,
            'query': query,
            'generated_keys': query_keys,
            'ground_truth_arxiv_ids': list(gt_arxiv_ids),
            'retrieved_arxiv_ids': list(retrieved_arxiv_ids),
            'recall': metrics['retrieval_recall'],
            'precision': metrics['retrieval_precision'],
            'matches': metrics['retrieval_matches'],
            'total_gt': len(gt_arxiv_ids),
            'avg_distance': avg_distance,
            'top_results': [
                {
                    'arxiv_id': paper_info.get('arxiv_id', 'N/A'),
                    'title': paper_info.get('title', 'N/A'),
                    'similarity': similarity,
                    'matched': paper_info.get('arxiv_id') in gt_arxiv_ids,
                }
                for _, similarity, paper_info in combined_results
            ]
        }

