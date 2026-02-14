#!/usr/bin/env python3
"""
Paper Summarizer Agent: Summarizes paper abstracts with caching support.
"""
import os
import json
from typing import Dict, List, Optional
from logger import get_logger
from api import _call_llm_async
from prompt import PAPER_SUMMARY_BATCH_PROMPT
import config
from structures import Paper
from utils import parse_json_from_tag

logger = get_logger(__name__, log_file='./log/summarizer.log')

class PaperSummarizer:
    """Summarizes paper abstracts, with JSONL file for caching."""
    
    def __init__(
        self,
        llm_model: str,
        gen_params: Dict,
        is_local: bool,
        cache_path: str = "summary_cache.jsonl",
        trace_recorder=None
    ):
        self.llm_model = llm_model
        self.gen_params = gen_params
        self.is_local = is_local
        self.cache_path = cache_path
        self.summary_cache = self._load_cache()
        self.trace_recorder = trace_recorder

    def _load_cache(self) -> Dict[str, Dict]:
        """Loads summaries from the JSONL cache file into memory."""
        cache = {}
        if not os.path.exists(self.cache_path):
            return cache
        
        with open(self.cache_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'paper_id' in data and 'model' in data and 'summary' in data:
                        cache[data['paper_id']] = {
                            'model': data['model'],
                            'summary': data['summary']
                        }
                except (json.JSONDecodeError, TypeError):
                    logger.warning(f"Skipping malformed line in cache file: {self.cache_path}")
        
        logger.info(f"Loaded {len(cache)} summaries from cache: {self.cache_path}")
        return cache

    def _get_summary_from_cache(self, paper_id: str) -> Optional[str]:
        """Retrieves a summary from cache if it matches the current model."""
        cached_entry = self.summary_cache.get(paper_id)
        if cached_entry and cached_entry.get('model') == self.llm_model:
            return cached_entry.get('summary')
        return None

    def _append_summary_to_cache(self, paper_id: str, summary: str):
        """Appends a new summary to the JSONL cache file and updates memory."""
        cache_entry = {
            'paper_id': paper_id,
            'model': self.llm_model,
            'summary': summary
        }
        
        try:
            with open(self.cache_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(cache_entry) + '\n')
            
            # Update in-memory cache
            self.summary_cache[paper_id] = {
                'model': self.llm_model,
                'summary': summary
            }
        except IOError as e:
            logger.error(f"Failed to write to cache file {self.cache_path}: {e}")

    async def batch_summarize(
        self,
        papers: List[Paper],
        user_query: str,
        sub_query_text: str,
        iteration_index: int = 1,
        idx: int = 0
    ) -> Dict[str, str]:
        """
        Batch summarizes multiple papers in a single LLM call.
        
        Args:
            papers: List of papers to summarize
            user_query: Original user research query
            sub_query_text: Subquery text for context
            
        Returns:
            Dict mapping paper_id to summary
        """
        # Filter out papers that already have cached summaries
        uncached_papers = [
            p for p in papers
            if not self._get_summary_from_cache(p.id)
        ]
        
        if not uncached_papers:
            logger.info("[Cache] All papers already have cached summaries")
            return {}
        
        logger.info(
            f"[✍️ Batch Summary] Generating summaries for "
            f"{len(uncached_papers)}/{len(papers)} papers..."
        )
        
        # Build papers block for prompt
        papers_block = "\n\n".join([
            f"paper_id={p.id}\nTitle: {p.title}\nAbstract: {p.abstract}"
            for p in uncached_papers
        ])
        
        prompt = PAPER_SUMMARY_BATCH_PROMPT.format(
            query=user_query,
            text=sub_query_text,
            papers_block=papers_block,
        )
        
        # Use structured output if enabled
        reasoning_content = None
        if config.ENABLE_STRUCTURED_OUTPUT:
            data = await _call_llm_async(
                prompt,
                self.llm_model,
                self.gen_params,
                self.is_local,
                return_structured=True,
                response_format={"type": "json_object"},
                enable_thinking=config.ENABLE_REASONING
            )
            # Extract summaries dict from structured output
            summaries = data.get("summaries", {}) if isinstance(data, dict) else {}
            response = json.dumps(data)  # For logging
        else:
            result = await _call_llm_async(
                prompt,
                self.llm_model,
                self.gen_params,
                self.is_local,
                enable_thinking=config.ENABLE_REASONING
            )
            if config.ENABLE_REASONING and isinstance(result, tuple):
                reasoning_content, response = result
            else:
                response = result
            
            # Parse from XML tag (old format)
            data = parse_json_from_tag(response, "summary_output")
            summaries = data if isinstance(data, dict) else {}
        
        # Record trace if trace recorder is enabled
        if self.trace_recorder and config.SAVE_AGENT_TRACES:
            stage_data = {
                'iteration': iteration_index,
                'sub_query_text': sub_query_text,
                'paper_count': len(uncached_papers),
                'prompt': prompt,
                'response': response,
                'reasoning': reasoning_content
            }
            self.trace_recorder.record_stage(idx, 'summarizer', stage_data)
        
        # Cache all new summaries
        for paper_id, summary in summaries.items():
            self._append_summary_to_cache(paper_id, summary)
        
        logger.info(f"[✍️ Batch Summary] Generated {len(summaries)} summaries")
        return summaries

    def apply_summaries_to_papers(
        self,
        papers: List[Paper],
        summaries: Dict[str, str]
    ) -> List[Paper]:
        """
        Applies summaries to papers and fills in from cache if available.
        
        Args:
            papers: List of papers to update
            summaries: Dict mapping paper_id to summary
            
        Returns:
            Updated list of papers with summaries
        """
        for paper in papers:
            if paper.id in summaries:
                paper.summary = summaries[paper.id]
            elif not paper.summary:
                # Try to get from cache
                cached = self._get_summary_from_cache(paper.id)
                if cached:
                    paper.summary = cached
                    logger.debug(f"[Cache] Applied cached summary for {paper.id}")
        
        return papers

