#!/usr/bin/env python3
"""
Deep Research Workflow (Refactored): Two-agent system with explicit Memory.
- Planner Agent: plans subqueries and target_k per subquery based on user query and memory.
- Selector Agent: batch-decides which retrieved papers to maintain per subquery using scores.
- MCP Retrieval Tool: `code/mcp/retrieval_mcp.py` provides unified search across methods.
- Paper Summarization: abstracts may be summarized when text budget is exceeded.
"""
import asyncio
import time
from typing import List, Dict, Set, Optional, Any

from logger import get_logger
from rag import CitationRAGSystem
import config
from structures import Paper, SubQuery, SubQueryState, ResearchMemory
from mcp.retrieval_mcp import search_papers
from agent import Planner, Selector, PaperSummarizer, Browser
from metrics import Timer, MetricsCalculator


logger = get_logger(__name__, log_file='./log/deeprag.log')

class DeepResearchWorkflow:
    """
    Orchestrates a multi-agent workflow for deep, iterative citation research.
    """
    def __init__(self, rag_system: CitationRAGSystem, llm_model: str, gen_params: Dict, is_local: bool, trace_recorder=None):
        self.planner = Planner(llm_model, gen_params, is_local, trace_recorder=trace_recorder)
        self.selector = Selector(llm_model, gen_params, is_local, trace_recorder=trace_recorder)
        self.browser = Browser(
            llm_model,
            gen_params,
            is_local,
            trace_recorder=trace_recorder
        )
        self.summarizer = PaperSummarizer(
            config.SUMMARY_LLM_MODEL_NAME,
            config.SUMMARY_LLM_GEN_PARAMS,
            config.SUMMARY_LLM_IS_LOCAL,
            cache_path=config.SUMMARY_CACHE_PATH,
            trace_recorder=trace_recorder
        )
        self.rag_system = rag_system
        self.trace_recorder = trace_recorder

    def _mcp_results_to_papers(self, results: List[Dict]) -> List[Paper]:
        papers: List[Paper] = []
        for item in results:
            paper = Paper(
                id=item.get("paper_id", ""),
                title=item.get("title", "N/A"),
                abstract=item.get("abstract", "N/A"),
                arxiv_id=item.get("arxiv_id", "N/A"),
                date=item.get("date"),
                score=item.get("score"),
            )
            papers.append(paper)
        return papers
    
    @staticmethod
    def _process_selector_results(
        selection_results: Dict,
        cur_iter_selected_papers: Dict[int, List[Paper]],
        subquery_states: Dict[int, List[SubQueryState]],
        checklist: str,
        papers_for_browsing: Dict[int, List[Dict[str, Any]]]
    ):
        """
        Helper method to process selection results:
        1. Deduplicate and merge papers into cur_iter_selected_papers.
        2. Update subquery states.
        3. Collect new browsing tasks.
        """
        for sq_id, result in selection_results.items():
            # Handle default empty result safely
            kept, overview, to_browse = result if result else ([], "", {})
            
            # --- 1. Deduplication and Merge ---
            if config.BROWSER_MODE == "INCREMENTAL":
                # If incremental, merge into existing list（下面的逻辑是为了去重）
                current_list = cur_iter_selected_papers.setdefault(sq_id, [])
                existing_ids = {p.id for p in current_list}
                new_unique_papers = [p for p in kept if p.id not in existing_ids]
                current_list.extend(new_unique_papers)
            else:
                # If not incremental, simply overwrite
                cur_iter_selected_papers[sq_id] = kept
            
            # --- 2. Update Memory State ---
            # Ensure we have a state to update
            if sq_id in subquery_states and subquery_states[sq_id]:
                st = subquery_states[sq_id][-1]
                st.selected_papers = cur_iter_selected_papers.get(sq_id, [])  # Update reference to the accumulated list
                st.checklist = checklist
                st.selector_overview = overview
            
            # --- 3. Update Browsing Tasks ---
            if to_browse:
                papers_for_browsing[sq_id] = list(to_browse.values())
    
    def run(
        self, 
        query: Dict, 
        gt_arxiv_ids: Set[str] = None,
        results_per_query: int = None, 
        max_iterations: int = 3, 
        idx: int = 1, 
    ) -> Optional[Dict]:
        """
        Executes the full Deep Research workflow.
        
        Args:
            query: The initial research query dictionary.
            gt_arxiv_ids: Ground truth arXiv IDs for evaluation metrics.
            results_per_query: Maximum results per retrieval request (defaults to config.MAX_RESULTS_PER_QUERY).
            max_iterations: Maximum number of research iterations (replanning loops).
            idx: Query index for case study logging.

        Returns:
            Dictionary containing final report, retrieved papers, history, and executed queries.
            Returns None if the workflow fails (e.g., planner marks complete but no papers selected).
        """
        final_selected_papers: Dict[int, List[Paper]] = {}
        executed_queries: Set[str] = set()
        subquery_states: Dict[int, List[SubQueryState]] = {}
        history: List[Dict] = []

        # Memory across the entire workflow
        memory = ResearchMemory()
        memory.root_subquery_id = 0
        memory.root_text = query.get('query')
        
        # Cross-iteration tracking for ranking and paper selection
        selected_min_rank_tracker: Dict[str, int] = {}  # Track minimum rank for selected papers across iterations
        selected_paper_ids_tracker: Set[str] = set()  # Track all selected paper IDs across iterations
        
        for i in range(max_iterations):
            iter_idx = i + 1
            logger.info(f"[🔄 Iteration {iter_idx}/{max_iterations}] Starting...")

            start_time = time.time()
            timings: Dict[str, float] = {
                "planner": 0.0,
                "retrieval": 0.0,
                "selector": 0.0,
                "browser": 0.0,
            }
            # 1) Planner iteration
            with Timer() as planner_timer:
                subqueries, experience, checklist, is_complete = self.planner.plan_iteration(
                    user_query=query, memory=memory, subquery_states=subquery_states, iteration_index=iter_idx, idx=idx
                )
            timings["planner"] += planner_timer.elapsed
            
            history.append({
                'stage': 'plan',
                'iter_idx': iter_idx,
                'sub_queries': [{"id": sq.id, "text": sq.text, "target_k": sq.target_k, "link_type": sq.link_type, "source_id": sq.source_subquery_id} for sq in subqueries.values()],
                'experience_replay': experience,
                'checklist': checklist,
            })

            # TODO[fix]: Handle completion scenarios, early stop
            if is_complete:
                if final_selected_papers:
                    logger.info("[✅] Planner decided research is complete with selected papers.")
                    break
                else:
                    logger.warning("[❌] Planner marked complete but no papers selected - query execution failed.")
                    return None
            
            # If no subqueries planned but not complete, something is wrong
            if not subqueries:
                logger.warning("[⚠️] No subqueries planned and research not complete - stopping iteration.")
                return None

            # 2) Retrieval via MCP per subquery with target_k and offsets (async)
            async def fetch_for_subquery(sq: SubQuery):
                # TODO: explain it
                already_states = subquery_states.get(sq.id, [])
                num_already = sum(len(st.retrieved_papers) for st in already_states)
                desired_total = sq.target_k
                # to_fetch = max(0, desired_total - num_already)
                
                to_fetch = desired_total
                if to_fetch == 0:
                    return sq.id, []
                per_call_k = min(to_fetch if to_fetch > 0 else results_per_query, config.MAX_RESULTS_PER_QUERY)
                results, rank_dict = await asyncio.get_running_loop().run_in_executor(
                    None,
                    search_papers,
                    self.rag_system,
                    sq.text,
                    per_call_k,
                    getattr(self.rag_system, 'search_method', 'hybrid'),
                    sq.before_date,
                    num_already,
                    gt_arxiv_ids,
                    selected_paper_ids_tracker
                )
                executed_queries.add(sq.text)
                return sq.id, self._mcp_results_to_papers(results), rank_dict

            async def run_retrieval(subqueries_list: List[SubQuery]):
                tasks = [fetch_for_subquery(sq) for sq in subqueries_list]
                results = await asyncio.gather(*tasks)
                return {sq_id: (papers, rank_dict) for sq_id, papers, rank_dict in results}

            
            with Timer() as retrieval_timer:
                retrieval_map = asyncio.run(run_retrieval(subqueries.values()))
            timings["retrieval"] += retrieval_timer.elapsed
                
            papers_for_selection: Dict[int, List[Paper]] = {}
            rank_dicts: Dict[int, dict] = {}

            for sq_id, (new_papers, rank_dict) in retrieval_map.items():
                sq_obj = subqueries.get(sq_id, None)
                new_state = SubQueryState(subquery=sq_obj, retrieved_papers=new_papers, total_requested=len(new_papers))
                if sq_id not in subquery_states:
                    subquery_states[sq_id] = [new_state]
                else:
                    subquery_states[sq_id].append(new_state)
                papers_for_selection[sq_id] = new_state.retrieved_papers
                rank_dicts[sq_id] = rank_dict

            original_papers_for_selection = {
                sq_id: list(papers) for sq_id, papers in papers_for_selection.items()
            }
            
            # 3) Optional batch summarization per subquery
            if config.ENABLE_SUMMARIZATION:
                async def run_batch_summaries():
                    tasks = [
                        self.summarizer.batch_summarize(
                            papers=plist,
                            user_query=query['query'],
                            sub_query_text=subquery_states[sq_id][-1].subquery.text,
                            iteration_index=iter_idx,
                            idx=idx
                        )
                        for sq_id, plist in papers_for_selection.items()
                    ]
                    results = await asyncio.gather(*tasks)
                    return {
                        sq_id: summaries
                        for (sq_id, _), summaries in zip(papers_for_selection.items(), results)
                    }

                summaries_map_by_sq = asyncio.run(run_batch_summaries())
                
                # Apply summaries to papers using the new method
                for sq_id, plist in papers_for_selection.items():
                    summaries = summaries_map_by_sq.get(sq_id, {})
                    self.summarizer.apply_summaries_to_papers(plist, summaries)

            
            # 4) Selector decisions per subquery
            async def run_selection(is_after_browsing: bool = False):
                tasks = [
                    self.selector.decide_for_subquery(
                        user_query=query['query'],
                        sub_query=subqueries[sq_id],
                        planner_checklist=checklist,
                        papers=plist,
                        iteration_index=iter_idx,
                        idx=idx,
                        old_overview=subquery_states[sq_id][-1].selector_overview,
                        is_after_browsing=is_after_browsing
                    )
                    for sq_id, plist in papers_for_selection.items()
                ]
                results = await asyncio.gather(*tasks)
                return {sq_id: result for (sq_id, _), result in zip(papers_for_selection.items(), results)}
            
            # 5) Browser tool calls for unsure papers
            async def run_browsing():
                tasks = [
                    self.browser.browse_papers(
                        subquery_text=subqueries[sq_id].text,  
                        paper=item['paper'],              
                        iteration_index=iter_idx,
                        idx=idx,                          
                        sq_id=sq_id,                      
                        paper_idx=p_idx,                  
                        task=item['goal']
                    )
                    for sq_id, items in papers_for_browsing.items()
                    for p_idx, item in enumerate(items)
                ]

                await asyncio.gather(*tasks)

            papers_for_browsing: Dict[int, List[Dict[str, Any]]] = {}
            browsing_arxiv_ids: Set[str] = set()
            
            # Mode 1: Pre-enrich all paper content
            if config.BROWSER_MODE == "PRE_ENRICH":
                for sq_id, paper_list in papers_for_selection.items():
                    for paper in paper_list:
                        # Unified packaging: no goal here, set to None
                        papers_for_browsing.setdefault(sq_id, []).append({
                            'paper': paper,
                            'goal': None
                        })
                
                for items in papers_for_browsing.values():
                    for item in items:
                        if item['paper'].arxiv_id:
                            browsing_arxiv_ids.add(item['paper'].arxiv_id)

                with Timer() as browse_timer:
                    asyncio.run(run_browsing())
                timings["browser"] += browse_timer.elapsed
                
            # Selector decision
            with Timer() as selector_timer:
                selection_results = asyncio.run(run_selection(is_after_browsing=False))
            timings["selector"] += selector_timer.elapsed
                
            cur_iter_selected_papers: Dict[int, List[Paper]] = {}
            
            self._process_selector_results(selection_results,cur_iter_selected_papers,subquery_states,checklist,papers_for_browsing)
            
            # Mode 2 & 3: Refresh or incrementally update browsing of uncertain papers
            if config.BROWSER_MODE in ["REFRESH", "INCREMENTAL"]:
                # for _ in range(config.MAX_BROWSER_CALLS):
                if papers_for_browsing:
                    for items in papers_for_browsing.values():
                        for item in items:
                            if item['paper'].arxiv_id:
                                browsing_arxiv_ids.add(item['paper'].arxiv_id)

                    with Timer() as browse_timer:
                        asyncio.run(run_browsing())
                    timings["browser"] += browse_timer.elapsed
                    
                    if config.BROWSER_MODE == "INCREMENTAL":
                    # INCREMENTAL only passes the browsed papers
                        papers_for_selection = {
                            sq_id: [item['paper'] for item in items]
                            for sq_id, items in papers_for_browsing.items()
                        }
                        
                    else:
                    # REFRESH mode requires retransmitting all papers for the sq_id involved in papers_for_browsing
                        papers_for_selection = {
                            sq_id: original_papers_for_selection.get(sq_id, [])
                            for sq_id in papers_for_browsing.keys()
                        }
                    
                    with Timer() as selector_timer:
                        selection_results = asyncio.run(run_selection(is_after_browsing=True))
                    timings["selector"] += selector_timer.elapsed
                    
                    self._process_selector_results(selection_results,cur_iter_selected_papers,subquery_states,checklist,papers_for_browsing)
            
            papers_for_selection = original_papers_for_selection

            # Update selected_paper_ids_tracker with all papers selected in this iteration
            for kept in cur_iter_selected_papers.values():
                if kept:
                    selected_paper_ids_tracker.update({p.id for p in kept})
            
            # Merge kept papers
            for sq_id, kept in cur_iter_selected_papers.items():
                if sq_id not in final_selected_papers:
                    final_selected_papers[sq_id] = []
                existing = {p.id for p in final_selected_papers[sq_id]}
                final_selected_papers[sq_id].extend([p for p in kept if p.id not in existing])

            # Update memory
            if experience:
                memory.last_experience_replay = experience
            if checklist:
                memory.last_checklist = checklist
            memory.subqueries_dag.extend([sq.id for sq in subqueries.values()])
            for sq in subqueries.values():
                memory.subqueries_meta[sq.id] = {
                    "text": sq.text,
                    "source": sq.source_subquery_id,
                    "iter": sq.iter_index,
                }

            # 6) Calculate metrics and logging for this iteration
            subquery_metrics = {}  # Store metrics for each subquery
            
            # Calculate metrics for each subquery
            for sq_id in subqueries.keys():
                retrieved_papers = papers_for_selection.get(sq_id, [])
                selected_papers = cur_iter_selected_papers.get(sq_id, [])
                
                retrieved_arxiv_ids = {p.arxiv_id for p in retrieved_papers if p.arxiv_id}
                selected_arxiv_ids = {p.arxiv_id for p in selected_papers if p.arxiv_id}
                
                # Calculate subquery-level metrics
                if gt_arxiv_ids:
                    metrics = MetricsCalculator.calculate_subquery_metrics(
                        retrieved_arxiv_ids,
                        selected_arxiv_ids,
                        gt_arxiv_ids
                    )
                    subquery_metrics[sq_id] = metrics
            
            # Calculate iteration-level metrics
            iteration_subquery_results = {}
            for sq_id in subqueries.keys():
                retrieved_papers = papers_for_selection.get(sq_id, [])
                selected_papers = cur_iter_selected_papers.get(sq_id, [])
                iteration_subquery_results[sq_id] = {
                    'retrieved_arxiv_ids': {p.arxiv_id for p in retrieved_papers if p.arxiv_id},
                    'selected_arxiv_ids': {p.arxiv_id for p in selected_papers if p.arxiv_id}
                }
            
            iteration_metrics = MetricsCalculator.calculate_iteration_metrics(
                iteration_subquery_results,
                gt_arxiv_ids
            ) if gt_arxiv_ids else {}
            
            # Log iteration metrics
            if iteration_metrics:
                logger.info(
                    f"[📊 Iteration {iter_idx}] "
                    f"Retrieval - P: {iteration_metrics['retrieval']['precision']:.3f}, "
                    f"R: {iteration_metrics['retrieval']['recall']:.3f} "
                    f"({iteration_metrics['retrieval']['matched']}/{iteration_metrics['retrieval']['total']}); "
                    f"Selection - P: {iteration_metrics['selection']['precision']:.3f}, "
                    f"R: {iteration_metrics['selection']['recall']:.3f} "
                    f"({iteration_metrics['selection']['matched']}/{iteration_metrics['selection']['total']}); "
                    f"Discarded GT: {iteration_metrics['discarded_gt_count']}; "
                    f"Discarded Total: {iteration_metrics['discarded_total_count']} "
                    f"(Ratio: {iteration_metrics['discarded_ratio']:.3f})"
                )
            
            # Calculate ground truth rank and distance metrics
            rank_distance_result = MetricsCalculator.calculate_gt_rank_and_distance(
                subqueries=subqueries,
                rank_dicts=rank_dicts,
                gt_arxiv_ids=gt_arxiv_ids,
                selected_paper_ids_tracker=selected_paper_ids_tracker,
                selected_min_rank_tracker=selected_min_rank_tracker,
                gt_rank_cutoff=config.GT_RANK_CUTOFF
            )
            
            gt_rank = rank_distance_result["gt_rank"]
            avg_distance = rank_distance_result["avg_distance"]
            selected_min_rank_tracker = rank_distance_result["updated_selected_min_rank_tracker"]

            end_time = time.time()
            total_time = round(end_time - start_time, 3)

            planner_time = round(timings["planner"], 3)
            retrieval_time = round(timings["retrieval"], 3)
            selector_time = round(timings["selector"], 3)
            browser_time = round(timings["browser"], 3)

            overhead_time = round(total_time - (planner_time + retrieval_time + selector_time + browser_time), 3)

            if browser_time > 0:
                logger.info(f"[🌐 Iteration {iter_idx}] Browser time: {browser_time:.3f}s")
            history.append({
                'stage': 'select',
                'iter_idx': iter_idx,
                'browsing_arxiv_ids': sorted(list(browsing_arxiv_ids)),
                'retrieved_papers': {
                    sq_id: [{"title": p.title, "arxiv_id": p.arxiv_id, "id": p.id} for p in papers]
                    for sq_id, papers in papers_for_selection.items()
                },
                'selected_papers': {
                    sq_id: [{"title": p.title, "arxiv_id": p.arxiv_id, "id": p.id} for p in kept]
                    for sq_id, kept in cur_iter_selected_papers.items()
                },
                
                'planner_during': planner_time,
                'retrieval_during': retrieval_time,
                'selector_during': selector_time,
                'browser_during': browser_time,
                'overhead_during': overhead_time,
                'total_during': total_time,
                'gt_rank': gt_rank,
                'avg_distance': avg_distance,
                'subquery_metrics': subquery_metrics,
                'iteration_metrics': iteration_metrics
            })
            
        # debug
        logger.info("[✅] Deep Research workflow completed.")
        
        # Deduplicate final list of selected papers
        final_selected_papers_list = list({paper.id: paper for papers in final_selected_papers.values() for paper in papers}.values())
        
        # Save agent traces if enabled
        if self.trace_recorder and config.SAVE_AGENT_TRACES:
            self.trace_recorder.save_sample(idx)
        
        return {
            "idx": idx,
            "final_report": '',
            "selected_papers": final_selected_papers_list,
            "history": history,
            "executed_queries": sorted(list(executed_queries))
        }
