#!/usr/bin/env python3
"""
Selector Agent: Performs batch decisions for each subquery.
"""
import json
from typing import Dict, List, Tuple
from logger import get_logger
from api import _call_llm_async
from prompt import SELECTOR_SYSTEM_PROMPT, SELECTOR_DECISION_PROMPT, SELECTOR_RECIPE, SELECTOR_SYSTEM_INCREMENTAL_PROMPT, SELECTOR_DECISION_INCREMENTAL_PROMPT,SELECTOR_DECISION_REFRESH_PROMPT,SELECTOR_SYSTEM_REFRESH_PROMPT,SELECTOR_DECISION_PRE_ENRICH_PROMPT,SELECTOR_SYSTEM_PRE_ENRICH_PROMPT
import config
from structures import Paper, SubQuery, SelectorOutput, SelectorOutputWithBrowser
from utils import parse_json_from_tag

logger = get_logger(__name__, log_file='./log/selector.log')

def _candidate_line(p: Paper) -> str:
    if p.browsing_content:
        text ="browser_summary=" + p.browsing_content
    else:
        text =p.summary or p.abstract
        text = "abstract=" + text
    # text = (text or "")[:1200]
    score_part = f"score={p.score:.4f}" if p.score is not None else "score=N/A"
    return f"- id={p.id} | {score_part} | year={p.date} |title={p.title}|  {text}"


class Selector:
    """Selector Agent performing batch decisions for each subquery."""

    def __init__(self, trace_recorder=None):
        self.llm_model = config.LLM_MODEL_NAME
        self.gen_params = config.LLM_GEN_PARAMS
        self.is_local = config.IS_LOCAL_LLM
        self.trace_recorder = trace_recorder

    
    async def decide_for_subquery(
        self,
        user_query: str,
        sub_query: SubQuery,
        planner_checklist: str,
        papers: List[Paper],
        iteration_index: int = 1,
        idx: int = 5,
        old_overview: str = "",
        is_after_browsing: bool = False
    ) -> Tuple[List[Paper], str, Dict]:
        """
        Decides which papers to maintain for a given subquery.
        
        Args:
            user_query: Original user research query
            sub_query: Current subquery object
            planner_checklist: Checklist from Planner
            papers: Candidate papers to select from
            iteration_index: Current iteration number
            idx: Query index for case study logging
            old_overview: Previous overview text for incremental update
            
        Returns:
            (selected_papers, overview, to_browse_mapped)
        """
        if not config.ENABLE_LLM_FILTERING:
            return papers, ""

        candidates_text = "\n".join(_candidate_line(p) for p in papers)

        if config.BROWSER_MODE == "INCREMENTAL":
            prompt = (
                SELECTOR_SYSTEM_INCREMENTAL_PROMPT
                + "\n\n"
                + SELECTOR_DECISION_INCREMENTAL_PROMPT.format(
                    user_query=user_query,
                    sub_query=sub_query.text,
                    planner_checklist=planner_checklist or "",
                    selector_recipe=SELECTOR_RECIPE,
                    old_overview = old_overview,
                    candidates=candidates_text or "(none)",
                )
            )
        
        elif config.BROWSER_MODE == "PRE_ENRICH":
            prompt = (
                SELECTOR_SYSTEM_PRE_ENRICH_PROMPT
                + "\n\n"
                + SELECTOR_DECISION_PRE_ENRICH_PROMPT.format(
                    user_query=user_query,
                    sub_query=sub_query.text,
                    planner_checklist=planner_checklist or "",
                    selector_recipe=SELECTOR_RECIPE,
                    candidates=candidates_text or "(none)",
                )
            )
        elif config.BROWSER_MODE == "REFRESH":    
            prompt = (
                SELECTOR_SYSTEM_REFRESH_PROMPT
                + "\n\n"
                + SELECTOR_DECISION_REFRESH_PROMPT.format(
                    user_query=user_query,
                    sub_query=sub_query.text,
                    planner_checklist=planner_checklist or "",
                    selector_recipe=SELECTOR_RECIPE,
                    candidates=candidates_text or "(none)",
                )
            )
        else:  # NONE
            prompt = (
                SELECTOR_SYSTEM_PROMPT
                + "\n\n"
                + SELECTOR_DECISION_PROMPT.format(
                    user_query=user_query,
                    sub_query=sub_query.text,
                    planner_checklist=planner_checklist or "",
                    selector_recipe=SELECTOR_RECIPE,
                    candidates=candidates_text or "(none)",
                )
            )
        
        response_format = SelectorOutput if config.BROWSER_MODE in ["PRE_ENRICH", "NONE"] else SelectorOutputWithBrowser
        
        # Use structured output if enabled, otherwise parse from text
        reasoning_content = None
        if config.ENABLE_STRUCTURED_OUTPUT:
            data = await _call_llm_async(
                prompt, self.llm_model, self.gen_params, self.is_local,
                return_structured=True, response_format=response_format,
                enable_thinking=config.ENABLE_REASONING
            )
            response = json.dumps(data)  # For logging
        else:
            result = await _call_llm_async(
                prompt, self.llm_model, self.gen_params, self.is_local,
                enable_thinking=config.ENABLE_REASONING
            )
            if config.ENABLE_REASONING and isinstance(result, tuple):
                reasoning_content, response = result  # Extract reasoning and content
            else:
                response = result
            data = parse_json_from_tag(response, "selector_output") or {}
        
        # Save prompt and response for case study
        if idx < 5 and config.SAVE_CASE_STUDY_ARTIFACTS:
            import os
            case_study_dir = os.path.join(config.CASE_STUDY_OUTPUT_DIR, f"qid_{idx}", f"iter_{iteration_index}", "selector")
            os.makedirs(case_study_dir, exist_ok=True)
            suffix = "_after_browsing" if is_after_browsing else ""
            with open(os.path.join(case_study_dir, f"prompt_sq{sub_query.id}{suffix}.txt"), "w", encoding="utf-8") as f:
                f.write(prompt)
            with open(os.path.join(case_study_dir, f"response_sq{sub_query.id}{suffix}.txt"), "w", encoding="utf-8") as f:
                f.write(response)
        
        # Record trace if trace recorder is enabled
        if self.trace_recorder and config.SAVE_AGENT_TRACES:
            stage_data = {
                'iteration': iteration_index,
                'sub_query_id': sub_query.id,
                'sub_query_text': sub_query.text,
                'prompt': prompt,
                'response': response,
                'reasoning': reasoning_content,
                'after_browsing': is_after_browsing
            }
            self.trace_recorder.record_stage(idx, 'selector', stage_data)
            
        id_to_paper = {p.id: p for p in papers}
        selected_ids = set(str(pid) for pid in data.get("selected", []))
        overview = data.get("overview", "")
        to_browse = data.get("to_browse", {})
        selected = [id_to_paper[pid] for pid in selected_ids if pid in id_to_paper]

        if config.VERBOSE:
            discarded_ids = set(id_to_paper.keys()) - selected_ids
            logger.info(f"[VERBOSE Selector] sq={sub_query.id} \"{sub_query.text[:60]}\"")
            for pid in selected_ids:
                p = id_to_paper.get(pid)
                logger.info(f"[VERBOSE Selector]   ✓ {pid} | {getattr(p, 'title', '?')[:70]}")
            for pid in discarded_ids:
                p = id_to_paper.get(pid)
                logger.info(f"[VERBOSE Selector]   ✗ {pid} | {getattr(p, 'title', '?')[:70]}")
            if overview:
                logger.info(f"[VERBOSE Selector]   overview: {overview[:150]}{'...' if len(overview)>150 else ''}")
        to_browse_mapped = {
            str(pid): {
                "goal": goal,
                "paper": id_to_paper.get(str(pid))
            }
            for pid, goal in to_browse.items()
            if str(pid) in id_to_paper
        }
        logger.info(f"[🔎 Selector] Selected {len(selected)}/{len(papers)} for sub-query '{sub_query.text}'.")
        
        return selected, overview, to_browse_mapped

