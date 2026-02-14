#!/usr/bin/env python3
"""
Planner Agent: Plans subqueries and per-subquery target_k using memory.
"""
import json
from typing import Dict, List, Tuple
from logger import get_logger
from api import _call_llm
from prompt import (
    PLANNER_SYSTEM_PROMPT_FULL_HISTORY,
    PLANNER_ITERATION_PROMPT_FULL_HISTORY,
    PLANNER_SYSTEM_PROMPT_ABLATION,
    PLANNER_ITERATION_PROMPT_ABLATION,
)
import config
from structures import SubQuery, SubQueryState, ResearchMemory, PlannerOutput,PlannerOutputABLATION
from utils import parse_json_from_tag

logger = get_logger(__name__, log_file='./log/planner.log')

class Planner:
    """Planner Agent producing subqueries and per-subquery target_k using memory."""
    
    def __init__(self, llm_model: str, gen_params: Dict, is_local: bool, trace_recorder=None):
        self.llm_model = llm_model
        self.gen_params = gen_params
        self.is_local = is_local
        self.trace_recorder = trace_recorder

    def _format_iteration_states(
        self,
        subquery_states: Dict[int, List[SubQueryState]],
        current_iteration: int,
    ) -> Tuple[str, str]:
        """
        Returns two strings:
        - last_iteration_state: Subquery states belonging to the current iteration
        - previous_iteration_state: Subquery states belonging to earlier iterations
        """
        if not subquery_states:
            return "none", "none"

        last_lines = []
        prev_lines = []

        for sq_id, states in subquery_states.items():
            for state in states:
                sq_iter = state.subquery.iter_index
                retrieved_count = state.total_requested
                selected_count = len(state.selected_papers)
                selector_overview = (state.selector_overview or "").replace("\n", " ").strip()
                target_k = state.subquery.target_k

                line = (
                    f"id={sq_id} | iteration={sq_iter} | text=\"{state.subquery.text}\" | target_k={target_k} "
                    f"| retrieved={retrieved_count} | selected={selected_count} | selector_overview=\"{selector_overview}\""
                )

                if sq_iter == current_iteration - 1:
                    last_lines.append(line)
                elif sq_iter is not None and sq_iter < current_iteration - 1:
                    prev_lines.append(line)

        last_str = "\n".join(last_lines) if last_lines else "none"
        prev_str = "\n".join(prev_lines) if prev_lines else "none"

        return last_str, prev_str

    def plan_iteration(
        self,
        user_query: Dict,
        memory: ResearchMemory,
        subquery_states: Dict[int, List[SubQueryState]],
        iteration_index: int,
        idx: int,
    ) -> Tuple[Dict[int, SubQuery], str, str, bool]:
        """
        Plans the next iteration of subqueries.
        
        Returns:
            (new_subqueries, experience_replay, checklist, is_complete)
        """
        logger.info("[🧠 Planner] Planning next iteration...")
        last_iteration_state, previous_iteration_state = self._format_iteration_states(
            subquery_states, iteration_index
        )
        
        # Build full historical subqueries listing (include root)
        all_subqueries_content = []
        root_text = memory.root_text or (
            user_query.get("query") if isinstance(user_query, dict) else str(user_query)
        )
        all_subqueries_content.append(f"id=0 | iteration=0 | text=\"{root_text}\"")

        # Include every known subquery from current memory/state
        known_ids = set(memory.subqueries_dag) | set(subquery_states.keys())
        for sid in sorted(known_ids):
            if sid == 0:
                continue
            # Prefer subquery_states for the freshest info
            if sid in subquery_states:
                for state in subquery_states[sid]:
                    sq = state.subquery
                    text = sq.text
                    source_id = sq.source_subquery_id if sq.source_subquery_id is not None else 0
                    link_type = sq.link_type or ""
                    it = getattr(sq, "iter_index", 0)
                    all_subqueries_content.append(
                        f"id={sid} | iteration={it} | source={source_id} | link={link_type} | text=\"{text}\""
                    )
        
        all_subqueries_text = "\n".join(all_subqueries_content) if all_subqueries_content else "none"
        
        if config.PLANNER_ABLATION:
            # Use full context ablation prompts
            # Construct planner_history text from memory
            planner_history_str = "\n".join([f"[iter{i+1}]\n{h}" for i, h in enumerate(memory.planner_history)]) if memory.planner_history else "none"
            
            prompt = (
                PLANNER_SYSTEM_PROMPT_ABLATION.format(
                    max_results_per_request=config.MAX_RESULTS_PER_QUERY,
                    max_pages_per_query=config.MAX_PAGES_PER_QUERY,
                    max_pages_per_query_minus_one=config.MAX_PAGES_PER_QUERY - 1,
                )
                + "\n\n"
                + PLANNER_ITERATION_PROMPT_ABLATION.format(
                    user_query=user_query["query"],
                    current_iteration=iteration_index,
                    last_iteration_state=last_iteration_state,
                    previous_iteration_state=previous_iteration_state,
                    last_checklist=memory.last_checklist or "none",
                    planner_history=planner_history_str,
                    max_results_per_request=config.MAX_RESULTS_PER_QUERY,
                    all_subqueries=all_subqueries_text,
                )
            )
        else:
            # Use default experience replay prompts
            prompt = (
                PLANNER_SYSTEM_PROMPT_FULL_HISTORY.format(
                    max_results_per_request=config.MAX_RESULTS_PER_QUERY,
                    max_pages_per_query=config.MAX_PAGES_PER_QUERY,
                    max_pages_per_query_minus_one=config.MAX_PAGES_PER_QUERY - 1,
                )
                + "\n\n"
                + PLANNER_ITERATION_PROMPT_FULL_HISTORY.format(
                    user_query=user_query["query"],
                    current_iteration=iteration_index,
                    last_iteration_state=last_iteration_state,
                    previous_iteration_state=previous_iteration_state,
                    last_checklist=memory.last_checklist or "none",
                    last_experience_replay=memory.last_experience_replay or "none",
                    max_results_per_request=config.MAX_RESULTS_PER_QUERY,
                    all_subqueries=all_subqueries_text,
                )
            )
        
        # Use structured output if enabled, otherwise parse from text
        reasoning_content = None
        if config.ENABLE_STRUCTURED_OUTPUT:
            response_format = PlannerOutputABLATION if config.PLANNER_ABLATION else PlannerOutput
            data = _call_llm(
                prompt, self.llm_model, self.gen_params, self.is_local,
                return_structured=True, response_format=response_format,
                enable_thinking=config.ENABLE_REASONING
            )
            response = json.dumps(data)
        else:
            result = _call_llm(
                prompt, self.llm_model, self.gen_params, self.is_local,
                enable_thinking=config.ENABLE_REASONING
            )
            if config.ENABLE_REASONING and isinstance(result, tuple):
                reasoning_content, response = result  # Extract reasoning and content
            else:
                response = result
            data = parse_json_from_tag(response, "planner_output") or {}
        
        if config.PLANNER_ABLATION:
            memory.planner_history.append(json.dumps(data, ensure_ascii=False))

        # Save prompt and response for case study analysis
        if idx < 5 and config.DEBUG:
            import os
            case_study_dir = os.path.join(config.CASE_STUDY_OUTPUT_DIR, f"qid_{idx}", f"iter_{iteration_index}", "planner")
            os.makedirs(case_study_dir, exist_ok=True)
            with open(os.path.join(case_study_dir, "prompt.txt"), "w", encoding="utf-8") as f:
                f.write(prompt)
            if response:
                with open(os.path.join(case_study_dir, "response.txt"), "w", encoding="utf-8") as f:
                    f.write(response)

        # Record trace if trace recorder is enabled
        if self.trace_recorder and config.SAVE_AGENT_TRACES:
            stage_data = {
                'iteration': iteration_index,
                'prompt': prompt,
                'response': response,
                'reasoning': reasoning_content
            }
            self.trace_recorder.record_stage(idx, 'planner', stage_data)
        
        subqueries_result = data.get("subqueries", [])
        checklist = data.get("checklist", "")
        experience = data.get("experience_replay", "")
        is_complete = bool(data.get("is_complete", data.get("complete", False)))

        # TODO[fix]: Early return if complete with no subqueries
        if is_complete and not subqueries_result:
            logger.info("[✅ Planner] Research marked complete with no new subqueries.")
            return {}, experience, checklist, is_complete

        subqueries: Dict[int, SubQuery] = {}

        # Helper to allocate fresh IDs (root id=0 is reserved)
        existing_ids = set(subquery_states.keys())
        # existing_ids.add(0) # TODO[fix]: planner can continue from the root query
        
        def allocate_new_id() -> int:
            candidate = 1
            while candidate in existing_ids:
                candidate += 1
            existing_ids.add(candidate)
            return candidate

        for item in subqueries_result:
            text = item.get("text", "")
            target_k = int(item.get("target_k", config.MAX_RESULTS_PER_QUERY))
            link_type = item.get("link_type")
            source_id = item.get("source_id")

            # TODO[Fix]: Validate link_type and text combination
            if not ((link_type == "continue") or (link_type in ["derive", "expand"] and text)):
                continue

            if link_type == "continue":
                # Reuse existing subquery id
                reuse_id = source_id if source_id in existing_ids else None
                if reuse_id is None:
                    # If invalid continue, skip
                    continue
                if len(subquery_states[reuse_id]) >= config.MAX_PAGES_PER_QUERY:
                    # If already continued max times, skip
                    continue
                new_id = reuse_id
                source = subquery_states[reuse_id][-1].subquery.source_subquery_id
                text = subquery_states[reuse_id][-1].subquery.text
                # TODO[fix]: planner can continue from the root query
                # source = subquery_states[reuse_id][-1].subquery.source_subquery_id if reuse_id != 0 else -1
                # text = subquery_states[reuse_id][-1].subquery.text if reuse_id != 0 else user_query["query"]
            else:
                # derive or expand => allocate new id
                new_id = allocate_new_id()
                source = source_id if isinstance(source_id, int) else 0

            subqueries[new_id] = SubQuery(
                id=new_id,
                text=text,
                before_date=user_query.get("date"),
                target_k=target_k,
                link_type=link_type,
                source_subquery_id=source,
                iter_index=iteration_index,
            )

        return subqueries, experience, checklist, is_complete
