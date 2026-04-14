"""Async workflow for the single-query WebSocket demo.

Reuses production agents and RAG, but drives a UI-oriented event stream
directly to a WebSocket. Intentionally isolated from workflows/deep_research.py
so eval's heavy path stays untouched.
"""
import json
import time
from typing import List, Optional, Dict, Any

import config
from agent import Planner, Selector, Browser, PaperSummarizer
from rag import CitationRAGSystem
from structures import Paper, SubQuery, SubQueryState, ResearchMemory
from logger import get_logger

logger = get_logger(__name__)


def _simple_metrics(selected_ids: set, gt_ids: set) -> dict:
    """Lightweight recall/precision/f1 over (selected, gt) sets.

    Demo is a visualization, not a benchmark — ignores rank-weighted metrics.
    """
    if not gt_ids:
        return {"recall": 0.0, "precision": 0.0, "f1": 0.0}
    matched = len(selected_ids & gt_ids)
    recall = matched / len(gt_ids)
    precision = matched / len(selected_ids) if selected_ids else 0.0
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0
    return {
        "recall": round(recall, 3),
        "precision": round(precision, 3),
        "f1": round(f1, 3),
    }


def _paper_to_payload(p: Paper) -> dict:
    """Serialize a Paper to the JSON shape the demo UI expects."""
    return {
        "arxiv_id": p.arxiv_id or p.id,
        "id": p.id,
        "title": p.title or "",
        "abstract": p.abstract or "",
        "summary": p.summary or "",
        "date": p.date,
        "score": p.score if p.score is not None else 0.0,
    }


class DemoWorkflow:
    """Async, single-query workflow driving a WebSocket event stream."""

    def __init__(self, rag_system: CitationRAGSystem, ws: Any):
        self.rag = rag_system
        self.ws = ws
        self.planner = Planner()
        self.selector = Selector()
        self.browser = Browser()
        self.summarizer = PaperSummarizer()

    async def _send(self, event: dict) -> None:
        await self.ws.send_text(json.dumps(event, ensure_ascii=False))

    async def _step(
        self,
        step_id: str,
        parent: Optional[str],
        label: str,
        status: str = "running",
        **extra,
    ) -> None:
        evt = {
            "type": "step",
            "id": step_id,
            "parent": parent,
            "label": label,
            "status": status,
        }
        evt.update(extra)
        await self._send(evt)

    async def run(
        self,
        query: str,
        max_iterations: int = 3,
        browser_mode: str = "NONE",
        enable_summarization: bool = False,
        gt_arxiv_ids: Optional[List[str]] = None,
    ) -> None:
        await self._send({
            "type": "status",
            "status": "thinking",
            "message": "Starting deep research...",
        })

        memory = ResearchMemory()
        memory.root_subquery_id = 0
        memory.root_text = query

        subquery_states: Dict[int, List[SubQueryState]] = {}
        all_selected: List[Paper] = []
        all_selected_ids: set = set()
        found_gt: set = set()
        gt_set = set(gt_arxiv_ids or [])
        last_checklist = ""

        for i in range(max_iterations):
            iteration_index = i + 1
            iid = f"it{i}"
            iter_start = time.time()
            await self._step(
                iid, None,
                f"Iteration {iteration_index}/{max_iterations}",
                variant="iteration", collapsible=True,
            )

            # ---- Planner ----
            pid = f"{iid}-plan"
            await self._step(pid, iid, "Planner: planning subqueries...",
                             variant="agent")
            try:
                subqueries_dict, experience, checklist, is_complete = \
                    self.planner.plan_iteration(
                        user_query={"query": query},
                        memory=memory,
                        subquery_states=subquery_states,
                        iteration_index=iteration_index,
                        idx=0,
                    )
            except Exception as e:
                logger.exception("Planner failed")
                await self._step(pid, iid, f"Planner failed: {e}",
                                 status="done", variant="error")
                await self._send({"type": "status", "status": "done",
                                  "message": f"Planner error: {e}"})
                return

            last_checklist = checklist or last_checklist
            memory.last_checklist = checklist
            memory.last_experience_replay = experience

            subqueries: List[SubQuery] = list(subqueries_dict.values()) \
                if isinstance(subqueries_dict, dict) else list(subqueries_dict)

            detail_lines = [
                f'  [{sq.link_type}] "{sq.text}" (k={sq.target_k})'
                for sq in subqueries
            ]
            detail_lines.append(f"  is_complete: {is_complete}")
            await self._step(
                pid, iid, f"Planner: {len(subqueries)} subqueries",
                status="done", variant="agent",
                detail="\n".join(detail_lines),
            )

            if is_complete or not subqueries:
                await self._step(
                    iid, None,
                    f"Iteration {iteration_index}/{max_iterations} (early stop)",
                    status="done", variant="iteration",
                    duration=round(time.time() - iter_start, 2),
                )
                break

            # ---- Retrieval ----
            rid = f"{iid}-ret"
            await self._step(
                rid, iid,
                f"Retrieving ({len(subqueries)} subqueries)...",
                variant="phase", collapsible=True,
            )
            method_label = "BM25" if self.rag.search_method == "bm25" else "Vector"
            ret_data: Dict[int, List[Paper]] = {}
            for sq in subqueries:
                sid = f"{rid}-{sq.id}"
                short = sq.text[:50] + ("..." if len(sq.text) > 50 else "")
                await self._step(sid, rid, f'{method_label}: "{short}"',
                                 variant="tool")
                try:
                    result = self.rag.search_citations(
                        query=sq.text,
                        top_k=sq.target_k,
                        search_method=self.rag.search_method,
                        debug=False,
                    )
                    papers = result[0] if isinstance(result, tuple) else result
                except Exception as e:
                    logger.warning(f"Retrieval failed for sq={sq.id}: {e}")
                    papers = []
                ret_data[sq.id] = papers
                await self._step(
                    sid, rid, f"{method_label}: {len(papers)} papers",
                    status="done", variant="tool",
                )

            total_ret = sum(len(p) for p in ret_data.values())
            await self._step(rid, iid, f"Retrieved {total_ret} papers total",
                             status="done", variant="phase")

            # ---- Summarizer (optional) ----
            if enable_summarization and total_ret > 0:
                smid = f"{iid}-sum"
                await self._step(
                    smid, iid,
                    f"Summarizer: compressing {total_ret} papers...",
                    variant="agent",
                )
                for sq in subqueries:
                    papers = ret_data.get(sq.id, [])
                    if not papers:
                        continue
                    try:
                        await self.summarizer.batch_summarize(
                            papers=papers,
                            user_query=query,
                            sub_query_text=sq.text,
                            iteration_index=iteration_index,
                            idx=0,
                        )
                    except Exception as e:
                        logger.warning(f"Summarization failed sq={sq.id}: {e}")
                await self._step(smid, iid, "Summarizer: done",
                                 status="done", variant="agent")

            # ---- Selector ----
            selid = f"{iid}-sel"
            await self._step(selid, iid, "Selector: filtering...",
                             variant="phase", collapsible=True)
            iter_to_browse: Dict[int, Dict] = {}
            for sq in subqueries:
                ssid = f"{selid}-{sq.id}"
                papers = ret_data.get(sq.id, [])
                short = sq.text[:50] + ("..." if len(sq.text) > 50 else "")
                await self._step(ssid, selid, f'Selector: "{short}"',
                                 variant="agent")
                try:
                    sel_result = await self.selector.decide_for_subquery(
                        user_query=query,
                        sub_query=sq,
                        planner_checklist=last_checklist,
                        papers=papers,
                        iteration_index=iteration_index,
                        idx=0,
                    )
                    if isinstance(sel_result, tuple) and len(sel_result) >= 2:
                        selected = sel_result[0] or []
                        overview = sel_result[1] if len(sel_result) > 1 else ""
                        to_browse = sel_result[2] if len(sel_result) > 2 else {}
                    else:
                        selected, overview, to_browse = [], "", {}
                except Exception as e:
                    logger.warning(f"Selection failed sq={sq.id}: {e}")
                    selected, overview, to_browse = [], "", {}

                # Update memory state for this subquery
                state = SubQueryState(
                    subquery=sq,
                    retrieved_papers=papers,
                    selected_papers=selected,
                    checklist=last_checklist,
                    selector_overview=overview or "",
                )
                subquery_states.setdefault(sq.id, []).append(state)

                for p in selected:
                    pid_key = p.arxiv_id or p.id
                    if pid_key and pid_key not in all_selected_ids:
                        all_selected_ids.add(pid_key)
                        all_selected.append(p)
                    if pid_key in gt_set:
                        found_gt.add(pid_key)
                iter_to_browse[sq.id] = to_browse or {}
                await self._step(
                    ssid, selid,
                    f"Selector: +{len(selected)}/{len(papers)}",
                    status="done", variant="agent",
                )
            await self._step(
                selid, iid,
                f"Selector: {len(all_selected)} selected so far",
                status="done", variant="phase",
            )

            # ---- Browser (optional) ----
            if browser_mode != "NONE":
                flat_to_browse = []
                for sq_id, mapped in iter_to_browse.items():
                    if not isinstance(mapped, dict):
                        continue
                    for p_key, info in mapped.items():
                        flat_to_browse.append((sq_id, p_key, info))
                if flat_to_browse:
                    bid = f"{iid}-br"
                    await self._step(
                        bid, iid,
                        f"Browser: fetching {len(flat_to_browse)} full texts...",
                        variant="agent",
                    )
                    success = 0
                    for sq_id, p_key, info in flat_to_browse:
                        if isinstance(info, dict):
                            paper = info.get("paper")
                            goal = info.get("goal") or info.get("task") or ""
                        else:
                            paper = None
                            goal = str(info) if info else ""
                        if paper is None:
                            continue
                        try:
                            sq_text = next(
                                (sq.text for sq in subqueries if sq.id == sq_id),
                                query,
                            )
                            await self.browser.browse_papers(
                                subquery_text=sq_text,
                                paper=paper,
                                iteration_index=iteration_index,
                                idx=0,
                                sq_id=sq_id,
                                paper_idx=0,
                                task=goal,
                            )
                            success += 1
                            pid_key = paper.arxiv_id or paper.id
                            if pid_key in gt_set:
                                found_gt.add(pid_key)
                            if pid_key and pid_key not in all_selected_ids:
                                all_selected_ids.add(pid_key)
                                all_selected.append(paper)
                        except Exception as e:
                            logger.warning(f"Browse failed: {e}")
                    await self._step(
                        bid, iid,
                        f"Browser: {success} fetched ({browser_mode})",
                        status="done", variant="agent",
                    )

            # ---- Memory ----
            mid = f"{iid}-mem"
            await self._step(
                mid, iid,
                f"Memory updated · {len(all_selected)} papers tracked",
                status="done", variant="info",
            )

            # ---- Metrics ----
            if gt_set:
                m = _simple_metrics(all_selected_ids, gt_set)
                await self._send({
                    "type": "metrics",
                    "parent": iid,
                    "metrics": m,
                    "found_gt": len(found_gt),
                    "total_gt": len(gt_set),
                })

            duration = round(time.time() - iter_start, 2)
            await self._step(
                iid, None,
                f"Iteration {iteration_index}/{max_iterations}",
                status="done", variant="iteration",
                duration=duration,
            )

        # ---- Final results ----
        papers_payload = [_paper_to_payload(p) for p in all_selected[:8]]
        await self._send({
            "type": "status",
            "status": "done",
            "message": f"Found {len(all_selected)} relevant papers",
        })
        await self._send({
            "type": "results",
            "query": query,
            "papers": papers_payload,
            "total_found": len(all_selected),
            "gt_found": len(found_gt),
            "gt_total": len(gt_set),
        })
