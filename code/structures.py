from dataclasses import dataclass, field
from typing import Optional, List, Dict
from pydantic import BaseModel


@dataclass
class Paper:
    id: str
    title: str
    abstract: str
    arxiv_id: Optional[str] = None
    summary: Optional[str] = None
    date: Optional[str] = None
    score: Optional[float] = None  # Retriever score for selector context
    browsing_content: Optional[str] = None  # Content fetched by the browser tool


@dataclass
class SubQuery:
    id: int
    text: str
    before_date: Optional[str] = None
    target_k: int = 10  # Planner-specified number of results to retrieve
    # Linking metadata in a DAG rooted at the user query (root id = 0)
    # link_type must be one of: "continue", "derive", "expand"
    link_type: Optional[str] = None
    source_subquery_id: Optional[int] = None  # link to the parent subquery id in the tree
    # Iteration index when this subquery was created (1-based)
    iter_index: int = 0


@dataclass
class SubQueryState:
    """State tracked in research memory for each subquery."""
    subquery: SubQuery
    retrieved_papers: List[Paper] = field(default_factory=list)
    selected_papers: List[Paper] = field(default_factory=list)
    checklist: Optional[str] = None
    # Track how many results have been requested cumulatively for pagination
    total_requested: int = 0
    selector_overview: str = ""


@dataclass
class ResearchMemory:
    """Snapshot of planner/selector memory exchanged between iterations."""
    last_experience_replay: Optional[str] = None
    last_checklist: Optional[str] = None
    # Optional selector recipe for guidance
    selector_recipe: Optional[str] = None
    # DAG history of subquery ids in execution order
    subqueries_dag: List[int] = field(default_factory=list)
    # Consolidated metadata for each subquery id
    # id -> {"text": str, "source": Optional[int], "iter": int}
    subqueries_meta: Dict[int, Dict] = field(default_factory=dict)
    # Root metadata
    root_subquery_id: int = 0
    root_text: Optional[str] = None
    # Full history of planner outputs (JSON strings/dicts) for ablation studies
    planner_history: List[str] = field(default_factory=list)


# ==================== Pydantic Models for Structured Output ====================

class SubQueryItem(BaseModel):
    """Single subquery item in planner output."""
    link_type: str  # "continue", "derive", or "expand"
    source_id: int
    text: Optional[str] = None
    target_k: int


class PlannerOutput(BaseModel):
    """Structured output format for Planner agent."""
    subqueries: List[SubQueryItem]
    checklist: str
    experience_replay: str
    is_complete: bool

class PlannerOutputABLATION(BaseModel):
    """Structured output format for Planner agent."""
    reason: str
    subqueries: List[SubQueryItem]
    checklist: str
    is_complete: bool

class SelectorOutputWithBrowser(BaseModel):
    """Minimal structured output for Selector agent."""
    selected: List[str]
    discarded: List[str]
    to_browse: List[dict]
    reasons: Dict[str, str]
    overview: str
    
class SelectorOutput(BaseModel):
    """Minimal structured output for Selector agent."""
    selected: List[str]
    reasons: Dict[str, str]
    overview: str