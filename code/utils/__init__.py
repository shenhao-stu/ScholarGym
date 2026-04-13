"""ScholarGym utilities package.

Re-exports all symbols so existing `from utils import X` call sites keep
working after the split into submodules by concern:

  - llm_parsing    : LLM response parsing (think blocks, XML tags, JSON)
  - metrics_helpers: GT extraction, result combination, retrieval metrics
  - trace          : AgentTraceRecorder
  - checkpoint     : CheckpointManager
"""
from utils.llm_parsing import (  # noqa: F401
    remove_think_blocks,
    parse_xml_tag,
    _extract_outermost_json,
    parse_json_from_tag,
    parse_response_to_keys,
)
from utils.metrics_helpers import (  # noqa: F401
    extract_ground_truth_arxiv_ids,
    combine_search_results,
    calculate_retrieval_metrics,
)
from utils.trace import AgentTraceRecorder  # noqa: F401
from utils.checkpoint import CheckpointManager  # noqa: F401
