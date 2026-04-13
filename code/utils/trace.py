"""AgentTraceRecorder — persists agent intermediate traces to JSONL for debugging."""
import json
import os
from typing import Any, Dict

from logger import get_logger

logger = get_logger(__name__, log_file='./log/utils.log')


class AgentTraceRecorder:
    """
    Records all agent intermediate traces (prompts, responses, reasoning) for debugging and analysis.
    Each sample is saved as a JSON record containing all agent stages.
    """

    def __init__(
        self,
        output_dir: str,
        model_name: str,
        prompt_type: str,
        search_method: str,
        workflow: str,
        top_k: int,
        max_results: int,
        enable_reasoning: bool,
        enable_structured: bool
    ):
        """
        Initialize the trace recorder with configuration parameters.

        Args:
            output_dir: Base output directory
            model_name: LLM model name
            prompt_type: Prompt type (e.g., 'complex', 'simple')
            search_method: Search method (e.g., 'vector', 'bm25')
            workflow: Workflow type (e.g., 'simple', 'deep_research')
            top_k: Top-k value for retrieval
            max_results: Maximum results per query
            enable_reasoning: Whether reasoning mode is enabled
            enable_structured: Whether structured output is enabled
        """
        reasoning_flag = "reasoning" if enable_reasoning else "instruct"
        structured_flag = "structured" if enable_structured else "non-structured"
        model_name = model_name.split('/')[-1] if '/' in model_name else model_name
        # Create trace directory following the same naming convention
        self.trace_dir = os.path.join(
            output_dir,
            f"{model_name}_{prompt_type}_{search_method}_{workflow}_topk-{top_k}_maxq-{max_results}_{reasoning_flag}_{structured_flag}",
            "agent_traces"
        )
        os.makedirs(self.trace_dir, exist_ok=True)

        self.trace_file = os.path.join(self.trace_dir, "traces.jsonl")
        self.sample_traces: Dict[int, Dict] = {}  # {sample_idx: trace_data}

        logger.info(f"[📝] Agent trace recorder initialized: {self.trace_file}")

    def record_stage(
        self,
        sample_idx: int,
        agent_name: str,
        stage_data: Dict[str, Any]
    ):
        """
        Record a single agent stage.

        Args:
            sample_idx: Sample/query index
            agent_name: Name of the agent (e.g., 'planner', 'selector', 'summarizer')
            stage_data: Dictionary containing stage information:
                - iteration: Iteration index (for iterative workflows)
                - sub_query_id: Sub-query ID (for selector)
                - prompt: Input prompt
                - response: Output response
                - reasoning: Reasoning content (optional)
                - timestamp: Timestamp (optional)
                - extra: Any additional metadata
        """
        if sample_idx not in self.sample_traces:
            self.sample_traces[sample_idx] = {
                'sample_idx': sample_idx,
                'stages': []
            }

        # Create stage record
        stage_record = {
            'agent': agent_name,
            **stage_data
        }

        self.sample_traces[sample_idx]['stages'].append(stage_record)

    def save_sample(self, sample_idx: int):
        """
        Save a single sample's traces to the JSONL file.

        Args:
            sample_idx: Sample index to save
        """
        if sample_idx not in self.sample_traces:
            logger.warning(f"[⚠️] No traces found for sample {sample_idx}")
            return

        try:
            with open(self.trace_file, 'a', encoding='utf-8') as f:
                json.dump(self.sample_traces[sample_idx], f, ensure_ascii=False)
                f.write('\n')

            logger.debug(f"[💾] Saved traces for sample {sample_idx}")

            # Clear from memory to save space
            del self.sample_traces[sample_idx]

        except IOError as e:
            logger.error(f"[💢] Failed to save traces for sample {sample_idx}: {e}")

    def save_all(self):
        """Save all remaining traces to the JSONL file."""
        for sample_idx in list(self.sample_traces.keys()):
            self.save_sample(sample_idx)

        logger.info(f"[✓] All traces saved to {self.trace_file}")
