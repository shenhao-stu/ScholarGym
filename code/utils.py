import json
import re
import os
from typing import Dict, List, Any, Optional, Set, Tuple
from logger import get_logger

logger = get_logger(__name__, log_file='./log/utils.log')

def remove_think_blocks(text: str) -> str:
    """Remove all <think>...</think> or <thought>...</thought> blocks from text."""
    return re.sub(r"<think>.*?</think>|<thought>.*?</thought>", "", text, flags=re.DOTALL)

def parse_xml_tag(response: str, tag: str) -> str:
    """Extracts content from a single XML tag."""
    match = re.search(f'<{tag}>(.*?)</{tag}>', response, re.DOTALL)
    return match.group(1).strip() if match else ""

def parse_json_from_tag(response: str, tag: str) -> Optional[Dict]:
    """
    Parses JSON object from various formats.
    """
    # Remove <think> blocks first
    response = remove_think_blocks(response)
    
    # Try XML tag extraction first
    content = parse_xml_tag(response, tag)

    # If no tag found, try to extract from code blocks or direct JSON
    if not content:
        # Try to find ```json code blocks
        json_block_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_block_match:
            content = json_block_match.group(1).strip()
        else:
            # Try to find direct JSON object (starts with {, ends with })
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                content = json_match.group(0).strip()
            else:
                return None
    
    # Clean and parse the content
    content = content.strip()
    content = content.replace("```json", "").replace("```", "").strip()
    
    try:
        return json.loads(content)
    except Exception as e:
        logger.warning(f"Failed to parse JSON from response: {e}")
        return None

def parse_response_to_keys(response: str) -> List[str]:
    """
    Parse LLM response into individual query keys, expecting an XML-like format.
    
    Args:
        response (str): LLM response containing query keys
        
    Returns:
        List[str]: List of parsed query keys
    """
    # Use regex to find content within <sub_queries> tag
    match = re.search(r'<sub_queries>(.*?)</sub_queries>', response, re.DOTALL)
    if not match:
        # Fallback for older format or malformed response
        cleaned_response = re.sub(r'<.*?>', '', response, flags=re.DOTALL)
        lines = cleaned_response.strip().split('\n')
    else:
        content = match.group(1).strip()
        lines = content.split('\n')

    keys = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Clean up potential list formatting artifacts
        line = re.sub(r'^\d+\.?\s*', '', line)
        line = re.sub(r'^[-•*]\s*', '', line)
        
        cleaned_line = line.strip()
        
        # Basic validation for key quality
        if 3 <= len(cleaned_line) <= 100:
            keys.append(cleaned_line)
    
    return keys


def extract_ground_truth_titles(ground_truth_papers: List[Dict], gt_labels: List[int]) -> set:
    """
    Extract and normalize ground truth paper titles.
    
    Args:
        ground_truth_papers (List[Dict]): List of ground truth papers
        gt_labels (List[int]): List of labels (1 for relevant, 0 for not)
        
    Returns:
        set: Set of ground truth paper titles
    """
    return {
        paper['title'] 
        for paper, label in zip(ground_truth_papers, gt_labels) 
        if label == 1 and 'title' in paper and paper['title'] != "Unknown"
    }


def extract_ground_truth_arxiv_ids(ground_truth_papers: List[Dict], gt_labels: List[int]) -> set:
    """
    Extract ground truth paper arxiv_ids.
    
    Args:
        ground_truth_papers (List[Dict]): List of ground truth papers
        gt_labels (List[int]): List of labels (1 for relevant, 0 for not)
        
    Returns:
        set: Set of ground truth arXiv IDs
    """
    return {
        paper['arxiv_id'] 
        for paper, label in zip(ground_truth_papers, gt_labels) 
        if label == 1 and 'arxiv_id' in paper and paper['arxiv_id']
    }


def combine_search_results(all_search_results: List[List[tuple]]) -> List[tuple]:
    """
    Combine and deduplicate search results from multiple queries.
    
    Args:
        all_search_results (List[List[tuple]]): List of search results, each is a list of (paper_id, similarity, paper_info)
        
    Returns:
        List[tuple]: Combined and sorted list of (paper_id, similarity, paper_info)
    """
    paper_scores = {}
    paper_info_map = {}
    
    for search_results in all_search_results:
        for paper_id, similarity, paper_info in search_results:
            if paper_id not in paper_scores or similarity > paper_scores[paper_id]:
                paper_scores[paper_id] = similarity
                paper_info_map[paper_id] = paper_info
    
    sorted_papers = sorted(paper_scores.items(), key=lambda x: x[1], reverse=True)
    
    return [
        (paper_id, paper_scores[paper_id], paper_info_map[paper_id])
        for paper_id, _ in sorted_papers
    ]


def calculate_retrieval_metrics(
    gt_arxiv_ids: Set[str],
    retrieved_arxiv_ids: Set[str],
    selected_arxiv_ids: Set[str] = None
) -> Dict[str, float]:
    """
    Calculate recall and precision metrics for retrieval and selection.
    
    Args:
        gt_arxiv_ids: Set of ground truth arXiv IDs
        retrieved_arxiv_ids: Set of retrieved arXiv IDs (before selection)
        selected_arxiv_ids: Set of selected arXiv IDs (after selection, optional)
        
    Returns:
        Dict containing recall and precision metrics
    """
    metrics = {}
    
    # Retrieval metrics
    retrieved_matches = len(gt_arxiv_ids.intersection(retrieved_arxiv_ids))
    metrics['retrieval_recall'] = retrieved_matches / len(gt_arxiv_ids) if gt_arxiv_ids else 0.0
    metrics['retrieval_precision'] = retrieved_matches / len(retrieved_arxiv_ids) if retrieved_arxiv_ids else 0.0
    metrics['retrieval_matches'] = retrieved_matches
    metrics['retrieved_count'] = len(retrieved_arxiv_ids)
    
    # Selection metrics (if provided)
    if selected_arxiv_ids is not None:
        selected_matches = len(gt_arxiv_ids.intersection(selected_arxiv_ids))
        metrics['recall'] = selected_matches / len(gt_arxiv_ids) if gt_arxiv_ids else 0.0
        metrics['precision'] = selected_matches / len(selected_arxiv_ids) if selected_arxiv_ids else 0.0
        metrics['matches'] = selected_matches
        metrics['selected_count'] = len(selected_arxiv_ids)
    
    return metrics
    
def clean_text_content(text: str) -> str:
    """
    Clean text content by removing newlines and extra whitespace.
    
    Args:
        text (str): Raw text content to clean
        
    Returns:
        str: Cleaned text content
    """
    if not text:
        return ""
    
    # Remove newline characters that break words (e.g., "high-\nlevel" -> "high-level")
    text = re.sub(r'-\n\s*', '-', text)
    text = re.sub(r'\n\s*-', '-', text)
    
    # Remove all remaining newlines and extra whitespace
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    return text.strip()

def parse_bib(bib_content: str) -> Dict[str, str]:
    pattern = re.compile(
        r"@\w+\{([^,]+),"  # Capture the key
        r"(?:.|\n)*?"  # Non-greedy match until title
        r"title\s*=\s*\{((?:[^{}]|\{[^{}]*\})+)\}",  # Capture title with nested braces
        re.IGNORECASE,
    )

    result = []
    matches = pattern.finditer(bib_content)
    for match in matches:
        key = match.group(1).strip()
        title = match.group(2).strip()
        result.append({
            'ID': key,
            'title': title
        })
    return result


def parse_authors_parsed(authors_parsed: List[List[str]]) -> List[str]:
    """
    Parse authors_parsed field into a list of formatted author names.
    
    Args:
        authors_parsed (List[List[str]]): List of author info [last_name, first_name, suffix]
        
    Returns:
        List[str]: List of formatted author names like "First Last" or "First Last Suffix"
    """
    if not authors_parsed:
        return []
    
    formatted_authors = []
    for author_info in authors_parsed:
        if not author_info:
            continue
        
        # Extract and clean the parts
        parts = [part.strip() for part in author_info if part.strip()]
        
        if len(parts) == 1:
            # Single name (could be full name or just last name)
            formatted_authors.append(parts[0])
        elif len(parts) >= 2:
            # Standard format: [last_name, first_name, suffix...]
            # Reorder to "First Last Suffix"
            name_parts = [parts[1], parts[0]] + parts[2:]  # first, last, suffix...
            formatted_authors.append(' '.join(name_parts))
    
    return formatted_authors


def generate_arxiv_url(arxiv_id: str) -> str:
    """
    Generate arXiv PDF URL from arXiv ID.
    
    Args:
        arxiv_id (str): arXiv paper ID (e.g., "0704.0001")
        
    Returns:
        str: Complete arXiv PDF URL
    """
    return f"http://arxiv.org/pdf/{arxiv_id}"


def process_arxiv_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single arXiv metadata entry by cleaning and transforming fields.
    
    Args:
        entry (Dict[str, Any]): Raw arXiv metadata entry
        
    Returns:
        Dict[str, Any]: Processed arXiv metadata entry
    """
    processed_entry = entry.copy()
    
    # Clean title and abstract
    if 'title' in processed_entry:
        processed_entry['title'] = clean_text_content(processed_entry['title'])
    
    if 'abstract' in processed_entry:
        processed_entry['abstract'] = clean_text_content(processed_entry['abstract'])
    
    # Use authors_parsed field instead of authors field for better accuracy
    if 'authors_parsed' in processed_entry:
        processed_entry['authors'] = parse_authors_parsed(processed_entry['authors_parsed'])
        # Remove the authors_parsed field since we've converted it to authors
    elif 'authors' in processed_entry and isinstance(processed_entry['authors'], str):
        # Fallback to parsing authors string if authors_parsed is not available
        processed_entry['authors'] = [author.strip() for author in processed_entry['authors'].split(',')]
    
    # Add URL field
    if 'id' in processed_entry:
        processed_entry['url'] = generate_arxiv_url(processed_entry['id'])
    
    return processed_entry


def process_arxiv_jsonl_file(input_file: str, output_file: str) -> None:
    """
    Process an entire arXiv JSONL file and save the cleaned results.
    
    Args:
        input_file (str): Path to input JSONL file
        output_file (str): Path to output JSONL file
    """
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                # Parse JSON entry
                entry = json.loads(line)
                
                # Process the entry
                processed_entry = process_arxiv_entry(entry)
                
                # Write processed entry to output file
                json.dump(processed_entry, outfile, ensure_ascii=False)
                outfile.write('\n')
                
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON at line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Error processing entry at line {line_num}: {e}")
                continue


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
            search_method: Search method (e.g., 'vector', 'bm25', 'hybrid')
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


class CheckpointManager:
    """
    Manages checkpoint loading, saving, and statistics rebuilding for evaluation.
    """
    
    def __init__(self, checkpoint_file: str):
        """
        Args:
            checkpoint_file: Path to the checkpoint JSONL file
        """
        self.checkpoint_file = checkpoint_file
        self.processed_indices: Set[int] = set()
        self.cached_results: List[Dict] = []
    
    def load_checkpoint(self) -> Tuple[Set[int], List[Dict]]:
        """
        Load checkpoint from file and return processed indices and cached results.
        
        Returns:
            Tuple of (processed_indices, cached_results)
        """
        if not os.path.exists(self.checkpoint_file):
            logger.info(f"[📝] No checkpoint found, starting fresh")
            return set(), []
        
        logger.info(f"[📂] Loading checkpoint: {self.checkpoint_file}")
        processed_indices = set()
        cached_results = []
        
        with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    result = json.loads(line.strip())
                    idx = result.get('idx', -1)
                    if idx >= 0:
                        processed_indices.add(idx)
                        cached_results.append(result)
        
        self.processed_indices = processed_indices
        self.cached_results = cached_results
        logger.info(f"[✓] Loaded {len(processed_indices)} processed queries from checkpoint")
        
        return processed_indices, cached_results
    
    def rebuild_statistics(
        self, 
        results: Dict, 
        workflow: str, 
        top_k_list: List[int] = None,
        max_iterations: int = 3
    ) -> Dict:
        """
        Rebuild statistics from cached results.
        
        Args:
            results: Results dictionary to update
            workflow: 'simple' or 'deep_research'
            top_k_list: Top-k values for simple workflow
            max_iterations: Max iterations for deep_research workflow
            
        Returns:
            Updated results dictionary
        """
        if not self.cached_results:
            return results
        
        logger.info(f"[📊] Rebuilding statistics from {len(self.cached_results)} cached results...")
        
        for query_result in self.cached_results:
            results['successful_queries'] += 1
            results['detailed_results'].append(query_result)
            
            if workflow == 'deep_research':
                self._rebuild_deep_research_stats(query_result, results, max_iterations)
            else:  # simple workflow
                self._rebuild_simple_stats(query_result, results, top_k_list)
        
        return results
    
    def _rebuild_deep_research_stats(
        self, 
        query_result: Dict, 
        results: Dict, 
        max_iterations: int
    ):
        """Rebuild statistics for deep_research workflow."""
        # Collect metrics by iteration
        metric_names = ['recall', 'precision', 'retrieval_recall', 'retrieval_precision']
        metrics_by_iter = {name: {} for name in metric_names}
        
        for res in query_result['iteration_results']:
            iter_idx = res['iter_idx']
            for metric_name in metric_names:
                if metric_name in res:
                    metrics_by_iter[metric_name][iter_idx] = res[metric_name]
        
        # Fill missing iterations with last value
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
        
        # Timing statistics
        # Keep consistent with eval.py so aggregated `avg_*` timing metrics are available
        # (e.g., browser_during for evaluation_summary.jsonl).
        for phase in ['planner', 'retrieval', 'selector', 'browser', 'overhead', 'total']:
            phase_key = f'{phase}_during'
            if phase_key not in results:
                results[phase_key] = []
            for res in query_result['iteration_results']:
                time_val = res.get(phase_key, -1)
                if time_val >= 0:
                    results[phase_key].append(time_val)
        
        # Distance statistics
        for res in query_result['iteration_results']:
            iter_idx = res['iter_idx']
            distance_key = f'avg_distance_iter_{iter_idx}'
            if distance_key not in results:
                results[distance_key] = []
            avg_distance = res.get('avg_distance', -1)
            if avg_distance >= 0:
                results[distance_key].append(avg_distance)
        
        # Discarded statistics
        for res in query_result['iteration_results']:
            iter_idx = res['iter_idx']
            iteration_metrics = res.get('iteration_metrics', {})
            
            ratio_key = f'discarded_ratio_iter_{iter_idx}'
            if ratio_key not in results:
                results[ratio_key] = []
            discarded_ratio = iteration_metrics.get('discarded_ratio', -1)
            if discarded_ratio >= 0:
                results[ratio_key].append(discarded_ratio)
            
            count_key = f'discarded_total_count_iter_{iter_idx}'
            if count_key not in results:
                results[count_key] = []
            discarded_count = iteration_metrics.get('discarded_total_count', -1)
            if discarded_count >= 0:
                results[count_key].append(discarded_count)
    
    def _rebuild_simple_stats(
        self, 
        query_result: Dict, 
        results: Dict, 
        top_k_list: List[int]
    ):
        """Rebuild statistics for simple workflow."""
        gt_arxiv_ids = set(query_result['ground_truth_arxiv_ids'])
        retrieved_arxiv_ids = [res['arxiv_id'] for res in query_result['top_results']]
        
        for k in top_k_list:
            top_k_arxiv_ids = set(retrieved_arxiv_ids[:k])
            matches_k = len(gt_arxiv_ids.intersection(top_k_arxiv_ids))
            recall_k = matches_k / len(gt_arxiv_ids) if gt_arxiv_ids else 0.0
            precision_k = matches_k / len(top_k_arxiv_ids) if top_k_arxiv_ids else 0.0
            
            results[f'recall@{k}'].append(recall_k)
            results[f'precision@{k}'].append(precision_k)
    
    def append_result(self, result: Dict):
        """
        Append a single query result to checkpoint file.
        
        Args:
            result: Query result dictionary
        """
        os.makedirs(os.path.dirname(self.checkpoint_file), exist_ok=True)
        with open(self.checkpoint_file, 'a', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False)
            f.write('\n')
        
        # Update internal cache
        idx = result.get('idx', -1)
        if idx >= 0:
            self.processed_indices.add(idx)
            self.cached_results.append(result)
    
    def is_processed(self, idx: int) -> bool:
        """Check if a query index has been processed."""
        return idx in self.processed_indices


# Example usage
if __name__ == "__main__":
    # Process the arXiv metadata file
    # input_file = "datasets/arxiv/arxiv-metadata-250704.jsonl"
    # output_file = "datasets/arxiv/cleaned-arxiv-metadata-250704.jsonl"
    
    # print("Processing arXiv metadata file...")
    # process_arxiv_jsonl_file(input_file, output_file)
    # print(f"Processed file saved to: {output_file}") 

    with open("datasets/survey/tex/arXiv-2404.04925/custom.bib", "r", encoding="utf-8") as f:
        content = f.read()
    result = parse_bib(content)
    print(result)