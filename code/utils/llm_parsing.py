"""LLM response parsing helpers (think blocks, XML tags, JSON extraction)."""
import json
import re
from typing import Dict, List, Optional

from logger import get_logger

logger = get_logger(__name__, log_file='./log/utils.log')


def remove_think_blocks(text: str) -> str:
    """Remove all <think>...</think> or <thought>...</thought> blocks from text."""
    return re.sub(r"<think>.*?</think>|<thought>.*?</thought>", "", text, flags=re.DOTALL)


def parse_xml_tag(response: str, tag: str) -> str:
    """Extracts content from a single XML tag."""
    match = re.search(f'<{tag}>(.*?)</{tag}>', response, re.DOTALL)
    return match.group(1).strip() if match else ""


def _extract_outermost_json(text: str) -> Optional[str]:
    """Extract the outermost balanced {...} from text.

    Uses a brace-depth counter instead of regex to avoid catastrophic
    backtracking on large responses.  Returns the matched substring or None.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


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
            json_match = _extract_outermost_json(response)
            if json_match:
                content = json_match.strip()
            else:
                return None

    # Clean and parse the content
    content = content.strip()
    content = content.replace("```json", "").replace("```", "").strip()

    for strict in (True, False):
        # strict=False 放行 string 里未转义的换行/控制字符（Gemma-4 小模型爱犯）
        try:
            return json.loads(content, strict=strict)
        except json.JSONDecodeError:
            # "Extra data" 情况：qwen3.5-9b think 有时会在合法 JSON 后拖尾
            # 第二段 JSON 或废话。用 raw_decode 只吃第一段合法 JSON。
            try:
                obj, _ = json.JSONDecoder(strict=strict).raw_decode(content)
                return obj
            except Exception:
                continue
        except Exception as e:
            logger.warning(f"Failed to parse JSON from response: {e}")
            return None
    logger.warning(f"Failed to parse JSON from response (both strict modes failed)")
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
