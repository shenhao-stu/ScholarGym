#!/usr/bin/env python3
"""
LLM API Interface for ScholarGym.

Provides unified _call_llm (sync) and _call_llm_async (async) functions
that support OpenAI-compatible APIs (OpenAI, DeepSeek, Qwen, GLM, Gemini, local Ollama, etc.).

Configuration:
    Set environment variables or modify the PROVIDER_CONFIG dict below.
    - OPENAI_API_KEY / OPENAI_BASE_URL  (for OpenAI, DeepSeek, Qwen cloud, etc.)
    - OLLAMA_URL                        (for local Ollama models)
"""

import os
import json
import hashlib
import asyncio
import logging
from pathlib import Path
from typing import Dict, Optional, Union, Tuple, Any

from openai import OpenAI, AsyncOpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM Cassette (record / replay) — behavior-preservation layer
# ---------------------------------------------------------------------------
# Set LLM_CASSETTE_MODE to:
#   "live"    (default) — call LLM normally, no recording
#   "record"  — call LLM normally AND write response to cassette file
#   "replay"  — read response from cassette file; raise if missing
#
# Cassettes are keyed by sha256(prompt+model+gen_params+thinking+structured),
# making pipeline behavior deterministic for refactor regression testing.

_CASSETTE_DIR = Path(__file__).parent / "tests" / "cassettes"
_CASSETTE_MODE = os.environ.get("LLM_CASSETTE_MODE", "live").lower()


def _cassette_key(prompt: str, model: str, gen_params: Dict,
                  enable_thinking: bool, return_structured: bool) -> str:
    payload = json.dumps(
        {
            "prompt": prompt,
            "model": model,
            "params": {
                "max_tokens": gen_params.get("max_tokens", 8192),
                "temperature": gen_params.get("temperature", 0),
                "top_p": gen_params.get("top_p", 1),
            },
            "thinking": enable_thinking,
            "structured": return_structured,
        },
        sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _cassette_load(key: str):
    """Returns cached response or None on miss. Re-hydrates tuples."""
    p = _CASSETTE_DIR / f"{key}.json"
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    resp = data["response"]
    if isinstance(resp, dict) and resp.get("__cassette_type__") == "tuple":
        return tuple(resp["items"])
    return resp


def _cassette_save(key: str, prompt: str, model: str, gen_params: Dict, response) -> None:
    _CASSETTE_DIR.mkdir(parents=True, exist_ok=True)
    serializable = response
    if isinstance(response, tuple):
        serializable = {"__cassette_type__": "tuple", "items": list(response)}
    with open(_CASSETTE_DIR / f"{key}.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": model,
                "prompt_preview": prompt[:200],
                "gen_params": gen_params,
                "response": serializable,
            },
            f, ensure_ascii=False, indent=2,
        )

# ---------------------------------------------------------------------------
# Provider Configuration
# ---------------------------------------------------------------------------
# Map model name prefixes to (api_key, base_url) pairs.
# Users should set the corresponding environment variables.
# For Ollama (local), no API key is needed.

PROVIDER_CONFIG = {
    # OpenAI / GPT models (gpt-*)
    "gpt": {
        "api_key": os.getenv("OPENAI_API_KEY", "your-openai-api-key"),
        "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    },
    # Claude models (claude-*)
    "claude": {
        "api_key": os.getenv("CLAUDE_API_KEY", "your-claude-api-key"),
        "base_url": os.getenv("CLAUDE_BASE_URL", "https://api.anthropic.com/v1"),
    },
    # DeepSeek models
    "deepseek": {
        "api_key": os.getenv("DEEPSEEK_API_KEY", "your-deepseek-api-key"),
        "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
    },
    # Qwen models (cloud API via DashScope-compatible endpoint)
    "qwen": {
        "api_key": os.getenv("DASHSCOPE_API_KEY", "your-dashscope-api-key"),
        "base_url": os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    },
    # GLM models (Zhipu AI)
    "glm": {
        "api_key": os.getenv("ZHIPU_API_KEY", "your-zhipu-api-key"),
        "base_url": os.getenv("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/"),
    },
    # MiniMax models
    "minimax": {
        "api_key": os.getenv("MINIMAX_API_KEY", "your-minimax-api-key"),
        "base_url": os.getenv("MINIMAX_BASE_URL", "https://api.minimax.chat/v1"),
    },
    # Google Gemini (via OpenAI-compatible endpoint)
    "gemini": {
        "api_key": os.getenv("GEMINI_API_KEY", "your-gemini-api-key"),
        "base_url": os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai"),
    },
    # Local Ollama (default)
    "ollama": {
        "api_key": "ollama",
        "base_url": os.getenv("OLLAMA_URL", "http://localhost:8001") + "/v1",
    },
}


def _resolve_provider(model_name: str, is_local: bool) -> Dict[str, str]:
    """Resolve the API provider config based on model name and is_local flag."""
    if is_local:
        return PROVIDER_CONFIG["ollama"]

    model_lower = model_name.lower()
    for prefix, cfg in PROVIDER_CONFIG.items():
        if prefix == "ollama":
            continue
        if model_lower.startswith(prefix):
            return cfg

    # Fallback: use OpenAI config
    logger.warning(f"No provider matched for model '{model_name}', falling back to OpenAI config.")
    return PROVIDER_CONFIG["gpt"]


def _build_messages(prompt: str) -> list:
    """Build chat messages from a prompt string.
    
    If the prompt contains a clear system/user split (separated by double newline
    after a [System] or similar marker), split accordingly. Otherwise treat the
    entire prompt as a user message.
    """
    return [{"role": "user", "content": prompt}]


def _call_llm(
    prompt: str,
    model: str,
    gen_params: Dict,
    is_local: bool = False,
    enable_thinking: bool = False,
    return_structured: bool = False,
    response_format: Any = None,
) -> Union[str, Tuple[str, str], Dict]:
    """Sync LLM call with optional cassette record/replay (see _CASSETTE_MODE)."""
    if _CASSETTE_MODE in ("replay", "record"):
        ck = _cassette_key(prompt, model, gen_params, enable_thinking, return_structured)
        if _CASSETTE_MODE == "replay":
            cached = _cassette_load(ck)
            if cached is None:
                raise RuntimeError(
                    f"Cassette miss in replay mode: key={ck} model={model} "
                    f"(prompt preview: {prompt[:80]!r})"
                )
            return cached
    result = _call_llm_impl(prompt, model, gen_params, is_local,
                            enable_thinking, return_structured, response_format)
    if _CASSETTE_MODE == "record":
        _cassette_save(ck, prompt, model, gen_params, result)
    return result


def _call_llm_impl(
    prompt: str,
    model: str,
    gen_params: Dict,
    is_local: bool = False,
    enable_thinking: bool = False,
    return_structured: bool = False,
    response_format: Any = None,
) -> Union[str, Tuple[str, str], Dict]:
    """
    Synchronous LLM call.

    Args:
        prompt: The full prompt string.
        model: Model name (e.g., 'qwen3-8b', 'gpt-5.2', 'deepseek-v3').
        gen_params: Generation parameters dict with keys like
                    'max_tokens', 'temperature', 'top_p', 'stream'.
        is_local: If True, route to local Ollama endpoint.
        enable_thinking: If True and the model supports it, return
                         (reasoning_content, response_content) tuple.
        return_structured: If True, parse response as JSON (structured output).
        response_format: Pydantic model or dict for structured output.

    Returns:
        - str: The response content (default).
        - Tuple[str, str]: (reasoning_content, response_content) if enable_thinking.
        - Dict: Parsed structured output if return_structured.
    """
    provider = _resolve_provider(model, is_local)
    client = OpenAI(api_key=provider["api_key"], base_url=provider["base_url"])

    messages = _build_messages(prompt)

    kwargs = {
        "model": model,
        "messages": messages,
        "max_tokens": gen_params.get("max_tokens", 8192),
        "temperature": gen_params.get("temperature", 0),
        "top_p": gen_params.get("top_p", 1),
        "stream": False,
    }

    # Structured output via response_format
    if return_structured and response_format is not None:
        try:
            result = client.beta.chat.completions.parse(
                **kwargs, response_format=response_format
            )
            parsed = result.choices[0].message.parsed
            return parsed.model_dump() if hasattr(parsed, "model_dump") else parsed
        except Exception as e:
            logger.warning(f"Structured output failed, falling back to text: {e}")
            # Fall through to normal text completion

    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        raise

    choice = response.choices[0]
    content = choice.message.content or ""

    # Handle reasoning/thinking content if available
    if enable_thinking:
        reasoning = getattr(choice.message, "reasoning_content", None)
        if reasoning:
            return reasoning, content

    return content


async def _call_llm_async(
    prompt: str,
    model: str,
    gen_params: Dict,
    is_local: bool = False,
    enable_thinking: bool = False,
    return_structured: bool = False,
    response_format: Any = None,
) -> Union[str, Tuple[str, str], Dict]:
    """Async LLM call with optional cassette record/replay (see _CASSETTE_MODE)."""
    if _CASSETTE_MODE in ("replay", "record"):
        ck = _cassette_key(prompt, model, gen_params, enable_thinking, return_structured)
        if _CASSETTE_MODE == "replay":
            cached = _cassette_load(ck)
            if cached is None:
                raise RuntimeError(
                    f"Cassette miss in replay mode: key={ck} model={model} "
                    f"(prompt preview: {prompt[:80]!r})"
                )
            return cached
    result = await _call_llm_async_impl(prompt, model, gen_params, is_local,
                                        enable_thinking, return_structured, response_format)
    if _CASSETTE_MODE == "record":
        _cassette_save(ck, prompt, model, gen_params, result)
    return result


async def _call_llm_async_impl(
    prompt: str,
    model: str,
    gen_params: Dict,
    is_local: bool = False,
    enable_thinking: bool = False,
    return_structured: bool = False,
    response_format: Any = None,
) -> Union[str, Tuple[str, str], Dict]:
    """
    Asynchronous LLM call. Same interface as _call_llm but async.

    Args:
        prompt: The full prompt string.
        model: Model name.
        gen_params: Generation parameters dict.
        is_local: If True, route to local Ollama endpoint.
        enable_thinking: If True, return (reasoning, content) tuple.
        return_structured: If True, parse response as JSON.
        response_format: Pydantic model or dict for structured output.

    Returns:
        Same as _call_llm.
    """
    provider = _resolve_provider(model, is_local)
    client = AsyncOpenAI(api_key=provider["api_key"], base_url=provider["base_url"])

    messages = _build_messages(prompt)

    kwargs = {
        "model": model,
        "messages": messages,
        "max_tokens": gen_params.get("max_tokens", 8192),
        "temperature": gen_params.get("temperature", 0),
        "top_p": gen_params.get("top_p", 1),
        "stream": False,
    }

    # Structured output
    if return_structured and response_format is not None:
        try:
            result = await client.beta.chat.completions.parse(
                **kwargs, response_format=response_format
            )
            parsed = result.choices[0].message.parsed
            return parsed.model_dump() if hasattr(parsed, "model_dump") else parsed
        except Exception as e:
            logger.warning(f"Async structured output failed, falling back to text: {e}")

    try:
        response = await client.chat.completions.create(**kwargs)
    except Exception as e:
        logger.error(f"Async LLM call failed: {e}")
        raise

    choice = response.choices[0]
    content = choice.message.content or ""

    if enable_thinking:
        reasoning = getattr(choice.message, "reasoning_content", None)
        if reasoning:
            return reasoning, content

    return content
