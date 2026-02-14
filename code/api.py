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
import asyncio
import logging
from typing import Dict, Optional, Union, Tuple, Any

from openai import OpenAI, AsyncOpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider Configuration
# ---------------------------------------------------------------------------
# Map model name prefixes to (api_key, base_url) pairs.
# Users should set the corresponding environment variables.
# For Ollama (local), no API key is needed.

PROVIDER_CONFIG = {
    # OpenAI models (gpt-*)
    "gpt": {
        "api_key": os.getenv("OPENAI_API_KEY", "your-openai-api-key"),
        "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
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
        "base_url": os.getenv("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
    },
    # Google Gemini (via OpenAI-compatible endpoint)
    "gemini": {
        "api_key": os.getenv("GEMINI_API_KEY", "your-gemini-api-key"),
        "base_url": os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai"),
    },
    # Local Ollama (default)
    "ollama": {
        "api_key": "ollama",
        "base_url": os.getenv("OLLAMA_URL", "http://localhost:11434") + "/v1",
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
