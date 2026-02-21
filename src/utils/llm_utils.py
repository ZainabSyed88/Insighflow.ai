"""Shared LLM call wrapper with retry logic, model selection, and prompt templating."""

import logging
import json
import signal
import threading
from typing import Any, Optional
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log
from src.config import settings

logger = logging.getLogger(__name__)

# Default timeout for any single LLM call (seconds)
LLM_CALL_TIMEOUT = 90


def get_llm(model: str | None = None, temperature: float | None = None, max_tokens: int = 4096) -> ChatNVIDIA:
    """Create a ChatNVIDIA instance with optional model override."""
    return ChatNVIDIA(
        model=model or settings.llm_model,
        api_key=settings.nvidia_api_key,
        base_url=settings.openai_base_url,
        max_tokens=max_tokens,
        temperature=temperature if temperature is not None else settings.llm_temperature,
        timeout=LLM_CALL_TIMEOUT,
    )


def _invoke_with_timeout(chain, input_data: dict, timeout: int = LLM_CALL_TIMEOUT):
    """Invoke a LangChain chain with a thread-based timeout.

    Raises TimeoutError if the call exceeds *timeout* seconds.
    """
    result_holder: list = []
    error_holder: list = []

    def _target():
        try:
            result_holder.append(chain.invoke(input_data))
        except Exception as e:
            error_holder.append(e)

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        logger.error(f"LLM call timed out after {timeout}s")
        raise TimeoutError(f"LLM call did not complete within {timeout}s")

    if error_holder:
        raise error_holder[0]

    return result_holder[0]


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def call_llm(
    prompt_template: ChatPromptTemplate,
    input_data: dict,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int = 4096,
    timeout: int = LLM_CALL_TIMEOUT,
) -> str:
    """Invoke an LLM chain with retry + timeout and return the raw content string."""
    llm = get_llm(model=model, temperature=temperature, max_tokens=max_tokens)
    chain = prompt_template | llm
    logger.info(f"LLM call starting (model={model or settings.llm_model}, timeout={timeout}s)")
    response = _invoke_with_timeout(chain, input_data, timeout=timeout)
    logger.info("LLM call completed")
    return response.content


def call_llm_json(
    prompt_template: ChatPromptTemplate,
    input_data: dict,
    model: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 4096,
) -> dict:
    """Invoke LLM and parse the response as JSON.

    Handles common markdown code fences around JSON.
    """
    content = call_llm(
        prompt_template, input_data,
        model=model, temperature=temperature, max_tokens=max_tokens,
    )
    # Strip code fences
    text = content.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return json.loads(text.strip())


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token)."""
    return len(text) // 4
