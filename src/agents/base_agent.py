"""Base agent class for all specialized agents."""

from abc import ABC, abstractmethod
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from src.config import settings
import logging
from typing import Any, Dict
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log

logger = logging.getLogger(__name__)


class Agent(ABC):
    """Abstract base class for all agents.
    
    Provides LLM initialization, retry-wrapped invocation, and logging.
    """
    
    def __init__(self, name: str, model: str = None):
        """
        Initialize an agent with LLM.
        
        Args:
            name: Agent name
            model: Model identifier (defaults to config)
        """
        self.name = name
        self.model = model or settings.llm_model
        self.llm = self._initialize_llm()
        logger.info(f"Initialized {self.name} agent with model {self.model}")
    
    def _initialize_llm(self) -> ChatNVIDIA:
        """Initialize NVIDIA API LLM client with timeout."""
        return ChatNVIDIA(
            model=self.model,
            api_key=settings.nvidia_api_key,
            base_url=settings.openai_base_url,
            max_tokens=4096,
            temperature=settings.llm_temperature,
            timeout=90,
        )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def invoke_llm_with_retry(self, chain, input_data: dict) -> Any:
        """Invoke an LLM chain with automatic retry on transient failures.
        
        Retries up to 3 times with exponential backoff (2s, 4s, 8s).
        Logs each retry attempt at WARNING level.
        """
        return chain.invoke(input_data)
    
    @abstractmethod
    def execute(self, state: Any, user_context: dict | None = None) -> Any:
        """
        Execute the agent's task.
        
        Args:
            state: Current application state
            user_context: Optional dict for future auth / RBAC injection (no-op now).
            
        Returns:
            Updated state
        """
        pass
    
    def _log_action(self, action: str, details: Dict[str, Any] = None):
        """Log agent actions for visibility."""
        msg = f"[{self.name}] {action}"
        if details:
            msg += f" - {details}"
        logger.info(msg)
