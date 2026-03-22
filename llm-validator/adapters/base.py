adapters/base.py — Abstract base class for all LLM adapters.

Every adapter (LiteLLM, OpenAI direct, Anthropic direct, Ollama, etc.)
must subclass BaseAdapter and implement its abstract methods.

This gives the runner and evaluators a stable contract to program against,
regardless of which provider is underneath.

Lifecycle
---------
1. Instantiate:   adapter = MyAdapter(model="...", **config)
2. (Optional)     adapter.health_check()  — verify connectivity before a run
3. Per test:      response = adapter.complete(prompt, system=...)
4. Metadata:      adapter.model_info()    — provider, context window, etc.
5. (Optional)     adapter.close()         — release resources (e.g. HTTP sessions)

Usage pattern in runner.py
---------------------------
    from adapters.base import BaseAdapter
    from adapters.litellm_adapter import LiteLLMAdapter

    adapter: BaseAdapter = LiteLLMAdapter(model="gpt-4o")
    adapter.health_check()
    response = adapter.complete("What is 2+2?")
    info = adapter.model_info()
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Message:
    """A single message in a conversation."""
    role: str          # "system" | "user" | "assistant"
    content: str

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class CompletionResponse:
    """Normalised response from any adapter."""
    text: str                          # The model's reply
    model: str                         # Model string as reported by the provider
    prompt_tokens: Optional[int]       # Input tokens used
    completion_tokens: Optional[int]   # Output tokens used
    total_tokens: Optional[int]        # Total tokens used
    latency_ms: float                  # Wall-clock time for the API call
    raw: Any = field(default=None, repr=False)  # Raw provider response object

    @property
    def cost_estimate(self) -> Optional[float]:
        """
        Returns a rough cost estimate in USD if token counts are available.
        Subclasses can override with provider-specific pricing.
        """
        return None


@dataclass
class ModelInfo:
    """Metadata about the model as reported or known by the adapter."""
    model: str
    provider: str
    max_input_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None
    supports_system_prompt: bool = True
    supports_streaming: bool = False
    supports_functions: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthStatus:
    """Result of a health / connectivity check."""
    ok: bool
    latency_ms: float
    message: str
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseAdapter(ABC):
    """
    Abstract base class for all LLM provider adapters.

    Subclasses must implement:
        complete()       — single-turn completion (required)
        model_info()     — return ModelInfo for this adapter (required)

    Subclasses may override:
        complete_chat()  — multi-turn conversation
        stream()         — streaming completion
        health_check()   — connectivity test
        close()          — cleanup resources
    """

    def __init__(self, model: str, timeout: int = 60, **kwargs):
        """
        Args:
            model:   The model string as understood by this provider.
            timeout: Request timeout in seconds.
            kwargs:  Provider-specific config (api_key, base_url, etc.).
                     Store what you need as instance attributes.
        """
        self.model = model
        self.timeout = timeout
        # Subclasses: call super().__init__() then set your own attributes.

    # ------------------------------------------------------------------
    # REQUIRED — must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> CompletionResponse:
        """
        Send a single user prompt and return a CompletionResponse.

        Args:
            prompt:      The user message.
            system:      Optional system prompt prepended to the conversation.
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens:  Cap on output tokens (None = provider default).
            **kwargs:    Any provider-specific parameters to pass through.

        Returns:
            CompletionResponse with text, token counts, and latency.

        Raises:
            AdapterAuthError:      Invalid or missing API key.
            AdapterRateLimitError: Rate limit exceeded.
            AdapterTimeoutError:   Request timed out.
            AdapterError:          Any other provider error.
        """
        ...

    @abstractmethod
    def model_info(self) -> ModelInfo:
        """
        Return metadata about the model this adapter is configured for.

        Should not make a network call if the info is statically known.
        Fall back to a network call only when necessary.
        """
        ...

    # ------------------------------------------------------------------
    # OPTIONAL — override to add capability
    # ------------------------------------------------------------------

    def complete_chat(
        self,
        messages: List[Message],
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> CompletionResponse:
        """
        Multi-turn conversation completion.

        Default implementation extracts the last user message and any leading
        system message, then delegates to complete(). Override for providers
        that natively support full message history.

        Args:
            messages:    Ordered list of Message objects.
            temperature: Sampling temperature.
            max_tokens:  Cap on output tokens.
            **kwargs:    Provider-specific pass-through parameters.
        """
        system = None
        user_prompt = ""

        for msg in messages:
            if msg.role == "system" and system is None:
                system = msg.content
            elif msg.role == "user":
                user_prompt = msg.content  # last user message wins

        if not user_prompt:
            raise ValueError("No user message found in message list.")

        return self.complete(
            prompt=user_prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    def stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Iterator[str]:
        """
        Streaming completion — yields text chunks as they arrive.

        Default implementation calls complete() and yields the full text
        as a single chunk. Override for true streaming support.

        Args:
            prompt:      The user message.
            system:      Optional system prompt.
            temperature: Sampling temperature.
            max_tokens:  Cap on output tokens.

        Yields:
            str chunks of the response as they are generated.
        """
        response = self.complete(
            prompt=prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        yield response.text

    def health_check(self) -> HealthStatus:
        """
        Verify that the adapter can reach the provider and the model responds.

        Default implementation sends a minimal probe prompt and measures
        latency. Override if a lighter-weight check is available (e.g. a
        provider /ping endpoint).

        Returns:
            HealthStatus with ok=True if the check passed.
        """
        start = time.monotonic()
        try:
            self.complete(
                prompt="Reply with the single word: OK",
                temperature=0,
                max_tokens=5,
            )
            latency_ms = (time.monotonic() - start) * 1000
            return HealthStatus(ok=True, latency_ms=latency_ms, message="Probe succeeded.")
        except AdapterAuthError as e:
            latency_ms = (time.monotonic() - start) * 1000
            return HealthStatus(ok=False, latency_ms=latency_ms, message="Authentication failed.", error=str(e))
        except AdapterTimeoutError as e:
            latency_ms = (time.monotonic() - start) * 1000
            return HealthStatus(ok=False, latency_ms=latency_ms, message="Request timed out.", error=str(e))
        except Exception as e:
            latency_ms = (time.monotonic() - start) * 1000
            return HealthStatus(ok=False, latency_ms=latency_ms, message="Unexpected error.", error=str(e))

    def close(self) -> None:
        """
        Release any resources held by this adapter (HTTP sessions, thread pools, etc.).
        Called automatically when used as a context manager.
        Default is a no-op — override if your adapter holds persistent resources.
        """
        pass

    # ------------------------------------------------------------------
    # Context manager support — allows: with LiteLLMAdapter(...) as a:
    # ------------------------------------------------------------------

    def __enter__(self) -> "BaseAdapter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Helpers available to all subclasses
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        prompt: str,
        system: Optional[str],
    ) -> List[Dict[str, str]]:
        """
        Utility: build a messages list in OpenAI chat format.
        Call this from your complete() implementation.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _timed_call(self, fn, *args, **kwargs):
        """
        Utility: call fn(*args, **kwargs), return (result, latency_ms).
        Use this in complete() to measure latency consistently.

        Example:
            raw, latency_ms = self._timed_call(client.chat.completions.create, **params)
        """
        start = time.monotonic()
        result = fn(*args, **kwargs)
        latency_ms = (time.monotonic() - start) * 1000
        return result, latency_ms

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r})"


# ---------------------------------------------------------------------------
# Exceptions — adapter implementations should raise these, not raw exceptions
# ---------------------------------------------------------------------------

class AdapterError(Exception):
    """Base exception for all adapter errors."""
    def __init__(self, message: str, provider: Optional[str] = None, model: Optional[str] = None):
        super().__init__(message)
        self.provider = provider
        self.model = model


class AdapterAuthError(AdapterError):
    """Raised when authentication fails (bad/missing API key)."""


class AdapterRateLimitError(AdapterError):
    """Raised when the provider rate-limits the request."""
    def __init__(self, message: str, retry_after: Optional[float] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after  # seconds to wait before retrying, if known


class AdapterTimeoutError(AdapterError):
    """Raised when the request exceeds the configured timeout."""


class AdapterContextLengthError(AdapterError):
    """Raised when the prompt exceeds the model's context window."""


class AdapterUnsupportedFeatureError(AdapterError):
    """Raised when a feature (e.g. streaming) is not supported by this adapter."""
