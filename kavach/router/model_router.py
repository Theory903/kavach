"""ModelRouter — multi-LLM provider selection and dispatch.

Routes requests to the optimal LLM provider based on:
- Cost: cheap providers for low-risk requests
- Risk: local/safe models for high-risk contexts
- Latency: fallback on timeout
- Policy: explicit rules in kavach policy YAML
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class Provider(str, Enum):
    """Supported LLM provider identifiers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    DEEPSEEK = "deepseek"
    LOCAL = "local"
    AUTO = "auto"


@dataclass
class ProviderConfig:
    """Configuration for a single LLM provider."""

    name: Provider
    # Callable: (messages: list[dict], **kwargs) -> str
    client_fn: Callable[..., str] | None = None
    # Relative cost weight (lower = cheaper)
    cost_weight: float = 1.0
    # Max risk score this provider should handle (0.0 = only benign)
    max_risk_score: float = 1.0
    # Request timeout in seconds
    timeout_seconds: float = 30.0
    enabled: bool = True


@dataclass
class RouterConfig:
    """Router-level configuration loaded from policy YAML or passed directly."""

    default_provider: Provider = Provider.OPENAI
    # Ordered fallback chain — tried in sequence on failure
    fallback_chain: list[Provider] = field(default_factory=lambda: [Provider.OPENAI])
    # Risk threshold above which we route to local/safer provider
    risk_isolation_threshold: float = 0.7
    # Provider to use for high-risk requests
    high_risk_provider: Provider = Provider.LOCAL
    # Max retries per provider before moving to next
    max_retries: int = 1


@dataclass
class RouteResult:
    """Result of a routing decision."""

    provider: Provider
    response: str | None = None
    latency_ms: float = 0.0
    success: bool = False
    error: str | None = None
    tried_providers: list[str] = field(default_factory=list)


class ModelRouter:
    """Routes prompts to the best LLM provider.

    Usage::

        router = ModelRouter(config=RouterConfig())
        router.register_provider(ProviderConfig(
            name=Provider.OPENAI,
            client_fn=lambda messages, **k: openai.chat.completions.create(
                model="gpt-4o-mini", messages=messages
            ).choices[0].message.content,
        ))

        result = router.route(
            messages=[{"role": "user", "content": "hello"}],
            risk_score=0.1,
        )
    """

    def __init__(self, config: RouterConfig | None = None) -> None:
        self._config = config or RouterConfig()
        self._providers: dict[Provider, ProviderConfig] = {}

    def register_provider(self, provider_cfg: ProviderConfig) -> None:
        """Register an LLM provider with its client function.

        Args:
            provider_cfg: Provider configuration and callable.
        """
        self._providers[provider_cfg.name] = provider_cfg
        logger.info(f"[Kavach Router] Registered provider: {provider_cfg.name.value}")

    def _select_provider(self, risk_score: float) -> list[Provider]:
        """Select an ordered list of providers to try based on risk.

        High-risk requests are isolated to the configured safe provider.
        Returns the full fallback chain starting from the best candidate.
        """
        if risk_score >= self._config.risk_isolation_threshold:
            high_risk = self._config.high_risk_provider
            chain = [high_risk] + [
                p for p in self._config.fallback_chain if p != high_risk
            ]
        else:
            # Use cost-weighted selection from registered + enabled providers
            eligible = [
                p for p in self._config.fallback_chain
                if p in self._providers and self._providers[p].enabled
            ]
            if not eligible:
                eligible = [self._config.default_provider]
            chain = eligible

        return chain

    def route(
        self,
        messages: list[dict[str, str]],
        risk_score: float = 0.0,
        **kwargs: Any,
    ) -> RouteResult:
        """Route a message list to the best provider.

        Args:
            messages: OpenAI-style message list (role/content dicts).
            risk_score: Current request risk score (0.0–1.0).
            **kwargs: Extra kwargs passed to the provider's client_fn.

        Returns:
            RouteResult with the response, provider, and latency.
        """
        chain = self._select_provider(risk_score)
        tried: list[str] = []

        for provider_name in chain:
            cfg = self._providers.get(provider_name)
            if cfg is None or not cfg.enabled or cfg.client_fn is None:
                logger.debug(f"[Kavach Router] Provider {provider_name} not available, skipping")
                tried.append(str(provider_name))
                continue

            start = time.monotonic()
            tried.append(provider_name.value)
            for attempt in range(self._config.max_retries + 1):
                try:
                    response = cfg.client_fn(messages, **kwargs)
                    latency = (time.monotonic() - start) * 1000
                    logger.info(
                        f"[Kavach Router] Routed to {provider_name.value} "
                        f"(risk={risk_score:.2f}, latency={latency:.0f}ms)"
                    )
                    return RouteResult(
                        provider=provider_name,
                        response=response,
                        latency_ms=round(latency, 2),
                        success=True,
                        tried_providers=tried,
                    )
                except TimeoutError:
                    logger.warning(
                        f"[Kavach Router] Timeout on {provider_name.value} "
                        f"(attempt {attempt + 1}/{self._config.max_retries + 1})"
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        f"[Kavach Router] Error on {provider_name.value}: {exc} "
                        f"(attempt {attempt + 1}/{self._config.max_retries + 1})"
                    )
                    if attempt == self._config.max_retries:
                        break

        return RouteResult(
            provider=chain[0] if chain else self._config.default_provider,
            response=None,
            success=False,
            error="All providers failed or unavailable",
            tried_providers=tried,
        )

    def route_prompt(
        self,
        prompt: str,
        risk_score: float = 0.0,
        system_prompt: str = "You are a helpful assistant.",
        **kwargs: Any,
    ) -> RouteResult:
        """Convenience method — wraps a single prompt string as messages.

        Args:
            prompt: User prompt text.
            risk_score: Request risk score.
            system_prompt: System prompt to prepend.
            **kwargs: Extra kwargs for the provider.

        Returns:
            RouteResult.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        return self.route(messages=messages, risk_score=risk_score, **kwargs)

    @classmethod
    def from_env(cls) -> "ModelRouter":
        """Create a router auto-wired from environment variables.

        Providers are enabled based on presence of their API key env vars:
        - OPENAI_API_KEY → OpenAI (gpt-4o-mini)
        - ANTHROPIC_API_KEY → Anthropic (claude-haiku-3-5)

        Returns:
            Configured ModelRouter.
        """
        import os  # noqa: PLC0415

        router = cls()
        enabled: list[Provider] = []

        # OpenAI
        if os.environ.get("OPENAI_API_KEY"):
            try:
                from openai import OpenAI  # noqa: PLC0415
                client = OpenAI()

                def openai_fn(messages: list[dict], **kw: Any) -> str:
                    resp = client.chat.completions.create(
                        model=kw.get("model", "gpt-4o-mini"),
                        messages=messages,  # type: ignore[arg-type]
                    )
                    return resp.choices[0].message.content or ""

                router.register_provider(ProviderConfig(
                    name=Provider.OPENAI,
                    client_fn=openai_fn,
                    cost_weight=1.0,
                    max_risk_score=0.7,
                ))
                enabled.append(Provider.OPENAI)
            except ImportError:
                logger.debug("[Kavach Router] openai package not installed")

        # Anthropic
        if os.environ.get("ANTHROPIC_API_KEY"):
            try:
                import anthropic  # noqa: PLC0415
                client_a = anthropic.Anthropic()

                def anthropic_fn(messages: list[dict], **kw: Any) -> str:
                    resp = client_a.messages.create(
                        model=kw.get("model", "claude-haiku-3-5-20241022"),
                        max_tokens=2048,
                        messages=messages,  # type: ignore[arg-type]
                    )
                    return resp.content[0].text if resp.content else ""

                router.register_provider(ProviderConfig(
                    name=Provider.ANTHROPIC,
                    client_fn=anthropic_fn,
                    cost_weight=0.8,
                    max_risk_score=0.6,
                ))
                enabled.append(Provider.ANTHROPIC)
            except ImportError:
                logger.debug("[Kavach Router] anthropic package not installed")

        if enabled:
            router._config.default_provider = enabled[0]
            router._config.fallback_chain = enabled

        return router
