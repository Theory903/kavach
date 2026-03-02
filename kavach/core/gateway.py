"""KavachGateway — main orchestrator for the Kavach SDK.

The gateway is the single entry point that coordinates:
identity binding → input guard → [LLM call] → output guard → decision

It wraps everything together so integrations only need to call
gateway.secure_call() to get full protection.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from kavach.core.identity import Identity, IdentityContext
from kavach.core.policy_engine import Decision, PolicyEngine
from kavach.core.risk_scorer import RiskScorer
from kavach.guards.input_guard import InputGuard
from kavach.guards.output_guard import OutputGuard
from kavach.guards.tool_guard import ToolGuard
from kavach.ml.rl_advisor import RLDecisionAdvisor
from kavach.observability.logger import KavachLogger
from kavach.observability.metrics import KavachMetrics
from kavach.observability.tracer import KavachTracer
from kavach.policies.validator import Action, Policy


class KavachGateway:
    """Main orchestrator for the Kavach security layer.

    Coordinates all guards, detectors, and policy evaluation
    in a single pipeline that wraps any LLM call.

    Usage:
        gateway = KavachGateway(policy="policy.yaml")

        # Option 1: Analyze only (no execution)
        decision = gateway.analyze(prompt="...", user_id="u1", role="analyst")

        # Option 2: Secure execution
        result = gateway.secure_call(
            prompt="...",
            user_id="u1",
            role="analyst",
            llm_call=lambda prompt: openai_client.chat(...),
        )
    """

    def __init__(
        self,
        policy: str | Path | dict[str, Any] | Policy | None = None,
        risk_scorer: RiskScorer | None = None,
    ) -> None:
        """Initialize the gateway with a policy.

        Args:
            policy: Policy source (file path, dict, Policy, or None for default).
            risk_scorer: Custom risk scorer, or None for default weights.
        """
        self._engine = PolicyEngine(policy)
        self._input_guard = InputGuard(
            policy=self._engine.policy,
            risk_scorer=risk_scorer,
        )
        self._output_guard = OutputGuard(policy=self._engine.policy)
        self._tool_guard = ToolGuard(policy=self._engine.policy)
        self._rl_advisor = RLDecisionAdvisor(persist_path="data/rl_q_table.npy")
        self._logger = KavachLogger(
            log_prompts=self._engine.policy.observability.log_prompts,
        )
        self._metrics = KavachMetrics()
        self._tracer = KavachTracer()

    @property
    def input_guard(self) -> InputGuard:
        return self._input_guard

    @property
    def output_guard(self) -> OutputGuard:
        return self._output_guard

    @property
    def tool_guard(self) -> ToolGuard:
        return self._tool_guard

    @property
    def metrics(self) -> KavachMetrics:
        return self._metrics

    def analyze(
        self,
        prompt: str,
        user_id: str = "anonymous",
        role: str = "default",
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Analyze a prompt without executing — returns decision.

        Args:
            prompt: The user prompt to analyze.
            user_id: User identifier.
            role: User's role for RBAC.
            session_id: Optional session ID.

        Returns:
            Decision dict with risk_score, action, and reasons.
        """
        start = time.monotonic()

        identity = Identity(
            user_id=user_id,
            role=role,
            session_id=session_id or f"sess_analyze",
        )

        with IdentityContext(identity):
            with self._tracer.span("kavach.analyze", {"role": role}):
                guard_result = self._input_guard.scan(prompt, identity=identity)

        decision = guard_result.decision
        decision.latency_ms = (time.monotonic() - start) * 1000
        decision.session_id = identity.session_id
        decision.trace_id = self._tracer.generate_trace_id()

        # ML intent parsing for RL
        intent_cat = "unknown"
        if guard_result.metadata.get("ml_details") and guard_result.metadata["ml_details"].get("intent_analysis"):
            intent_cat = guard_result.metadata["ml_details"]["intent_analysis"].get("predicted_category", "unknown")

        rl_sugg = self._rl_advisor.suggest(
            risk_score=decision.risk_score,
            intent_category=intent_cat,
            role=role,
            behavioral_multiplier=1.0,  # Could fetch from tracker if desired
        )

        final_action = self._rl_advisor.apply_policy_override(
            rl_suggestion=rl_sugg["action"],
            policy_decision=decision.action.value,
            risk_score=decision.risk_score,
        )

        decision.action = Action(final_action)

        if guard_result.clean_prompt:
            decision.clean_prompt = guard_result.clean_prompt

        result = decision.to_dict()
        result["rl_suggestion"] = rl_sugg

        # Log & record metrics
        self._logger.log_decision(result, prompt=prompt, identity=identity.to_dict())
        self._metrics.record(decision.action.value, decision.latency_ms, decision.risk_score)

        return result

    def secure_call(
        self,
        prompt: str,
        user_id: str = "anonymous",
        role: str = "default",
        llm_call: Callable[[str], str] | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Full secure pipeline: analyze → call LLM → guard output.

        Args:
            prompt: The user prompt.
            user_id: User identifier.
            role: User's role.
            llm_call: Function that takes a prompt and returns LLM response.
            session_id: Optional session ID.

        Returns:
            Dict with decision, response (if allowed), and metadata.
        """
        start = time.monotonic()

        identity = Identity(
            user_id=user_id,
            role=role,
            session_id=session_id or "",
        )

        with IdentityContext(identity):
            # Step 1: Input guard
            with self._tracer.span("kavach.input_guard"):
                guard_result = self._input_guard.scan(prompt, identity=identity)

            input_decision = guard_result.decision

            # Apply RL Advisor to input
            intent_cat = "unknown"
            if guard_result.metadata.get("ml_details") and guard_result.metadata["ml_details"].get("intent_analysis"):
                intent_cat = guard_result.metadata["ml_details"]["intent_analysis"].get("predicted_category", "unknown")

            rl_sugg = self._rl_advisor.suggest(
                risk_score=input_decision.risk_score,
                intent_category=intent_cat,
                role=role,
                behavioral_multiplier=1.0,
            )
            final_input_action = self._rl_advisor.apply_policy_override(
                rl_suggestion=rl_sugg["action"],
                policy_decision=input_decision.action.value,
                risk_score=input_decision.risk_score,
            )
            input_decision.action = Action(final_input_action)

            if input_decision.is_blocked:
                input_decision.latency_ms = (time.monotonic() - start) * 1000
                input_decision.session_id = identity.session_id
                result = {
                    **input_decision.to_dict(),
                    "rl_suggestion": rl_sugg,
                    "action_taken": "blocked_before_llm",
                    "response": None,
                }
                self._logger.log_decision(result, prompt=prompt, identity=identity.to_dict())
                self._metrics.record("block", input_decision.latency_ms, input_decision.risk_score)
                return result

            # Step 2: Use clean prompt if sanitized
            effective_prompt = guard_result.clean_prompt or prompt

            # Step 3: Call LLM (if provided)
            llm_response = None
            if llm_call is not None:
                with self._tracer.span("kavach.llm_call"):
                    llm_response = llm_call(effective_prompt)

            # Step 4: Output guard
            output_decision = Decision(action=Action.ALLOW)
            if llm_response:
                with self._tracer.span("kavach.output_guard"):
                    output_result = self._output_guard.scan(llm_response, identity=identity)
                    output_decision = output_result.decision

                    if output_result.redacted_output:
                        llm_response = output_result.redacted_output

            # Build final response
            total_latency = (time.monotonic() - start) * 1000
            final_action = (
                output_decision.action
                if output_decision.action != Action.ALLOW
                else input_decision.action
            )

            result = {
                "decision": final_action.value,
                "risk_score": round(input_decision.risk_score, 4),
                "reasons": input_decision.reasons + output_decision.reasons,
                "action_taken": "completed",
                "response": llm_response,
                "clean_prompt": guard_result.clean_prompt,
                "latency_ms": round(total_latency, 2),
                "session_id": identity.session_id,
                "trace_id": self._tracer.generate_trace_id(),
            }

            self._logger.log_decision(result, prompt=prompt, identity=identity.to_dict())
            self._metrics.record(final_action.value, total_latency, input_decision.risk_score)

            return result

    def sanitize_prompt(self, prompt: str) -> str:
        """Just sanitize a prompt — no policy evaluation.

        Args:
            prompt: Raw prompt text.

        Returns:
            Cleaned prompt text.
        """
        from kavach.sanitizers.prompt_cleaner import PromptCleaner
        return PromptCleaner().clean(prompt)
