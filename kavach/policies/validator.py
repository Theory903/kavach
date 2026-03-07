"""Policy schema validation using Pydantic models.

Defines the complete policy schema that maps to YAML/JSON policy files.
All policy evaluation operates on these validated models.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class Action(str, Enum):
    """Possible actions when a rule matches."""

    ALLOW = "allow"
    MONITOR = "monitor"
    SANITIZE = "sanitize"
    REQUIRE_APPROVAL = "require_approval"
    BLOCK = "block"


class PromptLogMode(str, Enum):
    """How to log prompts in audit trail."""

    RAW = "raw"
    HASHED = "hashed"
    REDACTED = "redacted"


class RolePolicy(BaseModel):
    """Permission set for a single role.

    Attributes:
        allowed_tools: List of tool names this role may invoke. ["*"] means all.
        blocked_tools: Tools explicitly denied even if allowed_tools is ["*"].
        data_scope: Glob patterns for accessible data paths.
        max_risk_score: Requests above this risk score are blocked.
        require_approval_above: Risk score threshold that triggers approval flow.
    """

    allowed_tools: list[str] = Field(default_factory=lambda: ["*"])
    blocked_tools: list[str] = Field(default_factory=list)
    data_scope: list[str] = Field(default_factory=lambda: ["*"])
    max_risk_score: float = Field(default=0.9, ge=0.0, le=1.0)
    require_approval_above: float = Field(default=0.7, ge=0.0, le=1.0)

    def is_tool_allowed(self, tool_name: str) -> bool:
        """Check if a tool is permitted for this role."""
        if tool_name in self.blocked_tools:
            return False
        if "*" in self.allowed_tools:
            return True
        return tool_name in self.allowed_tools


class Rule(BaseModel):
    """A single policy rule evaluated against request context.

    Attributes:
        id: Unique rule identifier.
        condition: Expression string evaluated against context variables.
        action: What to do when condition matches.
        reason: Human-readable explanation for audit logs.
        priority: Higher priority rules are evaluated first. Default 0.
    """

    id: str
    condition: str
    action: Action
    reason: str = ""
    priority: int = 0


class DetectionConfig(BaseModel):
    """Toggle detection capabilities."""

    injection_patterns: bool = True
    jailbreak_patterns: bool = True
    pii_detection: bool = True
    secret_detection: bool = True
    exfiltration_intent: bool = True


class OutputGuardConfig(BaseModel):
    """Output guard configuration."""

    scan_for_secrets: bool = True
    scan_tool_calls: bool = True
    block_unknown_tools: bool = True


class ObservabilityConfig(BaseModel):
    """Observability settings."""

    log_prompts: PromptLogMode = PromptLogMode.HASHED
    log_decisions: bool = True
    log_tool_calls: bool = True
    otel_endpoint: str | None = None


class Policy(BaseModel):
    """Root policy model — maps directly to policy.yaml.

    This is the single source of truth for all security enforcement
    configuration in a Kavach deployment.
    """

    version: str = "1.0"
    roles: dict[str, RolePolicy] = Field(default_factory=dict)
    rules: list[Rule] = Field(default_factory=list)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    output_guard: OutputGuardConfig = Field(default_factory=OutputGuardConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)

    @field_validator("rules")
    @classmethod
    def sort_rules_by_priority(cls, v: list[Rule]) -> list[Rule]:
        """Ensure rules are sorted by priority (highest first)."""
        return sorted(v, key=lambda r: r.priority, reverse=True)

    def get_role(self, role: str) -> RolePolicy | None:
        """Get policy for a specific role, or None if undefined."""
        return self.roles.get(role)

    def get_role_or_default(self, role: str) -> RolePolicy:
        """Get policy for a role, falling back to a restrictive default."""
        return self.roles.get(role, RolePolicy(
            allowed_tools=[],
            blocked_tools=["*"],
            max_risk_score=0.9,
        ))

    def validate_policy(self) -> list[str]:
        """Return a list of warnings/issues with this policy."""
        warnings: list[str] = []
        if not self.roles:
            warnings.append("No roles defined — all requests will use restrictive defaults")
        for rule in self.rules:
            if not rule.reason:
                warnings.append(f"Rule '{rule.id}' has no reason — audit logs will lack context")
        return warnings
