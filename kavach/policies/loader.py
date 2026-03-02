"""Policy loader — reads YAML/JSON files into validated Policy models.

Supports loading from file path, dict, or raw YAML string.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from kavach.policies.validator import Policy


def load_policy(source: str | Path | dict[str, Any]) -> Policy:
    """Load and validate a policy from various sources.

    Args:
        source: One of:
            - Path to a YAML/JSON file (str or Path)
            - Dict with policy data
            - Raw YAML string (must contain 'version:')

    Returns:
        Validated Policy model.

    Raises:
        FileNotFoundError: If file path doesn't exist.
        ValueError: If policy schema validation fails.
        yaml.YAMLError: If YAML parsing fails.
    """
    if isinstance(source, dict):
        return Policy.model_validate(source)

    if isinstance(source, Path) or (isinstance(source, str) and not source.strip().startswith("version")):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Policy file not found: {path}")
        raw = path.read_text(encoding="utf-8")
    else:
        raw = source

    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ValueError(f"Policy must be a YAML mapping, got {type(data).__name__}")

    return Policy.model_validate(data)


def load_default_policy() -> Policy:
    """Load the built-in default policy template.

    Returns:
        The default policy with analyst + admin roles and standard rules.
    """
    template_dir = Path(__file__).parent / "templates"
    default_path = template_dir / "default.yaml"
    if default_path.exists():
        return load_policy(default_path)
    # Fallback: return a minimal restrictive policy
    return Policy()
