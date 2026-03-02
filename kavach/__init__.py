"""Kavach — The security and permission layer for agentic AI systems.

Kavach provides identity enforcement, input/output guards, tool permission
controls, and policy-based decision making for any AI stack.

Usage:
    from kavach import KavachGateway, ToolGuard
    from kavach.integrations.openai import SecureOpenAI
"""

from kavach.core.gateway import KavachGateway
from kavach.guards.tool_guard import ToolGuard

__version__ = "0.1.0"
__all__ = ["KavachGateway", "ToolGuard", "__version__"]
