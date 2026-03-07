"""Kavach sandbox — isolated tool execution for agent security.

Runs potentially dangerous tool calls inside Docker containers or
subprocess jails to prevent privilege escalation and data exfiltration.
"""

from kavach.sandbox.docker_sandbox import DockerSandbox, SandboxPolicy, SandboxResult

__all__ = ["DockerSandbox", "SandboxPolicy", "SandboxResult"]
