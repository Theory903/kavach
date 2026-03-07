"""DockerSandbox — isolated tool execution for agent security.

Agents often need to execute tools with side effects: shell commands,
file operations, HTTP requests, database queries. Running these directly
grants the LLM (or an attacker hijacking it) full system access.

This sandbox wraps tool execution in Docker containers with:
- Disabled network (unless explicitly allowed)
- Read-only root filesystem (writable tmp only)
- CPU and memory limits
- Execution timeout
- Allowlist-based command filtering

Falls back gracefully to direct execution when Docker is not available.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Docker image used for sandboxed execution
_DEFAULT_SANDBOX_IMAGE = "python:3.12-alpine"
# Default execution timeout in seconds
_DEFAULT_TIMEOUT = 30


@dataclass
class SandboxPolicy:
    """Security policy for the sandbox.

    Defines what is and isn't allowed inside the container.
    """

    # Docker image to use for sandboxing
    image: str = _DEFAULT_SANDBOX_IMAGE
    # Allow network access inside the container
    allow_network: bool = False
    # Container memory limit (Docker syntax, e.g. "128m")
    memory_limit: str = "128m"
    # Container CPU share (relative weight, default 512)
    cpu_shares: int = 512
    # Execution timeout in seconds
    timeout_seconds: int = _DEFAULT_TIMEOUT
    # Allowed commands (empty = allow all non-blocked)
    allowed_commands: list[str] = field(default_factory=list)
    # Commands that are always blocked
    blocked_commands: list[str] = field(default_factory=lambda: [
        "rm -rf /",
        "dd if=/dev/zero",
        "fork bomb",
        ":(){ :|:& };:",
        "chmod 777",
        "curl",           # network — enforced separately by allow_network
        "wget",
        "nc ",
        "netcat",
    ])


@dataclass
class SandboxResult:
    """Result of a sandboxed tool execution."""

    success: bool
    stdout: str = ""
    stderr: str = ""
    exit_code: int = -1
    latency_ms: float = 0.0
    sandboxed: bool = True
    blocked_reason: str = ""


class DockerSandbox:
    """Run tool commands inside Docker containers.

    Usage::

        policy = SandboxPolicy(allow_network=False, memory_limit="64m")
        sandbox = DockerSandbox(policy=policy)

        result = sandbox.run("python3 -c 'print(1+1)'")
        print(result.stdout)  # "2"

    When Docker is not available, falls back to subprocess with timeout only.
    """

    def __init__(self, policy: SandboxPolicy | None = None) -> None:
        self._policy = policy or SandboxPolicy()
        self._docker_available = self._check_docker()

    def _check_docker(self) -> bool:
        """Check whether Docker CLI is available on this system."""
        available = shutil.which("docker") is not None
        if not available:
            logger.warning(
                "[DockerSandbox] Docker not available — falling back to subprocess execution"
            )
        return available

    def _is_command_blocked(self, command: str) -> str:
        """Return a block reason if the command matches a blocked pattern."""
        cmd_lower = command.lower().strip()

        for blocked in self._policy.blocked_commands:
            if blocked.lower() in cmd_lower:
                return f"Blocked command pattern: '{blocked}'"

        if self._policy.allowed_commands:
            is_allowed = any(
                cmd_lower.startswith(allowed.lower())
                for allowed in self._policy.allowed_commands
            )
            if not is_allowed:
                return f"Command not in allowed list: {self._policy.allowed_commands}"

        return ""

    def run(self, command: str, cwd: str | None = None) -> SandboxResult:
        """Execute a shell command in an isolated environment.

        Args:
            command: Shell command to run.
            cwd: Working directory (used only in fallback mode).

        Returns:
            SandboxResult with stdout, stderr, and exit code.
        """
        start = time.monotonic()

        # Pre-execution block check
        blocked_reason = self._is_command_blocked(command)
        if blocked_reason:
            logger.warning(f"[DockerSandbox] Blocked: {blocked_reason}")
            return SandboxResult(
                success=False,
                blocked_reason=blocked_reason,
                sandboxed=True,
                latency_ms=(time.monotonic() - start) * 1000,
            )

        if self._docker_available:
            return self._run_in_docker(command, start)
        else:
            return self._run_fallback(command, cwd, start)

    def _run_in_docker(self, command: str, start: float) -> SandboxResult:
        """Run command inside a Docker container with security constraints."""
        docker_args = [
            "docker", "run",
            "--rm",                          # Remove container after exit
            "--read-only",                   # Read-only root filesystem
            "--tmpfs", "/tmp:rw,size=32m",   # Writable /tmp only
            "--memory", self._policy.memory_limit,
            "--cpu-shares", str(self._policy.cpu_shares),
            "--pids-limit", "64",            # Limit process count
            "--security-opt", "no-new-privileges",
        ]

        if not self._policy.allow_network:
            docker_args += ["--network", "none"]

        docker_args += [
            self._policy.image,
            "sh", "-c", command,
        ]

        try:
            proc = subprocess.run(
                docker_args,
                capture_output=True,
                text=True,
                timeout=self._policy.timeout_seconds,
            )
            latency = (time.monotonic() - start) * 1000
            return SandboxResult(
                success=proc.returncode == 0,
                stdout=proc.stdout,
                stderr=proc.stderr,
                exit_code=proc.returncode,
                latency_ms=round(latency, 2),
                sandboxed=True,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(
                success=False,
                stderr=f"Execution timed out after {self._policy.timeout_seconds}s",
                exit_code=-1,
                latency_ms=(time.monotonic() - start) * 1000,
                sandboxed=True,
                blocked_reason="timeout",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[DockerSandbox] Docker run failed: {exc}")
            return SandboxResult(
                success=False,
                stderr=str(exc),
                sandboxed=True,
                latency_ms=(time.monotonic() - start) * 1000,
            )

    def _run_fallback(self, command: str, cwd: str | None, start: float) -> SandboxResult:
        """Fallback: run in subprocess with timeout only (no container isolation)."""
        logger.warning(f"[DockerSandbox] Using subprocess fallback for: {command[:60]}...")
        try:
            proc = subprocess.run(
                command,
                shell=True,  # noqa: S602
                capture_output=True,
                text=True,
                timeout=self._policy.timeout_seconds,
                cwd=cwd,
            )
            latency = (time.monotonic() - start) * 1000
            return SandboxResult(
                success=proc.returncode == 0,
                stdout=proc.stdout,
                stderr=proc.stderr,
                exit_code=proc.returncode,
                latency_ms=round(latency, 2),
                sandboxed=False,  # Not truly sandboxed
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(
                success=False,
                stderr="Execution timed out",
                exit_code=-1,
                latency_ms=(time.monotonic() - start) * 1000,
                sandboxed=False,
                blocked_reason="timeout",
            )

    def run_python(self, code: str) -> SandboxResult:
        """Execute a Python snippet in the sandbox.

        Args:
            code: Python source code string.

        Returns:
            SandboxResult with stdout from the script.
        """
        escaped = code.replace("'", "'\"'\"'")
        command = f"python3 -c '{escaped}'"
        return self.run(command)
