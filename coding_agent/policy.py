from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class RiskLevel(str, Enum):
    READ = "read"
    WRITE = "write"
    DANGEROUS = "dangerous"


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    risk: RiskLevel
    reason: str


class ToolPolicy:
    DANGEROUS_PATTERNS = [
        (r"\b(rm|del|rmdir|remove-item)\b", "deletes files"),
        (r"\b(sudo|shutdown|reboot)\b", "changes host system state"),
        (r"\b(git\s+(push|reset|clean)|git\s+checkout\s+--)\b", "destructive or remote Git operation"),
        (r"\b(pip|npm|pnpm|yarn|uv|apt|brew|choco)\s+(install|add)\b", "installs dependencies"),
        (r"\b(curl|wget|invoke-webrequest|ssh|scp)\b", "uses the network"),
        (r"\|\s*(sh|bash|powershell|pwsh)\b", "pipes untrusted content to a shell"),
    ]
    READ_PREFIXES = (
        "git status", "git diff", "git log", "git show", "git branch", "git rev-parse",
        "rg ", "grep ", "find ", "ls", "dir", "pwd", "get-childitem", "get-content",
        "python -m py_compile", "python --version", "pytest --collect-only",
    )

    def __init__(self, workspace: Path):
        self.workspace = workspace.resolve()

    def resolve_path(self, raw: str) -> Path:
        if not raw or "\x00" in raw:
            raise ValueError("Invalid empty or NUL-containing path")
        candidate = (self.workspace / raw).resolve()
        try:
            candidate.relative_to(self.workspace)
        except ValueError as exc:
            raise ValueError(f"Path escapes workspace: {raw}") from exc
        return candidate

    def classify_path(self, raw: str, write: bool) -> PolicyDecision:
        try:
            self.resolve_path(raw)
        except ValueError as exc:
            return PolicyDecision(RiskLevel.DANGEROUS, str(exc))
        return PolicyDecision(RiskLevel.WRITE if write else RiskLevel.READ, "workspace file access")

    def classify_command(self, command: str) -> PolicyDecision:
        normalized = " ".join(command.strip().lower().split())
        if not normalized:
            return PolicyDecision(RiskLevel.DANGEROUS, "empty command")
        for pattern, reason in self.DANGEROUS_PATTERNS:
            if re.search(pattern, normalized, re.IGNORECASE):
                return PolicyDecision(RiskLevel.DANGEROUS, reason)
        if re.search(r"(^|\s)(\.\.[/\\]|[a-z]:[/\\]|/etc/|/home/|/root/|~[/\\])", normalized):
            return PolicyDecision(RiskLevel.DANGEROUS, "may access paths outside the workspace")
        if any(normalized == prefix.rstrip() or normalized.startswith(prefix) for prefix in self.READ_PREFIXES):
            return PolicyDecision(RiskLevel.READ, "recognized read-only command")
        return PolicyDecision(RiskLevel.WRITE, "command may change workspace state")
