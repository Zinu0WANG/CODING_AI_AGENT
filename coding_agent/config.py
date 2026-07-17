from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml


DEFAULT_IGNORES = [
    ".git/**", ".venv/**", "venv/**", "__pycache__/**", "node_modules/**",
    "build/**", "dist/**", ".runs/**", ".team/**", ".tasks/**",
]


@dataclass(slots=True)
class AgentConfig:
    test_commands: list[str] = field(default_factory=list)
    lint_commands: list[str] = field(default_factory=list)
    ignore_patterns: list[str] = field(default_factory=lambda: list(DEFAULT_IGNORES))
    approval_policy: str = "ask_on_write"
    max_steps: int = 40
    max_fix_attempts: int = 2
    command_timeout: int = 120
    max_file_bytes: int = 250_000

    @classmethod
    def load(cls, workspace: Path) -> "AgentConfig":
        path = workspace / ".agent.yml"
        if not path.exists():
            return cls()
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(".agent.yml must contain a mapping")
        allowed = {item.name for item in fields(cls)}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"Unknown .agent.yml keys: {', '.join(sorted(unknown))}")
        config = cls(**raw)
        config.ignore_patterns = list(dict.fromkeys(DEFAULT_IGNORES + config.ignore_patterns))
        if config.approval_policy not in {"ask_on_write", "allow_write", "read_only"}:
            raise ValueError("approval_policy must be ask_on_write, allow_write, or read_only")
        if not 1 <= config.max_steps <= 200:
            raise ValueError("max_steps must be between 1 and 200")
        return config
