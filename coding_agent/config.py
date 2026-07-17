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
    context_keep_tool_batches: int = 3
    artifact_threshold_tokens: int = 1000
    artifact_read_default_chars: int = 8000
    artifact_search_max_hits: int = 5
    context_window_tokens: int = 128_000
    context_compaction_trigger_ratio: float = 0.70
    context_compaction_target_tokens: int = 25_000
    context_summary_max_tokens: int = 12_000
    context_summary_retry_count: int = 1
    context_message_trim_trigger: int = 60
    context_message_keep_head: int = 3
    context_message_keep_tail: int = 47
    team_auto_receive: bool = True
    team_message_batch_size: int = 20
    team_message_token_limit: int = 4000
    team_delivery_timeout_seconds: int = 60
    team_session_recent_messages: int = 12
    team_session_summary_tokens: int = 2000
    team_require_write_scope: bool = True

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
        if not 0 <= config.context_keep_tool_batches <= 20:
            raise ValueError("context_keep_tool_batches must be between 0 and 20")
        if config.artifact_threshold_tokens < 1:
            raise ValueError("artifact_threshold_tokens must be positive")
        if not 1 <= config.artifact_read_default_chars <= 12000:
            raise ValueError("artifact_read_default_chars must be between 1 and 12000")
        if not 1 <= config.artifact_search_max_hits <= 20:
            raise ValueError("artifact_search_max_hits must be between 1 and 20")
        if not 16_000 <= config.context_window_tokens <= 2_000_000:
            raise ValueError("context_window_tokens must be between 16000 and 2000000")
        if not 0.1 <= config.context_compaction_trigger_ratio <= 0.95:
            raise ValueError("context_compaction_trigger_ratio must be between 0.1 and 0.95")
        trigger_tokens = int(config.context_window_tokens * config.context_compaction_trigger_ratio)
        if not 1_000 <= config.context_compaction_target_tokens < trigger_tokens:
            raise ValueError("context_compaction_target_tokens must be at least 1000 and below the trigger")
        if not 256 <= config.context_summary_max_tokens <= config.context_compaction_target_tokens:
            raise ValueError("context_summary_max_tokens must be between 256 and the compaction target")
        if not 0 <= config.context_summary_retry_count <= 3:
            raise ValueError("context_summary_retry_count must be between 0 and 3")
        if not 10 <= config.context_message_trim_trigger <= 10_000:
            raise ValueError("context_message_trim_trigger must be between 10 and 10000")
        if not 1 <= config.context_message_keep_head < config.context_message_trim_trigger:
            raise ValueError("context_message_keep_head must be positive and below the trim trigger")
        if not 1 <= config.context_message_keep_tail < config.context_message_trim_trigger:
            raise ValueError("context_message_keep_tail must be positive and below the trim trigger")
        if config.context_message_keep_head + config.context_message_keep_tail >= config.context_message_trim_trigger:
            raise ValueError("message head and tail retention must leave trim hysteresis")
        if not 1 <= config.team_message_batch_size <= 100:
            raise ValueError("team_message_batch_size must be between 1 and 100")
        if not 256 <= config.team_message_token_limit <= 32000:
            raise ValueError("team_message_token_limit must be between 256 and 32000")
        if not 1 <= config.team_delivery_timeout_seconds <= 3600:
            raise ValueError("team_delivery_timeout_seconds must be between 1 and 3600")
        if not 1 <= config.team_session_recent_messages <= 100:
            raise ValueError("team_session_recent_messages must be between 1 and 100")
        if not 256 <= config.team_session_summary_tokens <= 12000:
            raise ValueError("team_session_summary_tokens must be between 256 and 12000")
        return config
