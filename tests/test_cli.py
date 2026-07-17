from pathlib import Path

from coding_agent.cli import AgentCLI


def test_cli_accepts_dashscope_key_for_anthropic_compatible_provider(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-only-key")
    monkeypatch.setenv("MODEL_ID", "qwen3.7-plus")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://dashscope.aliyuncs.com/apps/anthropic")
    cli = AgentCLI(workspace=tmp_path)
    assert cli.model.model == "qwen3.7-plus"
