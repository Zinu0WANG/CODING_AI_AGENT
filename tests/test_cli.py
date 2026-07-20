from pathlib import Path

from rich.console import Console

from coding_agent.cli import AgentCLI
from coding_agent.runtime import FakeModel


PLAN_TEXT = "\n".join([
    "## 目标与验收标准", "## 仓库现状", "## 实施步骤",
    "## 预计修改文件及原因", "## 测试方案", "## 风险与假设",
])


def test_cli_accepts_dashscope_key_for_anthropic_compatible_provider(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-only-key")
    monkeypatch.setenv("MODEL_ID", "qwen3.7-plus")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://dashscope.aliyuncs.com/apps/anthropic")
    cli = AgentCLI(workspace=tmp_path)
    assert cli.model.model == "qwen3.7-plus"


def test_cli_plans_then_implements_and_links_both_runs(tmp_path: Path, monkeypatch):
    model = FakeModel([
        {"stop_reason": "end_turn", "content": [{"type": "text", "text": PLAN_TEXT}]},
        {"stop_reason": "end_turn", "content": [{"type": "text", "text": "Implemented"}]},
    ])
    monkeypatch.setattr(AgentCLI, "_create_model", lambda self: model)
    cli = AgentCLI(workspace=tmp_path, console=Console(record=True))

    plan = cli.create_plan("improve app")
    assert plan and plan["status"] == "ready"
    planning_run = plan["planning_run_id"]
    result = cli.implement_plan(plan["plan_id"])

    assert result and result.status == "completed"
    saved = cli.plan_store.load(plan["plan_id"])
    assert saved["status"] == "completed"
    assert saved["implementation_run_id"] == result.run_id
    assert saved["implementation_run_id"] != planning_run
    planning_events = cli.last_runtime.events.__class__(tmp_path, planning_run).read_events()
    implementation_events = cli.last_runtime.events.read_events()
    assert any(event["type"] == "plan_created" for event in planning_events)
    assert any(event["type"] == "plan_implementation_started" for event in implementation_events)
    assert any(event["type"] == "plan_implementation_completed" for event in implementation_events)


def test_cli_refuses_to_implement_stale_plan(tmp_path: Path, monkeypatch):
    (tmp_path / "app.py").write_text("before", encoding="utf-8")
    model = FakeModel([
        {"stop_reason": "end_turn", "content": [{"type": "text", "text": PLAN_TEXT}]},
    ])
    monkeypatch.setattr(AgentCLI, "_create_model", lambda self: model)
    cli = AgentCLI(workspace=tmp_path, console=Console(record=True))
    plan = cli.create_plan("change app")
    (tmp_path / "app.py").write_text("after", encoding="utf-8")

    assert cli.implement_plan(plan["plan_id"]) is None
    assert cli.plan_store.load(plan["plan_id"])["status"] == "stale"
    planning_events = cli.last_runtime.events.read_events()
    assert any(event["type"] == "plan_stale" for event in planning_events)


def test_cli_plan_commands_are_routed(tmp_path: Path, monkeypatch):
    model = FakeModel([
        {"stop_reason": "end_turn", "content": [{"type": "text", "text": PLAN_TEXT}]},
    ])
    monkeypatch.setattr(AgentCLI, "_create_model", lambda self: model)
    cli = AgentCLI(workspace=tmp_path, console=Console(record=True))
    assert cli.handle_command("/plan inspect repository") is True
    plan_id = cli.plan_store.list_all()[0]["plan_id"]
    assert cli.handle_command("/plans") is True
    assert cli.handle_command(f"/show-plan {plan_id}") is True


def test_cli_does_not_save_empty_or_failed_plan(tmp_path: Path, monkeypatch):
    model = FakeModel([
        {"stop_reason": "end_turn", "content": [{"type": "text", "text": ""}]},
    ])
    monkeypatch.setattr(AgentCLI, "_create_model", lambda self: model)
    cli = AgentCLI(workspace=tmp_path, console=Console(record=True))
    assert cli.create_plan("unclear request") is None
    assert cli.plan_store.list_all() == []
