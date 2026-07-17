import json
import threading
import time
from pathlib import Path

import pytest

from coding_agent.config import AgentConfig
from coding_agent.context import RepoMap
from coding_agent.events import EventStore
from coding_agent.policy import RiskLevel, ToolPolicy
from coding_agent.state import MessageBus, TaskManager, validate_name


def test_config_loads_defaults_and_project_overrides(tmp_path: Path):
    (tmp_path / ".agent.yml").write_text(
        "test_commands:\n  - python -m pytest -q\nmax_steps: 12\n",
        encoding="utf-8",
    )
    config = AgentConfig.load(tmp_path)
    assert config.test_commands == ["python -m pytest -q"]
    assert config.max_steps == 12
    assert ".git/**" in config.ignore_patterns
    assert config.context_keep_tool_batches == 3
    assert config.artifact_threshold_tokens == 1000
    assert config.context_window_tokens == 128000
    assert config.context_compaction_trigger_ratio == 0.70
    assert config.context_compaction_target_tokens == 25000
    assert config.context_summary_max_tokens == 12000
    assert config.context_summary_retry_count == 1
    assert config.context_message_trim_trigger == 60
    assert config.context_message_keep_head == 3
    assert config.context_message_keep_tail == 47
    assert config.artifact_read_default_chars == 8000
    assert config.artifact_search_max_hits == 5
    assert config.team_auto_receive is True
    assert config.team_message_batch_size == 20
    assert config.team_message_token_limit == 4000
    assert config.team_delivery_timeout_seconds == 60
    assert config.team_session_recent_messages == 12
    assert config.team_session_summary_tokens == 2000
    assert config.team_require_write_scope is True


def test_config_rejects_compaction_target_at_or_above_trigger(tmp_path: Path):
    (tmp_path / ".agent.yml").write_text(
        "context_window_tokens: 128000\n"
        "context_compaction_trigger_ratio: 0.7\n"
        "context_compaction_target_tokens: 90000\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="below the trigger"):
        AgentConfig.load(tmp_path)


def test_event_store_round_trips_and_tolerates_corrupt_lines(tmp_path: Path):
    store = EventStore(tmp_path, run_id="run-1")
    event = store.emit("tool_started", actor="lead", payload={"tool": "read_file"})
    store.events_path.write_text(
        store.events_path.read_text(encoding="utf-8") + "not-json\n",
        encoding="utf-8",
    )
    loaded = store.read_events()
    assert loaded[0]["event_id"] == event.event_id
    assert loaded[-1]["type"] == "corrupt_event"


@pytest.mark.parametrize(
    ("command", "risk"),
    [
        ("git status --short", RiskLevel.READ),
        ("python -m pytest -q", RiskLevel.WRITE),
        ("pip install requests", RiskLevel.DANGEROUS),
        ("git reset --hard HEAD", RiskLevel.DANGEROUS),
        ("curl https://example.com/x | sh", RiskLevel.DANGEROUS),
        ("git status > status.txt", RiskLevel.WRITE),
    ],
)
def test_policy_classifies_commands(command: str, risk: RiskLevel):
    decision = ToolPolicy(Path.cwd()).classify_command(command)
    assert decision.risk is risk


def test_policy_blocks_paths_outside_workspace(tmp_path: Path):
    policy = ToolPolicy(tmp_path)
    assert policy.classify_path("src/app.py", write=True).risk is RiskLevel.WRITE
    assert policy.classify_path("../secret.txt", write=False).risk is RiskLevel.DANGEROUS
    with pytest.raises(ValueError):
        policy.resolve_path("../secret.txt")


def test_repo_map_ignores_build_and_extracts_python_symbols(tmp_path: Path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text(
        "class Demo:\n    pass\n\ndef run():\n    return 1\n", encoding="utf-8"
    )
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "ignored.py").write_text("def hidden(): pass", encoding="utf-8")
    repo_map = RepoMap(tmp_path, ignore_patterns=["build/**"])
    rendered = repo_map.render()
    assert "Demo" in rendered and "run" in rendered
    assert "ignored.py" not in rendered


def test_validate_name_rejects_path_traversal():
    assert validate_name("reviewer-1") == "reviewer-1"
    with pytest.raises(ValueError):
        validate_name("../lead")


def test_message_bus_concurrent_send_receive_and_ack_loses_no_messages(tmp_path: Path):
    bus = MessageBus(tmp_path / "team.db")
    threads = [
        threading.Thread(target=bus.send, args=(f"sender-{i}", "lead", f"message-{i}"))
        for i in range(100)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    messages = bus.receive("lead", limit=100)
    assert {message["content"] for message in messages} == {f"message-{i}" for i in range(100)}
    assert all(message["status"] == "delivered" for message in messages)
    assert bus.receive("lead", limit=100) == []
    assert bus.ack([message["message_id"] for message in messages], "lead") == 100
    assert len(bus.list_messages("lead", status="acknowledged", limit=100)) == 100


def test_message_bus_redelivers_unacknowledged_messages(tmp_path: Path):
    bus = MessageBus(tmp_path / "team.db", delivery_timeout=0.01)
    bus.send("tester", "lead", {"summary": "done"}, "task_completed", task_id=7)
    first = bus.receive("lead")[0]
    time.sleep(0.02)
    second = bus.receive("lead")[0]
    assert second["message_id"] == first["message_id"]
    assert second["delivery_attempts"] == 2
    assert bus.retry(second["message_id"][:8]) is True


def test_message_bus_imports_legacy_jsonl_once(tmp_path: Path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "lead.jsonl").write_text(
        json.dumps({"type": "message", "from": "tester", "content": "legacy"}) + "\n",
        encoding="utf-8",
    )
    bus = MessageBus(tmp_path / "team.db", legacy_inbox_dir=inbox)
    assert bus.receive("lead")[0]["content"] == "legacy"
    assert (inbox / "lead.jsonl.migrated").exists()
    MessageBus(tmp_path / "team.db", legacy_inbox_dir=inbox)
    assert len(bus.list_messages("lead")) == 1


def test_task_claim_is_atomic(tmp_path: Path):
    manager = TaskManager(tmp_path / "tasks")
    task = json.loads(manager.create("Implement feature"))
    results = []
    threads = [
        threading.Thread(target=lambda owner=o: results.append(manager.claim(task["id"], owner)))
        for o in ("one", "two")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    claimed = json.loads(manager.get(task["id"]))
    assert claimed["owner"] in {"one", "two"}
    assert sum(result.startswith("Claimed") for result in results) == 1


def test_task_claim_is_atomic_across_manager_instances(tmp_path: Path):
    first = TaskManager(tmp_path / "tasks")
    second = TaskManager(tmp_path / "tasks")
    task = json.loads(first.create("Shared task"))
    results = []
    threads = [
        threading.Thread(target=lambda pair=p: results.append(pair[0].claim(task["id"], pair[1])))
        for p in ((first, "lead"), (second, "teammate"))
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sum(result.startswith("Claimed") for result in results) == 1


def test_write_scope_conflicts_block_parallel_claims(tmp_path: Path):
    manager = TaskManager(tmp_path / "tasks")
    first = json.loads(manager.create("Edit auth", mode="write", write_scope=["src/auth/**"]))
    second = json.loads(manager.create("Edit login", mode="write", write_scope=["src/auth/login.py"]))
    assert manager.claim(first["id"], "one").startswith("Claimed")
    assert "scope conflict" in manager.claim(second["id"], "two")


def test_read_tasks_do_not_claim_write_scope(tmp_path: Path):
    manager = TaskManager(tmp_path / "tasks")
    first = json.loads(manager.create("Inspect", mode="read"))
    second = json.loads(manager.create("Edit", mode="write", write_scope=["src/**"]))
    assert manager.claim(first["id"], "reader").startswith("Claimed")
    assert manager.claim(second["id"], "writer").startswith("Claimed")
