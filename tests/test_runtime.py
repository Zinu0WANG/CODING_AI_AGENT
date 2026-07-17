import json
from pathlib import Path

from coding_agent.config import AgentConfig
from coding_agent.runtime import AgentRuntime, FakeModel


def test_fake_model_run_writes_file_records_events_and_validates(tmp_path: Path):
    config = AgentConfig(test_commands=[], lint_commands=[], approval_policy="allow_write")
    model = FakeModel(
        [
            {
                "stop_reason": "tool_use",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool-1",
                        "name": "write_file",
                        "input": {"path": "hello.py", "content": "print('hello')\n"},
                    }
                ],
            },
            {"stop_reason": "end_turn", "content": [{"type": "text", "text": "Done"}]},
        ]
    )
    runtime = AgentRuntime(tmp_path, config=config, model_client=model, interactive=False)
    result = runtime.run("Create hello.py")
    assert result.status == "completed"
    assert (tmp_path / "hello.py").read_text(encoding="utf-8") == "print('hello')\n"
    event_types = [event["type"] for event in runtime.events.read_events()]
    assert "tool_requested" in event_types
    assert "tool_finished" in event_types
    assert "validation_finished" in event_types
    assert event_types[-1] == "run_completed"


def test_replay_only_reads_events(tmp_path: Path):
    runtime = AgentRuntime(
        tmp_path,
        config=AgentConfig(approval_policy="allow_write"),
        model_client=FakeModel([{"stop_reason": "end_turn", "content": [{"type": "text", "text": "ok"}]}]),
        interactive=False,
    )
    runtime.run("Do nothing")
    before = list(tmp_path.rglob("*"))
    replayed = runtime.events.read_events()
    after = list(tmp_path.rglob("*"))
    assert replayed
    assert before == after


def test_dangerous_model_command_is_denied_even_when_writes_are_allowed(tmp_path: Path):
    model = FakeModel([
        {"stop_reason": "tool_use", "content": [{"type": "tool_use", "id": "bad", "name": "bash", "input": {"command": "pip install definitely-not-a-package"}}]},
        {"stop_reason": "end_turn", "content": [{"type": "text", "text": "Could not install"}]},
    ])
    runtime = AgentRuntime(tmp_path, AgentConfig(approval_policy="allow_write"), model, interactive=False)
    result = runtime.run("Install a package")
    tool_event = next(event for event in runtime.events.read_events() if event["type"] == "tool_finished")
    assert "dangerous operation denied" in tool_event["payload"]["output"]
    assert result.status == "completed"


def test_quality_gate_failure_gets_two_fix_attempts_then_fails(tmp_path: Path):
    responses = [
        {"stop_reason": "end_turn", "content": [{"type": "text", "text": f"attempt {i}"}]}
        for i in range(3)
    ]
    config = AgentConfig(test_commands=["python -c \"raise SystemExit(1)\""], approval_policy="allow_write", max_fix_attempts=2)
    runtime = AgentRuntime(tmp_path, config, FakeModel(responses), interactive=False)
    result = runtime.run("Make a passing change")
    events = runtime.events.read_events()
    assert result.status == "failed"
    assert sum(event["type"] == "validation_finished" for event in events) == 3
    assert events[-1]["type"] == "run_failed"


def test_write_approval_can_be_rejected(tmp_path: Path):
    model = FakeModel([
        {"stop_reason": "tool_use", "content": [{"type": "tool_use", "id": "write", "name": "write_file", "input": {"path": "denied.txt", "content": "no"}}]},
        {"stop_reason": "end_turn", "content": [{"type": "text", "text": "Denied"}]},
    ])
    runtime = AgentRuntime(tmp_path, AgentConfig(approval_policy="ask_on_write"), model,
                           approval_callback=lambda *_: False, interactive=False)
    runtime.run("Write a file")
    assert not (tmp_path / "denied.txt").exists()
    event_types = [event["type"] for event in runtime.events.read_events()]
    assert "approval_requested" in event_types and "approval_resolved" in event_types


def test_diff_captures_files_changed_by_shell_commands(tmp_path: Path):
    model = FakeModel([
        {"stop_reason": "tool_use", "content": [{"type": "tool_use", "id": "shell", "name": "bash",
         "input": {"command": "python -c \"from pathlib import Path; Path('shell.txt').write_text('created')\""}}]},
        {"stop_reason": "end_turn", "content": [{"type": "text", "text": "Done"}]},
    ])
    runtime = AgentRuntime(tmp_path, AgentConfig(approval_policy="allow_write"), model, interactive=False)
    result = runtime.run("Create via shell")
    assert "shell.txt" in result.diff
    assert "+created" in result.diff


class ArtifactRetrievalModel:
    def __init__(self):
        self.call = 0
        self.saw_externalized_context = False
        self.artifact_id = None

    def create(self, **kwargs):
        self.call += 1
        messages = kwargs["messages"]
        if self.call <= 4:
            return {"stop_reason": "tool_use", "content": [{"type": "tool_use", "id": f"read-{self.call}",
                    "name": "read_file", "input": {"path": "large.txt", "reason": "integration test"}}]}
        if self.call == 5:
            first_result = messages[2]["content"][0]
            self.saw_externalized_context = first_result["tool_use_id"] == "read-1" and "[artifact:" in first_result["content"]
            return {"stop_reason": "tool_use", "content": [{"type": "tool_use", "id": "search",
                    "name": "artifact_search", "input": {"query": "UNIQUE-MARKER"}}]}
        if self.call == 6:
            search_result = json.loads(messages[-1]["content"][0]["content"])
            self.artifact_id = search_result[0]["artifact_id"]
            return {"stop_reason": "tool_use", "content": [{"type": "tool_use", "id": "artifact-read",
                    "name": "artifact_read", "input": {"artifact_id": self.artifact_id, "offset": 0, "limit": 100}}]}
        assert "UNIQUE-MARKER" in json.loads(messages[-1]["content"][0]["content"])["content"]
        return {"stop_reason": "end_turn", "content": [{"type": "text", "text": "Recovered artifact"}]}


def test_runtime_externalizes_searches_and_reads_old_tool_results(tmp_path: Path):
    (tmp_path / "large.txt").write_text("UNIQUE-MARKER " + "x" * 5000, encoding="utf-8")
    model = ArtifactRetrievalModel()
    config = AgentConfig(approval_policy="allow_write", context_keep_tool_batches=3,
                         artifact_threshold_tokens=1000)
    runtime = AgentRuntime(tmp_path, config, model, interactive=False)
    result = runtime.run("Read and later recover a large result")

    assert result.status == "completed"
    assert result.answer == "Recovered artifact"
    assert model.saw_externalized_context
    assert model.artifact_id
    event_types = [event["type"] for event in runtime.events.read_events()]
    assert "artifact_created" in event_types
    assert "context_externalized" in event_types
    assert event_types.count("artifact_accessed") == 2


class CompactingModel:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["tools"] == []:
            return {"stop_reason": "end_turn", "content": [{"type": "text", "text": "Goal and decisions preserved"}]}
        return {"stop_reason": "end_turn", "content": [{"type": "text", "text": "Finished after compaction"}]}


def test_runtime_compacts_large_prompt_with_toolless_summary_call(tmp_path: Path):
    model = CompactingModel()
    config = AgentConfig(
        approval_policy="allow_write", context_window_tokens=16000,
        context_compaction_trigger_ratio=0.5, context_compaction_target_tokens=4000,
        context_summary_max_tokens=1000,
    )
    runtime = AgentRuntime(tmp_path, config, model, interactive=False)
    result = runtime.run("PRESERVE-ME " + "x" * 1000)

    assert result.status == "completed"
    assert len(model.calls) == 2
    assert model.calls[0]["tools"] == []
    assert model.calls[0]["max_tokens"] <= 1000
    assert "Goal and decisions preserved" in model.calls[1]["messages"][0]["content"]
    assert any(item.kind == "context_archive" for item in runtime.artifacts.list_metadata())
    event_types = [event["type"] for event in runtime.events.read_events()]
    assert "context_compaction_started" in event_types
    assert "context_compaction_completed" in event_types
