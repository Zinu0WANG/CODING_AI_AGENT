from pathlib import Path
from types import SimpleNamespace

from coding_agent.config import AgentConfig
from coding_agent.events import EventStore
from coding_agent.team import TeammateManager


def test_teammate_restores_compact_session_between_work_items(tmp_path: Path):
    prompts = []

    def run_agent(prompt: str, actor: str, write_scope: list[str]):
        prompts.append((prompt, actor, write_scope))
        return SimpleNamespace(status="completed", answer=f"answer-{len(prompts)}", run_id=f"run-{len(prompts)}")

    manager = TeammateManager(tmp_path, EventStore(tmp_path), run_agent, AgentConfig())
    manager._work("tester", "QA", "first request", 1, ["tests/**"])
    manager._work("tester", "QA", "follow-up request", 1, ["tests/**"])

    assert "PREVIOUS SESSION SUMMARY" not in prompts[0][0]
    assert "PREVIOUS SESSION SUMMARY" in prompts[1][0]
    assert "answer-1" in prompts[1][0]
    session = (tmp_path / ".team" / "sessions" / "tester.json").read_text(encoding="utf-8")
    assert "run-2" in session


def test_corrupt_teammate_session_falls_back_and_records_event(tmp_path: Path):
    events = EventStore(tmp_path)
    manager = TeammateManager(
        tmp_path, events,
        lambda *_: SimpleNamespace(status="completed", answer="ok", run_id="run"),
        AgentConfig(),
    )
    session = tmp_path / ".team" / "sessions" / "reviewer.json"
    session.write_text("not-json", encoding="utf-8")
    manager._work("reviewer", "Review", "continue")
    assert any(event["type"] == "teammate_session_corrupt" for event in events.read_events())
