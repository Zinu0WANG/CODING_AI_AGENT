import json
from pathlib import Path

import pytest

from coding_agent.context_management import ArtifactStore, ContextManager, estimate_tokens
from coding_agent.events import EventStore


def make_result(tool_id: str, content: str) -> dict:
    return {"type": "tool_result", "tool_use_id": tool_id, "content": content}


def test_estimate_tokens_is_conservative_for_english_chinese_and_mixed_text():
    assert estimate_tokens("a" * 400) == 100
    assert estimate_tokens("中文测试") == 4
    assert estimate_tokens("中" + "a" * 8) == 3


def test_artifact_store_writes_index_reads_pages_and_searches(tmp_path: Path):
    events = EventStore(tmp_path, "run-a")
    store = ArtifactStore(events.run_dir, events)
    artifact = store.create("read_file", "prefix NEEDLE suffix" * 100, "success", 500)

    page = store.read(artifact.artifact_id, offset=7, limit=12)
    assert page["content"] == "NEEDLE suffi"
    assert page["next_offset"] == 19
    assert page["total_chars"] == len("prefix NEEDLE suffix" * 100)

    hits = store.search("needle", max_hits=5)
    assert hits[0]["artifact_id"] == artifact.artifact_id
    assert "NEEDLE" in hits[0]["snippet"]
    index = [json.loads(line) for line in store.index_path.read_text(encoding="utf-8").splitlines()]
    assert index[0]["sha256"] == artifact.sha256


def test_artifact_store_rejects_missing_corrupt_and_cross_run_ids(tmp_path: Path):
    first_events = EventStore(tmp_path, "first")
    first = ArtifactStore(first_events.run_dir, first_events)
    artifact = first.create("bash", "secret output", "success", 3)
    second_events = EventStore(tmp_path, "second")
    second = ArtifactStore(second_events.run_dir, second_events)

    with pytest.raises(ValueError, match="unknown artifact"):
        second.read(artifact.artifact_id)
    first.artifact_path(artifact.artifact_id).write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        first.read(artifact.artifact_id)
    with pytest.raises(ValueError, match="unknown artifact"):
        first.read("not-an-id")


def test_context_manager_externalizes_only_large_results_older_than_three_batches(tmp_path: Path):
    events = EventStore(tmp_path, "run")
    manager = ContextManager(ArtifactStore(events.run_dir, events), events, keep_batches=3, threshold_tokens=10)
    results = []
    for index in range(4):
        result = make_result(f"id-{index}", f"result-{index}-" + "x" * 80)
        results.append(result)
        manager.register_batch([("read_file", result)])
    manager.compact()

    assert results[0]["tool_use_id"] == "id-0"
    assert "[artifact:" in results[0]["content"]
    assert "tool=read_file" in results[0]["content"]
    assert all("[artifact:" not in result["content"] for result in results[1:])


def test_context_manager_keeps_small_old_results_and_groups_parallel_tools(tmp_path: Path):
    events = EventStore(tmp_path, "run")
    manager = ContextManager(ArtifactStore(events.run_dir, events), events, keep_batches=1, threshold_tokens=10)
    small = make_result("small", "tiny")
    large_a = make_result("large-a", "a" * 100)
    large_b = make_result("large-b", "b" * 100)
    newest = make_result("newest", "c" * 100)
    manager.register_batch([("read_file", small)])
    manager.register_batch([("bash", large_a), ("read_file", large_b)])
    manager.register_batch([("read_file", newest)])
    manager.compact()

    assert small["content"] == "tiny"
    assert "[artifact:" in large_a["content"]
    assert "[artifact:" in large_b["content"]
    assert "[artifact:" not in newest["content"]
    manager.compact()
    assert len(manager.store.list_metadata()) == 2
