import json
from pathlib import Path

import pytest

from coding_agent.context_management import (
    ArtifactStore,
    ContextManager,
    ConversationCompactor,
    MessageCountTrimmer,
    estimate_request_tokens,
    estimate_tokens,
)
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


def test_context_archive_is_searchable_and_exposes_safe_metadata(tmp_path: Path):
    events = EventStore(tmp_path, "run-archive")
    store = ArtifactStore(events.run_dir, events)
    archive = store.create(
        "context_compactor", '{"messages":[{"role":"user","content":"OLD-DECISION"}]}',
        "success", kind="context_archive",
        details={"message_start": 0, "message_end": 1, "roles": ["user"]},
    )

    hit = store.search("old-decision")[0]
    assert hit["artifact_id"] == archive.artifact_id
    assert hit["kind"] == "context_archive"
    assert hit["details"]["roles"] == ["user"]
    assert store.read(archive.artifact_id)["kind"] == "context_archive"


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
    with pytest.raises(ValueError, match="query must be at most"):
        first.search("x" * 201)


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


def test_context_manager_marks_failed_shell_results_as_errors(tmp_path: Path):
    events = EventStore(tmp_path, "run")
    store = ArtifactStore(events.run_dir, events)
    manager = ContextManager(store, events, keep_batches=0, threshold_tokens=1)
    failed = make_result("failed", "exit_code=2\n" + "failure " * 20)
    manager.register_batch([("bash", failed)])
    manager.compact()
    assert "status=error" in failed["content"]
    assert store.list_metadata()[0].status == "error"


def test_request_estimate_includes_system_tools_messages_and_output_reserve():
    estimated = estimate_request_tokens(
        "s" * 40,
        [{"role": "user", "content": "m" * 40}],
        [{"name": "tool", "description": "t" * 40}],
        output_reserve_tokens=8000,
    )
    assert estimated >= 8030


def test_compactor_does_nothing_below_seventy_percent_threshold(tmp_path: Path):
    events = EventStore(tmp_path, "below-threshold")
    manager = ContextManager(ArtifactStore(events.run_dir, events), events)
    messages = [{"role": "user", "content": "small request"}]
    compactor = ConversationCompactor(
        manager, events, window_tokens=128000, trigger_ratio=0.70,
        target_tokens=25000, summary_max_tokens=12000, summary_retry_count=1,
    )

    unchanged = compactor.compact_if_needed(
        "system", messages, [], lambda _archive, _limit: "must not run",
    )

    assert unchanged is messages
    assert not manager.store.list_metadata()
    assert not any(event["type"].startswith("context_compaction") for event in events.read_events())


def test_compactor_archives_complete_tool_pair_and_injects_summary(tmp_path: Path):
    events = EventStore(tmp_path, "compact")
    manager = ContextManager(ArtifactStore(events.run_dir, events), events, keep_batches=1, threshold_tokens=10)
    old_result = make_result("old-tool", "OLD-RESULT " + "x" * 300)
    new_result = make_result("new-tool", "NEW-RESULT " + "y" * 80)
    messages = [
        {"role": "user", "content": "ORIGINAL-GOAL " + "g" * 200},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "old-tool", "name": "read_file", "input": {}}]},
        {"role": "user", "content": [old_result]},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "new-tool", "name": "read_file", "input": {}}]},
        {"role": "user", "content": [new_result]},
    ]
    manager.register_batch([("read_file", old_result)])
    manager.register_batch([("read_file", new_result)])
    compactor = ConversationCompactor(
        manager, events, window_tokens=600, trigger_ratio=0.2,
        target_tokens=220, summary_max_tokens=80, summary_retry_count=1,
        output_reserve_tokens=20,
    )

    compacted = compactor.compact_if_needed(
        "system", messages, [], lambda archive, _limit: "Structured summary for ORIGINAL-GOAL",
    )

    assert compacted is not messages
    assert "Structured summary" in compacted[0]["content"]
    archives = [item for item in manager.store.list_metadata() if item.kind == "context_archive"]
    assert len(archives) == 1
    archived = manager.store.read(archives[0].artifact_id)["content"]
    assert "ORIGINAL-GOAL" in archived
    assert ('"id": "old-tool"' in archived) == ('"tool_use_id": "old-tool"' in archived)
    event_types = [event["type"] for event in events.read_events()]
    assert event_types.index("context_archive_created") < event_types.index("context_compaction_completed")


def test_compactor_retries_then_uses_local_fallback_without_losing_archive(tmp_path: Path):
    events = EventStore(tmp_path, "fallback")
    manager = ContextManager(ArtifactStore(events.run_dir, events), events, keep_batches=0, threshold_tokens=10)
    compactor = ConversationCompactor(
        manager, events, window_tokens=200, trigger_ratio=0.5,
        target_tokens=80, summary_max_tokens=30, summary_retry_count=1,
        output_reserve_tokens=10,
    )
    calls = 0

    def fail_summary(_archive: str, _limit: int) -> str:
        nonlocal calls
        calls += 1
        raise RuntimeError("summary unavailable")

    compacted = compactor.compact_if_needed(
        "system", [{"role": "user", "content": "IMPORTANT " + "x" * 500}], [], fail_summary,
    )

    assert calls == 2
    assert "Automatic fallback summary" in compacted[0]["content"]
    assert "artifact" in compacted[0]["content"]
    assert any(item.kind == "context_archive" for item in manager.store.list_metadata())
    assert "context_compaction_failed" in [event["type"] for event in events.read_events()]


def test_compactor_bounds_summary_request_but_keeps_complete_archive(tmp_path: Path):
    events = EventStore(tmp_path, "oversized")
    manager = ContextManager(ArtifactStore(events.run_dir, events), events)
    compactor = ConversationCompactor(
        manager, events, window_tokens=16000, trigger_ratio=0.5,
        target_tokens=4000, summary_max_tokens=1000, summary_retry_count=0,
        output_reserve_tokens=100,
    )
    original = "HEAD-MARKER " + "x" * 80000 + " TAIL-MARKER"
    received = ""

    def summarize(archive: str, _limit: int) -> str:
        nonlocal received
        received = archive
        return "bounded summary"

    compactor.compact_if_needed("system", [{"role": "user", "content": original}], [], summarize)

    assert "HEAD-MARKER" in received and "TAIL-MARKER" in received
    assert "middle omitted" in received
    archive = next(item for item in manager.store.list_metadata() if item.kind == "context_archive")
    assert original in manager.store.artifact_path(archive.artifact_id).read_text(encoding="utf-8")


def _tool_pair(tool_id: str) -> list[dict]:
    return [
        {"role": "assistant", "content": [{"type": "tool_use", "id": tool_id,
          "name": "read_file", "input": {"path": f"{tool_id}.txt"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_id,
          "content": f"result-{tool_id}"}]},
    ]


def test_message_trimmer_ignores_sixty_and_trims_sixty_one(tmp_path: Path):
    events = EventStore(tmp_path, "message-limit")
    manager = ContextManager(ArtifactStore(events.run_dir, events), events)
    trimmer = MessageCountTrimmer(manager, events, trigger=60, keep_head=3, keep_tail=47)
    sixty = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m-{i}"} for i in range(60)]
    assert trimmer.trim_if_needed(sixty) is sixty

    sixty_one = sixty + [{"role": "user", "content": "m-60"}]
    trimmed = trimmer.trim_if_needed(sixty_one)
    assert trimmed is not sixty_one
    assert len(trimmed) == 50
    assert trimmed[0]["content"] == "m-0"
    assert trimmed[-1]["content"] == "m-60"
    assert any(item.kind == "context_archive" for item in manager.store.list_metadata())


def test_message_trimmer_keeps_head_three_tail_forty_seven_and_archives_middle(tmp_path: Path):
    events = EventStore(tmp_path, "message-160")
    manager = ContextManager(ArtifactStore(events.run_dir, events), events)
    trimmer = MessageCountTrimmer(manager, events, trigger=60, keep_head=3, keep_tail=47)
    messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"MESSAGE-{i}"} for i in range(160)]

    trimmed = trimmer.trim_if_needed(messages)

    assert len(trimmed) == 50
    assert [message["content"].splitlines()[0] for message in trimmed[:3]] == ["MESSAGE-0", "MESSAGE-1", "MESSAGE-2"]
    assert [message["content"] for message in trimmed[-47:]] == [f"MESSAGE-{i}" for i in range(113, 160)]
    assert "trimmed context archive:" in trimmed[2]["content"]
    archive = next(item for item in manager.store.list_metadata() if item.details.get("reason") == "message_count_trim")
    archived = json.loads(manager.store.artifact_path(archive.artifact_id).read_text(encoding="utf-8"))
    archived_contents = [message["content"] for message in archived["messages"]]
    assert archived_contents[0] == "MESSAGE-3" and archived_contents[-1] == "MESSAGE-112"
    assert "MESSAGE-2" not in archived_contents and "MESSAGE-113" not in archived_contents
    assert manager.store.search("MESSAGE-50")[0]["artifact_id"] == archive.artifact_id


def test_message_trimmer_expands_head_and_tail_to_keep_tool_pairs(tmp_path: Path):
    events = EventStore(tmp_path, "pair-boundaries")
    manager = ContextManager(ArtifactStore(events.run_dir, events), events)
    trimmer = MessageCountTrimmer(manager, events, trigger=60, keep_head=3, keep_tail=47)
    messages = [{"role": "user", "content": "initial"}, *_tool_pair("head")]
    # Put another result exactly at the nominal tail boundary: assistant at 15, result at 16.
    messages.extend({"role": "assistant" if i % 2 else "user", "content": f"filler-{i}"} for i in range(12))
    messages.extend(_tool_pair("tail"))
    messages.extend({"role": "assistant" if i % 2 else "user", "content": f"recent-{i}"} for i in range(46))
    assert len(messages) == 63

    trimmed = trimmer.trim_if_needed(messages)

    flattened = json.dumps(trimmed, ensure_ascii=False)
    assert '"id": "head"' in flattened and '"tool_use_id": "head"' in flattened
    assert ('"id": "tail"' in flattened) == ('"tool_use_id": "tail"' in flattened)
    completed = next(event for event in events.read_events() if event["type"] == "context_message_trim_completed")
    assert completed["payload"]["boundary_expanded"] is True


def test_message_trimmer_skips_malformed_tool_protocol_without_archiving(tmp_path: Path):
    events = EventStore(tmp_path, "malformed")
    manager = ContextManager(ArtifactStore(events.run_dir, events), events)
    trimmer = MessageCountTrimmer(manager, events, trigger=60, keep_head=3, keep_tail=47)
    messages = [{"role": "user", "content": "initial"}, *_tool_pair("valid")]
    messages.extend({"role": "user", "content": f"m-{i}"} for i in range(58))
    messages[2]["content"][0]["tool_use_id"] = "wrong-id"

    unchanged = trimmer.trim_if_needed(messages)

    assert unchanged is messages
    assert not manager.store.list_metadata()
    skipped = next(event for event in events.read_events() if event["type"] == "context_message_trim_skipped")
    assert skipped["payload"]["reason"] == "invalid_tool_protocol"


def test_message_trimmer_supports_repeated_archives_and_releases_batch_references(tmp_path: Path):
    events = EventStore(tmp_path, "repeated")
    manager = ContextManager(ArtifactStore(events.run_dir, events), events)
    trimmer = MessageCountTrimmer(manager, events, trigger=60, keep_head=3, keep_tail=47)
    messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"first-{i}"} for i in range(80)]
    first = trimmer.trim_if_needed(messages)
    second_input = [*first, *(
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"second-{i}"}
        for i in range(11)
    )]
    second = trimmer.trim_if_needed(second_input)

    archives = [item for item in manager.store.list_metadata() if item.details.get("reason") == "message_count_trim"]
    assert len(archives) == 2
    assert all(artifact.artifact_id in json.dumps(second, ensure_ascii=False) for artifact in archives)

    result = make_result("forgotten", "small")
    manager.register_batch([("read_file", result)])
    manager.forget_results([{"role": "user", "content": [result]}])
    assert manager.batches == []
