import json
import threading
from pathlib import Path

import pytest

from coding_agent.plans import PlanStore


PLAN_TEXT = "\n".join([
    "## 目标与验收标准", "## 仓库现状", "## 实施步骤",
    "## 预计修改文件及原因", "## 测试方案", "## 风险与假设",
])


def test_plan_store_persists_plan_and_ignores_internal_runtime_files(tmp_path: Path):
    (tmp_path / "app.py").write_text("value = 1\n", encoding="utf-8")
    store = PlanStore(tmp_path, [".runs/**", ".plans/**"])
    plan = store.create("change app", PLAN_TEXT, "run-plan", ["app.py"])

    assert len(plan["plan_id"]) == 8
    assert plan["status"] == "ready"
    assert store.load(plan["plan_id"])["original_request"] == "change app"
    original = plan["workspace_fingerprint"]

    (tmp_path / ".runs" / "new").mkdir(parents=True)
    (tmp_path / ".runs" / "new" / "events.jsonl").write_text("event", encoding="utf-8")
    (tmp_path / ".plans" / "extra.tmp").write_text("internal", encoding="utf-8")
    assert store.workspace_fingerprint() == original

    (tmp_path / "app.py").write_text("value = 2\n", encoding="utf-8")
    assert store.workspace_fingerprint() != original


def test_plan_store_allows_only_one_atomic_begin(tmp_path: Path):
    store = PlanStore(tmp_path, [".plans/**"])
    plan = store.create("request", PLAN_TEXT, "planning-run", [])
    outcomes = []

    def begin():
        try:
            outcomes.append(store.begin(plan["plan_id"])["status"])
        except ValueError as exc:
            outcomes.append(str(exc))

    threads = [threading.Thread(target=begin) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert outcomes.count("executing") == 1
    assert sum("must be ready" in item for item in outcomes) == 1


def test_plan_store_marks_changed_workspace_stale(tmp_path: Path):
    (tmp_path / "app.py").write_text("before", encoding="utf-8")
    store = PlanStore(tmp_path, [".plans/**"])
    plan = store.create("request", PLAN_TEXT, "planning-run", ["app.py"])
    (tmp_path / "app.py").write_text("after", encoding="utf-8")

    stale = store.begin(plan["plan_id"])
    assert stale["status"] == "stale"
    assert store.load(plan["plan_id"])["status"] == "stale"


def test_plan_store_rejects_invalid_unknown_and_corrupt_ids(tmp_path: Path):
    store = PlanStore(tmp_path, [".plans/**"])
    with pytest.raises(ValueError, match="invalid plan ID"):
        store.load("../escape")
    with pytest.raises(ValueError, match="not found"):
        store.load("deadbeef")

    corrupt = tmp_path / ".plans" / "cafebabe.json"
    corrupt.write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt"):
        store.load("cafebabe")


def test_plan_store_finishes_with_implementation_run(tmp_path: Path):
    store = PlanStore(tmp_path, [".plans/**"])
    plan = store.create("request", PLAN_TEXT, "planning-run", [])
    store.begin(plan["plan_id"])
    completed = store.finish(plan["plan_id"], "completed", "implementation-run")
    assert completed["status"] == "completed"
    assert completed["implementation_run_id"] == "implementation-run"


def test_plan_store_rejects_plan_missing_required_sections(tmp_path: Path):
    store = PlanStore(tmp_path, [".plans/**"])
    with pytest.raises(ValueError, match="missing required sections"):
        store.create("request", "## 实施步骤\n1. edit", "planning-run", [])
    assert store.list_all() == []
