from __future__ import annotations

import fnmatch
import hashlib
import json
import re
import subprocess
import threading
import time
import uuid
from pathlib import Path


PLAN_ID_PATTERN = re.compile(r"^[0-9a-f]{8}$")
PLAN_STATUSES = {"ready", "executing", "completed", "failed", "stale"}
REQUIRED_PLAN_SECTIONS = (
    "目标与验收标准", "仓库现状", "实施步骤", "预计修改文件及原因", "测试方案", "风险与假设",
)
_LOCKS_GUARD = threading.Lock()
_LOCKS: dict[str, threading.RLock] = {}


def _plan_lock(root: Path) -> threading.RLock:
    key = str(root.resolve()).lower()
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(key, threading.RLock())


class PlanStore:
    def __init__(self, workspace: Path, ignore_patterns: list[str], max_file_bytes: int = 250_000):
        self.workspace = workspace.resolve()
        self.root = self.workspace / ".plans"
        self.root.mkdir(parents=True, exist_ok=True)
        self.ignore_patterns = list(dict.fromkeys([*ignore_patterns, ".plans/**"]))
        self.max_file_bytes = max_file_bytes
        self._lock = _plan_lock(self.root)

    @staticmethod
    def _validate_id(plan_id: str) -> str:
        if not PLAN_ID_PATTERN.fullmatch(plan_id or ""):
            raise ValueError("invalid plan ID")
        return plan_id

    def _path(self, plan_id: str) -> Path:
        return self.root / f"{self._validate_id(plan_id)}.json"

    def _ignored(self, relative: str) -> bool:
        normalized = relative.replace("\\", "/")
        return any(
            fnmatch.fnmatch(normalized, pattern) or fnmatch.fnmatch(normalized + "/", pattern)
            for pattern in self.ignore_patterns
        )

    def workspace_fingerprint(self) -> str:
        digest = hashlib.sha256()
        for path in sorted(self.workspace.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(self.workspace).as_posix()
            if self._ignored(relative):
                continue
            stat = path.stat()
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(stat.st_size).encode("ascii"))
            digest.update(b"\0")
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
            digest.update(b"\0")
        return digest.hexdigest()

    def git_head(self) -> str | None:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=self.workspace,
                capture_output=True, text=True, timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    def _save(self, plan: dict) -> None:
        if plan.get("status") not in PLAN_STATUSES:
            raise ValueError("invalid plan status")
        path = self._path(plan["plan_id"])
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)

    def load(self, plan_id: str) -> dict:
        path = self._path(plan_id)
        with self._lock:
            if not path.exists():
                raise ValueError(f"plan {plan_id} not found")
            try:
                plan = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"plan {plan_id} is corrupt") from exc
            if not isinstance(plan, dict) or plan.get("plan_id") != plan_id or plan.get("status") not in PLAN_STATUSES:
                raise ValueError(f"plan {plan_id} is corrupt")
            return plan

    def create(self, original_request: str, plan_text: str, planning_run_id: str,
               selected_files: list[str]) -> dict:
        if not original_request.strip() or not plan_text.strip():
            raise ValueError("request and plan text are required")
        missing = [section for section in REQUIRED_PLAN_SECTIONS if section not in plan_text]
        if missing:
            raise ValueError("plan is missing required sections: " + ", ".join(missing))
        with self._lock:
            for _ in range(20):
                plan_id = uuid.uuid4().hex[:8]
                if not self._path(plan_id).exists():
                    break
            else:
                raise RuntimeError("could not allocate plan ID")
            plan = {
                "plan_id": plan_id,
                "status": "ready",
                "created_at": time.time(),
                "original_request": original_request.strip(),
                "plan": plan_text.strip(),
                "planning_run_id": planning_run_id,
                "implementation_run_id": None,
                "git_head": self.git_head(),
                "workspace_fingerprint": self.workspace_fingerprint(),
                "selected_files": list(dict.fromkeys(selected_files)),
            }
            self._save(plan)
            return plan

    def begin(self, plan_id: str) -> dict:
        with self._lock:
            plan = self.load(plan_id)
            if plan["status"] != "ready":
                raise ValueError(f"plan {plan_id} must be ready, not {plan['status']}")
            if plan["workspace_fingerprint"] != self.workspace_fingerprint() or plan.get("git_head") != self.git_head():
                plan["status"] = "stale"
                self._save(plan)
                return plan
            plan["status"] = "executing"
            self._save(plan)
            return plan

    def finish(self, plan_id: str, status: str, implementation_run_id: str) -> dict:
        if status not in {"completed", "failed"}:
            raise ValueError("finished plan status must be completed or failed")
        with self._lock:
            plan = self.load(plan_id)
            if plan["status"] != "executing":
                raise ValueError(f"plan {plan_id} must be executing, not {plan['status']}")
            plan["status"] = status
            plan["implementation_run_id"] = implementation_run_id
            self._save(plan)
            return plan

    def list_all(self) -> list[dict]:
        plans = []
        with self._lock:
            for path in sorted(self.root.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
                try:
                    plans.append(self.load(path.stem))
                except ValueError:
                    plans.append({"plan_id": path.stem, "status": "corrupt", "created_at": 0,
                                  "original_request": "", "planning_run_id": "", "implementation_run_id": None})
        return plans
