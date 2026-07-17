from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path


NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def validate_name(value: str) -> str:
    if not NAME_PATTERN.fullmatch(value or ""):
        raise ValueError("Name must be 1-64 ASCII letters, digits, underscores, or hyphens")
    return value


class MessageBus:
    def __init__(self, inbox_dir: Path):
        self.inbox_dir = inbox_dir
        self.inbox_dir.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def _lock(self, name: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(name, threading.Lock())

    def send(self, sender: str, to: str, content: str, msg_type: str = "message", extra: dict | None = None) -> str:
        sender, to = validate_name(sender), validate_name(to)
        if msg_type not in {"message", "broadcast", "shutdown_request", "shutdown_response", "plan_approval_response"}:
            raise ValueError(f"Invalid message type: {msg_type}")
        message = {"type": msg_type, "from": sender, "content": str(content)[:50_000], "timestamp": time.time()}
        if extra:
            message.update(extra)
        path = self.inbox_dir / f"{to}.jsonl"
        with self._lock(to), path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(message, ensure_ascii=False) + "\n")
            stream.flush()
        return f"Sent {msg_type} to {to}"

    def read_inbox(self, name: str) -> list[dict]:
        name = validate_name(name)
        path = self.inbox_dir / f"{name}.jsonl"
        with self._lock(name):
            if not path.exists():
                return []
            lines = path.read_text(encoding="utf-8").splitlines()
            path.write_text("", encoding="utf-8")
        messages = []
        for line in lines:
            if not line:
                continue
            try:
                messages.append(json.loads(line))
            except json.JSONDecodeError:
                messages.append({"type": "corrupt_message", "content": line[:500]})
        return messages

    def broadcast(self, sender: str, content: str, names: list[str]) -> str:
        count = 0
        for name in names:
            if name != sender:
                self.send(sender, name, content, "broadcast")
                count += 1
        return f"Broadcast to {count} teammates"


class TaskManager:
    def __init__(self, tasks_dir: Path):
        self.tasks_dir = tasks_dir
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _path(self, task_id: int) -> Path:
        if not isinstance(task_id, int) or task_id <= 0:
            raise ValueError("task_id must be a positive integer")
        return self.tasks_dir / f"task_{task_id}.json"

    def _next_id(self) -> int:
        ids = []
        for path in self.tasks_dir.glob("task_*.json"):
            try:
                ids.append(int(path.stem.split("_", 1)[1]))
            except ValueError:
                continue
        return max(ids, default=0) + 1

    def _load(self, task_id: int) -> dict:
        path = self._path(task_id)
        if not path.exists():
            raise ValueError(f"Task {task_id} not found")
        return json.loads(path.read_text(encoding="utf-8"))

    def _save(self, task: dict) -> None:
        path = self._path(task["id"])
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(task, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)

    def create(self, subject: str, description: str = "") -> str:
        if not subject.strip():
            raise ValueError("subject is required")
        with self._lock:
            task = {"id": self._next_id(), "subject": subject.strip(), "description": description,
                    "status": "pending", "owner": None, "blockedBy": []}
            self._save(task)
        return json.dumps(task, indent=2, ensure_ascii=False)

    def get(self, task_id: int) -> str:
        with self._lock:
            return json.dumps(self._load(task_id), indent=2, ensure_ascii=False)

    def update(self, task_id: int, status: str | None = None, add_blocked_by: list[int] | None = None,
               remove_blocked_by: list[int] | None = None) -> str:
        with self._lock:
            task = self._load(task_id)
            if status not in {None, "pending", "in_progress", "completed", "deleted"}:
                raise ValueError(f"Invalid status: {status}")
            if status == "deleted":
                self._path(task_id).unlink()
                return f"Task {task_id} deleted"
            if status:
                task["status"] = status
            if add_blocked_by:
                for blocker in add_blocked_by:
                    self._load(blocker)
                    if blocker == task_id:
                        raise ValueError("A task cannot block itself")
                task["blockedBy"] = sorted(set(task["blockedBy"] + add_blocked_by))
            if remove_blocked_by:
                task["blockedBy"] = [item for item in task["blockedBy"] if item not in remove_blocked_by]
            self._save(task)
            if status == "completed":
                for path in self.tasks_dir.glob("task_*.json"):
                    dependent = json.loads(path.read_text(encoding="utf-8"))
                    if task_id in dependent.get("blockedBy", []):
                        dependent["blockedBy"].remove(task_id)
                        self._save(dependent)
            return json.dumps(task, indent=2, ensure_ascii=False)

    def claim(self, task_id: int, owner: str) -> str:
        owner = validate_name(owner)
        with self._lock:
            task = self._load(task_id)
            if task["status"] != "pending" or task.get("owner") or task.get("blockedBy"):
                return f"Task #{task_id} is not available"
            task.update({"owner": owner, "status": "in_progress"})
            self._save(task)
            return f"Claimed task #{task_id} for {owner}"

    def list_all(self) -> str:
        with self._lock:
            tasks = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(self.tasks_dir.glob("task_*.json"))]
        if not tasks:
            return "No tasks."
        lines = []
        for task in tasks:
            marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}.get(task["status"], "[?]")
            owner = f" @{task['owner']}" if task.get("owner") else ""
            blocked = f" (blocked by: {task['blockedBy']})" if task.get("blockedBy") else ""
            lines.append(f"{marker} #{task['id']}: {task['subject']}{owner}{blocked}")
        return "\n".join(lines)

    def next_available(self) -> dict | None:
        with self._lock:
            for path in sorted(self.tasks_dir.glob("task_*.json")):
                task = json.loads(path.read_text(encoding="utf-8"))
                if task["status"] == "pending" and not task.get("owner") and not task.get("blockedBy"):
                    return task
        return None
