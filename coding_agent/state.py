from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path


NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_LOCK_REGISTRY_GUARD = threading.Lock()
_LOCK_REGISTRY: dict[str, threading.RLock] = {}


def _shared_lock(path: Path) -> threading.RLock:
    key = str(path.resolve()).lower()
    with _LOCK_REGISTRY_GUARD:
        return _LOCK_REGISTRY.setdefault(key, threading.RLock())


def validate_name(value: str) -> str:
    if not NAME_PATTERN.fullmatch(value or ""):
        raise ValueError("Name must be 1-64 ASCII letters, digits, underscores, or hyphens")
    return value


class MessageBus:
    TYPES = {"instruction", "progress", "task_completed", "task_failed", "blocked", "broadcast",
             "shutdown_request", "shutdown_response"}

    def __init__(self, database: Path, delivery_timeout: float = 60,
                 legacy_inbox_dir: Path | None = None, event_callback=None):
        self.database = database if database.suffix == ".db" else database.parent / "team.db"
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self.delivery_timeout = delivery_timeout
        self.event_callback = event_callback
        self._initialize()
        if legacy_inbox_dir:
            self._import_legacy(legacy_inbox_dir)

    def _connect(self):
        connection = sqlite3.connect(self.database, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS messages (
                message_id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL, task_id INTEGER,
                sender TEXT NOT NULL, recipient TEXT NOT NULL, type TEXT NOT NULL,
                status TEXT NOT NULL, content TEXT NOT NULL, created_at REAL NOT NULL,
                delivered_at REAL, acknowledged_at REAL, delivery_attempts INTEGER NOT NULL DEFAULT 0
            )""")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_messages_recipient_status ON messages(recipient, status, created_at)")

    def _emit(self, event_type: str, actor: str, payload: dict) -> None:
        if self.event_callback:
            self.event_callback(event_type, actor, payload)

    @staticmethod
    def _decode(row) -> dict:
        item = dict(row)
        try:
            decoded = json.loads(item["content"])
            item["content"] = decoded
        except (json.JSONDecodeError, TypeError):
            pass
        return item

    def send(self, sender: str, to: str, content, msg_type: str = "instruction", extra: dict | None = None,
             *, task_id: int | None = None, conversation_id: str | None = None) -> str:
        sender, to = validate_name(sender), validate_name(to)
        if msg_type == "message":
            msg_type = "instruction"
        if msg_type not in self.TYPES:
            raise ValueError(f"Invalid message type: {msg_type}")
        extra = extra or {}
        task_id = task_id if task_id is not None else extra.get("task_id")
        conversation_id = conversation_id or extra.get("conversation_id") or str(uuid.uuid4())
        message_id = str(uuid.uuid4())
        serialized = json.dumps(content, ensure_ascii=False, default=str)
        if len(serialized) > 50_000:
            serialized = json.dumps({"truncated": True, "preview": serialized[:48_000]}, ensure_ascii=False)
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?, NULL, NULL, 0)",
                (message_id, conversation_id, task_id, sender, to, msg_type, serialized, time.time()),
            )
        self._emit("message_sent", sender, {"message_id": message_id, "to": to, "type": msg_type, "task_id": task_id})
        return f"Sent {msg_type} to {to}"

    def receive(self, name: str, limit: int = 20) -> list[dict]:
        name = validate_name(name)
        now, cutoff = time.time(), time.time() - self.delivery_timeout
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            stale = connection.execute(
                "SELECT message_id FROM messages WHERE recipient=? AND status='delivered' AND delivered_at<?",
                (name, cutoff),
            ).fetchall()
            if stale:
                connection.execute(
                    "UPDATE messages SET status='pending', delivered_at=NULL WHERE recipient=? AND status='delivered' AND delivered_at<?",
                    (name, cutoff),
                )
            rows = connection.execute(
                """SELECT * FROM messages WHERE recipient=? AND status='pending'
                   ORDER BY CASE type WHEN 'task_failed' THEN 0 WHEN 'blocked' THEN 1 WHEN 'task_completed' THEN 2 ELSE 3 END,
                            created_at LIMIT ?""", (name, limit),
            ).fetchall()
            ids = [row["message_id"] for row in rows]
            if ids:
                placeholders = ",".join("?" for _ in ids)
                connection.execute(
                    f"UPDATE messages SET status='delivered', delivered_at=?, delivery_attempts=delivery_attempts+1 WHERE message_id IN ({placeholders})",
                    (now, *ids),
                )
                rows = connection.execute(
                    f"SELECT * FROM messages WHERE message_id IN ({placeholders}) ORDER BY created_at", ids,
                ).fetchall()
        for row in stale:
            self._emit("message_redelivered", "runtime", {"message_id": row["message_id"], "recipient": name})
        messages = [self._decode(row) for row in rows]
        for message in messages:
            self._emit("message_delivered", name, {"message_id": message["message_id"], "from": message["sender"]})
        return messages

    def ack(self, message_ids: list[str], recipient: str) -> int:
        recipient = validate_name(recipient)
        if not message_ids:
            return 0
        placeholders = ",".join("?" for _ in message_ids)
        with self._connect() as connection:
            acknowledged = connection.execute(
                f"SELECT message_id FROM messages WHERE recipient=? AND status='delivered' AND message_id IN ({placeholders})",
                (recipient, *message_ids),
            ).fetchall()
            cursor = connection.execute(
                f"UPDATE messages SET status='acknowledged', acknowledged_at=? WHERE recipient=? AND status='delivered' AND message_id IN ({placeholders})",
                (time.time(), recipient, *message_ids),
            )
        for row in acknowledged:
            self._emit("message_acknowledged", recipient, {"message_id": row["message_id"]})
        return cursor.rowcount

    def retry(self, message_id: str) -> bool:
        with self._connect() as connection:
            matches = connection.execute(
                "SELECT message_id FROM messages WHERE message_id LIKE ? AND status!='acknowledged'", (message_id + "%",),
            ).fetchall()
            if len(matches) != 1:
                return False
            cursor = connection.execute(
                "UPDATE messages SET status='pending', delivered_at=NULL WHERE message_id=?", (matches[0]["message_id"],),
            )
        return cursor.rowcount == 1

    def list_messages(self, recipient: str | None = None, status: str | None = None, limit: int = 20) -> list[dict]:
        clauses, values = [], []
        if recipient:
            clauses.append("recipient=?")
            values.append(validate_name(recipient))
        if status:
            if status not in {"pending", "delivered", "acknowledged"}:
                raise ValueError("invalid message status")
            clauses.append("status=?")
            values.append(status)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self._connect() as connection:
            rows = connection.execute(f"SELECT * FROM messages{where} ORDER BY created_at DESC LIMIT ?", (*values, limit)).fetchall()
        return [self._decode(row) for row in rows]

    def read_inbox(self, name: str, status: str | None = None, limit: int = 20) -> list[dict]:
        return self.list_messages(name, status, limit)

    def _import_legacy(self, inbox_dir: Path) -> None:
        for path in inbox_dir.glob("*.jsonl"):
            backup = path.with_suffix(path.suffix + ".migrated")
            if backup.exists():
                continue
            recipient = validate_name(path.stem)
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    old = json.loads(line)
                    self.send(old.get("from", "runtime"), recipient, old.get("content", ""), old.get("type", "instruction"))
                except (ValueError, json.JSONDecodeError):
                    continue
            path.replace(backup)

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
        self._lock = _shared_lock(self.tasks_dir)

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

    @staticmethod
    def _scope(value: str) -> str:
        raw = value.replace("\\", "/").strip()
        if raw.startswith("/") or re.match(r"^[A-Za-z]:", raw) or ".." in raw.split("/"):
            raise ValueError("write_scope must contain safe workspace-relative paths")
        normalized = raw[2:] if raw.startswith("./") else raw
        if not normalized:
            raise ValueError("write_scope must contain safe workspace-relative paths")
        return normalized

    @staticmethod
    def _scope_root(value: str) -> str:
        parts = []
        for part in value.split("/"):
            if any(char in part for char in "*?["):
                break
            parts.append(part)
        return "/".join(parts).rstrip("/")

    @classmethod
    def scopes_overlap(cls, left: list[str], right: list[str]) -> bool:
        for first in left:
            for second in right:
                a, b = cls._scope_root(first), cls._scope_root(second)
                if not a or not b or a == b or a.startswith(b + "/") or b.startswith(a + "/"):
                    return True
        return False

    def create(self, subject: str, description: str = "", mode: str = "read",
               write_scope: list[str] | None = None, conversation_id: str | None = None) -> str:
        if not subject.strip():
            raise ValueError("subject is required")
        if mode not in {"read", "write"}:
            raise ValueError("mode must be read or write")
        scopes = [self._scope(item) for item in (write_scope or [])]
        if mode == "write" and not scopes:
            raise ValueError("write tasks require write_scope")
        with self._lock:
            task = {"id": self._next_id(), "subject": subject.strip(), "description": description,
                    "status": "pending", "owner": None, "blockedBy": [], "mode": mode,
                    "write_scope": scopes, "conversation_id": conversation_id or str(uuid.uuid4())}
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
                if status == "pending":
                    task["owner"] = None
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
            if task.get("mode", "read") == "write":
                for path in self.tasks_dir.glob("task_*.json"):
                    active = json.loads(path.read_text(encoding="utf-8"))
                    if active.get("status") == "in_progress" and active.get("mode") == "write" and self.scopes_overlap(
                        task.get("write_scope", []), active.get("write_scope", []),
                    ):
                        return f"Task #{task_id} scope conflict with task #{active['id']} owned by {active.get('owner')}"
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
