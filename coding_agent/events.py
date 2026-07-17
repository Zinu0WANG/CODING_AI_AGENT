from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class AgentEvent:
    event_id: str
    run_id: str
    timestamp: float
    type: str
    actor: str
    payload: dict[str, Any]


class EventStore:
    def __init__(self, workspace: Path, run_id: str | None = None):
        self.root = workspace / ".runs"
        self.run_id = run_id or str(uuid.uuid4())
        self.run_dir = self.root / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_dir / "events.jsonl"
        self._lock = threading.Lock()

    def emit(self, event_type: str, actor: str = "lead", payload: dict | None = None) -> AgentEvent:
        event = AgentEvent(str(uuid.uuid4()), self.run_id, time.time(), event_type, actor, payload or {})
        line = json.dumps(asdict(event), ensure_ascii=False, default=str)
        with self._lock:
            with self.events_path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
                stream.flush()
        return event

    def read_events(self) -> list[dict]:
        if not self.events_path.exists():
            return []
        events = []
        for number, line in enumerate(self.events_path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as exc:
                events.append({
                    "event_id": f"corrupt-{number}", "run_id": self.run_id,
                    "timestamp": 0, "type": "corrupt_event", "actor": "runtime",
                    "payload": {"line": number, "error": str(exc)},
                })
        return events

    @classmethod
    def list_runs(cls, workspace: Path) -> list[dict]:
        root = workspace / ".runs"
        runs = []
        if not root.exists():
            return runs
        for directory in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            path = directory / "events.jsonl"
            if not directory.is_dir() or not path.exists():
                continue
            store = cls.__new__(cls)
            store.root, store.run_id, store.run_dir, store.events_path = root, directory.name, directory, path
            store._lock = threading.Lock()
            events = store.read_events()
            terminal = next((e for e in reversed(events) if e["type"] in {"run_completed", "run_failed"}), None)
            runs.append({"run_id": directory.name, "events": len(events), "status": terminal["type"] if terminal else "running"})
        return runs
