from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Callable

from .events import EventStore
from .state import MessageBus, TaskManager, validate_name


class TeammateManager:
    """Thread-based teammates with atomic state and explicit final reporting."""

    def __init__(self, workspace: Path, events: EventStore, run_agent: Callable[[str, str], object], idle_timeout: int = 60):
        self.workspace = workspace
        self.events = events
        self.run_agent = run_agent
        self.idle_timeout = idle_timeout
        self.team_dir = workspace / ".team"
        self.team_dir.mkdir(parents=True, exist_ok=True)
        self.config_path = self.team_dir / "config.json"
        self.bus = MessageBus(self.team_dir / "inbox")
        self.tasks = TaskManager(workspace / ".tasks")
        self._lock = threading.RLock()
        self._threads: dict[str, threading.Thread] = {}
        self.config = self._load()

    def _load(self) -> dict:
        if not self.config_path.exists():
            return {"team_name": "default", "members": []}
        try:
            return json.loads(self.config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"team_name": "default", "members": []}

    def _save(self) -> None:
        temporary = self.config_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.config, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(self.config_path)

    def _member(self, name: str) -> dict | None:
        return next((member for member in self.config["members"] if member["name"] == name), None)

    def _status(self, name: str, status: str, error: str | None = None) -> None:
        with self._lock:
            member = self._member(name)
            if member:
                member["status"] = status
                if error:
                    member["error"] = error
                self._save()

    def spawn(self, name: str, role: str, prompt: str) -> str:
        name = validate_name(name)
        with self._lock:
            member = self._member(name)
            if member and member["status"] not in {"idle", "shutdown", "failed"}:
                return f"Error: {name} is currently {member['status']}"
            if member:
                member.update({"role": role, "status": "working"})
            else:
                self.config["members"].append({"name": name, "role": role, "status": "working"})
            self._save()
            thread = threading.Thread(target=self._loop, args=(name, role, prompt), daemon=True, name=f"agent-{name}")
            self._threads[name] = thread
            thread.start()
        return f"Spawned {name} ({role})"

    def _work(self, name: str, role: str, prompt: str) -> None:
        result = self.run_agent(f"You are teammate {name}, role: {role}. {prompt}", name)
        status = getattr(result, "status", "failed")
        answer = getattr(result, "answer", "")
        self.bus.send(name, "lead", f"[{status}] {answer}")
        self.events.emit("message_sent", name, {"to": "lead", "status": status, "content": answer[:1000]})

    def _loop(self, name: str, role: str, prompt: str) -> None:
        try:
            self._work(name, role, prompt)
            self._status(name, "idle")
            deadline = time.monotonic() + self.idle_timeout
            while time.monotonic() < deadline:
                messages = self.bus.read_inbox(name)
                if messages:
                    for message in messages:
                        if message.get("type") == "shutdown_request":
                            self._status(name, "shutdown")
                            self.bus.send(name, "lead", "Shutdown acknowledged", "shutdown_response")
                            return
                        self._status(name, "working")
                        self._work(name, role, message.get("content", ""))
                        self._status(name, "idle")
                    deadline = time.monotonic() + self.idle_timeout
                    continue
                task = self.tasks.next_available()
                if task and self.tasks.claim(task["id"], name).startswith("Claimed"):
                    self._status(name, "working")
                    self._work(name, role, f"Task #{task['id']}: {task['subject']}\n{task.get('description', '')}")
                    self.tasks.update(task["id"], "completed")
                    self._status(name, "idle")
                    deadline = time.monotonic() + self.idle_timeout
                time.sleep(1)
            self._status(name, "shutdown")
        except Exception as exc:
            self._status(name, "failed", str(exc))
            self.events.emit("run_failed", name, {"reason": str(exc)})
            try:
                self.bus.send(name, "lead", f"[failed] {exc}")
            except Exception:
                pass

    def list_all(self) -> str:
        with self._lock:
            if not self.config["members"]:
                return "No teammates."
            return "\n".join([f"Team: {self.config['team_name']}"] + [f"- {m['name']} ({m['role']}): {m['status']}" for m in self.config["members"]])

    def names(self) -> list[str]:
        with self._lock:
            return [member["name"] for member in self.config["members"]]
