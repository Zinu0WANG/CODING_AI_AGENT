from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Callable

from .events import EventStore
from .config import AgentConfig
from .state import MessageBus, TaskManager, validate_name


class TeammateManager:
    """Thread-based teammates with atomic state and explicit final reporting."""

    def __init__(self, workspace: Path, events: EventStore, run_agent: Callable[..., object],
                 config: AgentConfig, idle_timeout: int = 60):
        self.workspace = workspace
        self.events = events
        self.run_agent = run_agent
        self.idle_timeout = idle_timeout
        self.team_dir = workspace / ".team"
        self.team_dir.mkdir(parents=True, exist_ok=True)
        self.config_path = self.team_dir / "config.json"
        self.sessions_dir = self.team_dir / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.bus = MessageBus(
            self.team_dir / "team.db", config.team_delivery_timeout_seconds,
            legacy_inbox_dir=self.team_dir / "inbox",
            event_callback=lambda kind, actor, payload: self.events.emit(kind, actor, payload),
        )
        self.tasks = TaskManager(workspace / ".tasks")
        self.session_recent_messages = config.team_session_recent_messages
        self.session_summary_tokens = config.team_session_summary_tokens
        self._lock = threading.RLock()
        self._threads: dict[str, threading.Thread] = {}
        self._reported_blocked: set[tuple[str, int]] = set()
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
                member["last_heartbeat"] = time.time()
                if error:
                    member["error"] = error
                self._save()

    def spawn(self, name: str, role: str, prompt: str, task_id: int | None = None,
              write_scope: list[str] | None = None) -> str:
        name = validate_name(name)
        with self._lock:
            member = self._member(name)
            if member and member["status"] not in {"idle", "shutdown", "failed"}:
                return f"Error: {name} is currently {member['status']}"
            if member:
                member.update({"role": role, "status": "working", "current_task": task_id,
                               "write_scope": write_scope or [], "last_heartbeat": time.time()})
            else:
                self.config["members"].append({"name": name, "role": role, "status": "working",
                                               "current_task": task_id, "write_scope": write_scope or [],
                                               "last_heartbeat": time.time()})
            self._save()
            thread = threading.Thread(target=self._loop, args=(name, role, prompt, task_id, write_scope or []), daemon=True, name=f"agent-{name}")
            self._threads[name] = thread
            thread.start()
        return f"Spawned {name} ({role})"

    def _session_path(self, name: str) -> Path:
        return self.sessions_dir / f"{validate_name(name)}.json"

    def _load_session(self, name: str) -> dict:
        path = self._session_path(name)
        if not path.exists():
            return {"summary": "", "recent_messages": []}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {"summary": "", "recent_messages": []}
        except (OSError, json.JSONDecodeError) as exc:
            self.events.emit("teammate_session_corrupt", name, {"reason": str(exc)})
            return {"summary": "", "recent_messages": []}

    def _save_session(self, name: str, role: str, prompt: str, result, task_id: int | None) -> None:
        previous = self._load_session(name)
        recent = [*previous.get("recent_messages", []), {"prompt": prompt, "answer": getattr(result, "answer", "")[:4000]}]
        data = {
            "name": name, "role": role, "current_task": task_id,
            "summary": getattr(result, "answer", "")[: self.session_summary_tokens * 4],
            "recent_messages": recent[-self.session_recent_messages:],
            "last_run_id": getattr(result, "run_id", None), "updated_at": time.time(),
        }
        path = self._session_path(name)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)

    def _work(self, name: str, role: str, prompt: str, task_id: int | None = None,
              write_scope: list[str] | None = None) -> object:
        session = self._load_session(name)
        continuity = ""
        if session.get("summary") or session.get("recent_messages"):
            continuity = f"\nPREVIOUS SESSION SUMMARY:\n{session.get('summary', '')}\nRECENT EXCHANGES:\n{json.dumps(session.get('recent_messages', []), ensure_ascii=False)}"
        result = self.run_agent(f"You are teammate {name}, role: {role}. {prompt}{continuity}", name, write_scope or [])
        status = getattr(result, "status", "failed")
        answer = getattr(result, "answer", "")
        self._save_session(name, role, prompt, result, task_id)
        message_type = "task_completed" if status == "completed" else "task_failed"
        self.bus.send(name, "lead", {"status": status, "answer": answer, "run_id": getattr(result, "run_id", None)},
                      message_type, task_id=task_id)
        return result

    def _loop(self, name: str, role: str, prompt: str, task_id: int | None, write_scope: list[str]) -> None:
        try:
            self._work(name, role, prompt, task_id, write_scope)
            self._status(name, "idle")
            deadline = time.monotonic() + self.idle_timeout
            while time.monotonic() < deadline:
                messages = self.bus.receive(name)
                if messages:
                    for message in messages:
                        if message.get("type") == "shutdown_request":
                            self._status(name, "shutdown")
                            self.bus.send(name, "lead", "Shutdown acknowledged", "shutdown_response")
                            self.bus.ack([message["message_id"]], name)
                            return
                        self._status(name, "working")
                        content = message.get("content", "")
                        prompt_text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
                        self._work(name, role, prompt_text, message.get("task_id"), write_scope)
                        self.bus.ack([message["message_id"]], name)
                        self._status(name, "idle")
                    deadline = time.monotonic() + self.idle_timeout
                    continue
                task = self.tasks.next_available()
                claim = self.tasks.claim(task["id"], name) if task else ""
                if task and claim.startswith("Claimed"):
                    with self._lock:
                        member = self._member(name)
                        if member:
                            member["current_task"] = task["id"]
                            member["write_scope"] = task.get("write_scope", [])
                            self._save()
                    self._status(name, "working")
                    self._work(name, role, f"Task #{task['id']}: {task['subject']}\n{task.get('description', '')}",
                               task["id"], task.get("write_scope", []))
                    self.tasks.update(task["id"], "completed")
                    with self._lock:
                        member = self._member(name)
                        if member:
                            member["current_task"] = None
                            member["write_scope"] = []
                            self._save()
                    self._status(name, "idle")
                    deadline = time.monotonic() + self.idle_timeout
                elif task and "scope conflict" in claim:
                    marker = (name, task["id"])
                    if marker not in self._reported_blocked:
                        self._reported_blocked.add(marker)
                        self.bus.send(name, "lead", {"reason": claim, "write_scope": task.get("write_scope", [])},
                                      "blocked", task_id=task["id"], conversation_id=task.get("conversation_id"))
                time.sleep(1)
            self._status(name, "shutdown")
        except Exception as exc:
            self._status(name, "failed", str(exc))
            self.events.emit("run_failed", name, {"reason": str(exc)})
            try:
                self.bus.send(name, "lead", {"error": str(exc)}, "task_failed")
            except Exception:
                pass

    def list_all(self) -> str:
        with self._lock:
            if not self.config["members"]:
                return "No teammates."
            pending = {name: len(self.bus.list_messages(name, status="pending", limit=100)) for name in self.names()}
            return "\n".join([f"Team: {self.config['team_name']}"] + [
                f"- {m['name']} ({m['role']}): {m['status']} task={m.get('current_task')} "
                f"scope={m.get('write_scope', [])} heartbeat={m.get('last_heartbeat')} "
                f"pending={pending.get(m['name'], 0)}"
                for m in self.config["members"]
            ])

    def names(self) -> list[str]:
        with self._lock:
            return [member["name"] for member in self.config["members"]]
