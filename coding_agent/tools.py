from __future__ import annotations

import difflib
import json
import subprocess
import threading
import uuid
from pathlib import Path
from typing import Callable

from .config import AgentConfig
from .context import RepoMap
from .events import EventStore
from .policy import PolicyDecision, RiskLevel, ToolPolicy
from .state import TaskManager


ApprovalCallback = Callable[[str, dict, PolicyDecision], bool]


class ToolRegistry:
    def __init__(self, workspace: Path, config: AgentConfig, events: EventStore,
                 approval_callback: ApprovalCallback | None = None, actor: str = "lead"):
        self.workspace = workspace.resolve()
        self.config = config
        self.events = events
        self.policy = ToolPolicy(self.workspace)
        self.approval_callback = approval_callback
        self.actor = actor
        self.tasks = TaskManager(self.workspace / ".tasks")
        self.repo_map = RepoMap(self.workspace, config.ignore_patterns, config.max_file_bytes)
        self._approved_for_run: set[RiskLevel] = set()
        self._before: dict[Path, bytes | None] = self._snapshot_workspace()
        self._background: dict[str, dict] = {}
        self._background_lock = threading.Lock()
        self.aborted = False

    def _snapshot_workspace(self) -> dict[Path, bytes | None]:
        snapshot = {}
        for relative in self.repo_map.build()["files"]:
            path = self.workspace / relative
            try:
                snapshot[path] = path.read_bytes()
            except OSError:
                continue
        return snapshot

    @property
    def schemas(self) -> list[dict]:
        return [
            self._schema("bash", "Run a shell command under the application approval policy.", {"command": {"type": "string"}}, ["command"]),
            self._schema("read_file", "Read a workspace file and record why it was selected.", {"path": {"type": "string"}, "reason": {"type": "string"}, "limit": {"type": "integer"}}, ["path", "reason"]),
            self._schema("write_file", "Write a workspace file.", {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
            self._schema("edit_file", "Replace one exact occurrence in a workspace file.", {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, ["path", "old_text", "new_text"]),
            self._schema("repo_map", "Refresh and show the repository map.", {}, []),
            self._schema("background_run", "Run an approved command in a background thread.", {"command": {"type": "string"}, "timeout": {"type": "integer"}}, ["command"]),
            self._schema("check_background", "Check one or all background commands.", {"task_id": {"type": "string"}}, []),
            self._schema("task_create", "Create a persistent task.", {"subject": {"type": "string"}, "description": {"type": "string"}}, ["subject"]),
            self._schema("task_list", "List persistent tasks.", {}, []),
            self._schema("task_update", "Update a persistent task.", {"task_id": {"type": "integer"}, "status": {"type": "string", "enum": ["pending", "in_progress", "completed", "deleted"]}}, ["task_id"]),
            self._schema("load_skill", "Load a local SKILL.md by name.", {"name": {"type": "string"}}, ["name"]),
        ]

    @staticmethod
    def _schema(name: str, description: str, properties: dict, required: list[str]) -> dict:
        return {"name": name, "description": description,
                "input_schema": {"type": "object", "properties": properties, "required": required}}

    def _decision(self, name: str, arguments: dict) -> PolicyDecision:
        if name in {"bash", "background_run"}:
            return self.policy.classify_command(arguments["command"])
        if name in {"write_file", "edit_file"}:
            return self.policy.classify_path(arguments["path"], write=True)
        if name == "read_file":
            return self.policy.classify_path(arguments["path"], write=False)
        if name in {"task_create", "task_update"}:
            return PolicyDecision(RiskLevel.WRITE, "updates workspace task state")
        return PolicyDecision(RiskLevel.READ, "read-only agent operation")

    def _allowed(self, name: str, arguments: dict, decision: PolicyDecision) -> bool:
        if decision.risk is RiskLevel.DANGEROUS:
            return False
        if decision.risk is RiskLevel.READ:
            return True
        if self.config.approval_policy == "read_only":
            return False
        if self.config.approval_policy == "allow_write" or decision.risk in self._approved_for_run:
            return True
        self.events.emit("approval_requested", self.actor, {"tool": name, "arguments": arguments, "risk": decision.risk.value, "reason": decision.reason})
        approved = bool(self.approval_callback and self.approval_callback(name, arguments, decision))
        self.events.emit("approval_resolved", self.actor, {"tool": name, "approved": approved})
        return approved

    def approve_for_run(self, risk: RiskLevel = RiskLevel.WRITE) -> None:
        self._approved_for_run.add(risk)

    def authorize(self, name: str, arguments: dict, decision: PolicyDecision) -> bool:
        """Apply the same approval flow to runtime-managed tools such as delegation."""
        return self._allowed(name, arguments, decision)

    def execute(self, name: str, arguments: dict) -> str:
        self.events.emit("tool_requested", self.actor, {"tool": name, "arguments": arguments})
        if self.aborted:
            output = "Error: run aborted"
            self.events.emit("tool_finished", self.actor, {"tool": name, "ok": False, "output": output})
            return output
        decision = self._decision(name, arguments)
        if not self._allowed(name, arguments, decision):
            output = f"Error: {decision.risk.value} operation denied: {decision.reason}"
            self.events.emit("tool_finished", self.actor, {"tool": name, "ok": False, "output": output, "risk": decision.risk.value})
            return output
        self.events.emit("tool_started", self.actor, {"tool": name, "risk": decision.risk.value})
        try:
            output = self._dispatch(name, arguments)
            ok = not output.startswith("Error:")
        except Exception as exc:
            output, ok = f"Error: {exc}", False
        self.events.emit("tool_finished", self.actor, {"tool": name, "ok": ok, "output": output[:2000]})
        return output[:50_000]

    def _remember(self, path: Path) -> None:
        if path not in self._before:
            self._before[path] = path.read_bytes() if path.exists() else None

    def _dispatch(self, name: str, arguments: dict) -> str:
        if name == "bash":
            result = subprocess.run(arguments["command"], shell=True, cwd=self.workspace, capture_output=True,
                                    text=True, timeout=self.config.command_timeout)
            output = (result.stdout + result.stderr).strip() or "(no output)"
            return f"exit_code={result.returncode}\n{output}"
        if name == "read_file":
            path = self.policy.resolve_path(arguments["path"])
            text = path.read_text(encoding="utf-8")
            lines = text.splitlines()
            limit = arguments.get("limit")
            if limit and len(lines) > limit:
                text = "\n".join(lines[:limit] + [f"... ({len(lines) - limit} more lines)"])
            self.events.emit("context_selected", self.actor, {"path": arguments["path"], "reason": arguments["reason"]})
            return text
        if name == "write_file":
            path = self.policy.resolve_path(arguments["path"])
            self._remember(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(arguments["content"], encoding="utf-8")
            return f"Wrote {len(arguments['content'])} characters to {arguments['path']}"
        if name == "edit_file":
            path = self.policy.resolve_path(arguments["path"])
            self._remember(path)
            content = path.read_text(encoding="utf-8")
            if content.count(arguments["old_text"]) != 1:
                return f"Error: old_text must occur exactly once; found {content.count(arguments['old_text'])}"
            path.write_text(content.replace(arguments["old_text"], arguments["new_text"], 1), encoding="utf-8")
            return f"Edited {arguments['path']}"
        if name == "repo_map":
            return self.repo_map.render()
        if name == "background_run":
            task_id = str(uuid.uuid4())[:8]
            with self._background_lock:
                self._background[task_id] = {"status": "running", "command": arguments["command"], "result": None}
            thread = threading.Thread(target=self._background_exec,
                                      args=(task_id, arguments["command"], arguments.get("timeout", self.config.command_timeout)),
                                      daemon=True, name=f"background-{task_id}")
            thread.start()
            return f"Background task {task_id} started"
        if name == "check_background":
            task_id = arguments.get("task_id")
            with self._background_lock:
                if task_id:
                    task = self._background.get(task_id)
                    return json.dumps(task, ensure_ascii=False) if task else f"Error: unknown background task {task_id}"
                return json.dumps(self._background, ensure_ascii=False)
        if name == "task_create":
            return self.tasks.create(arguments["subject"], arguments.get("description", ""))
        if name == "task_list":
            return self.tasks.list_all()
        if name == "task_update":
            return self.tasks.update(arguments["task_id"], arguments.get("status"))
        if name == "load_skill":
            skill_name = arguments["name"]
            if not re_safe_name(skill_name):
                return "Error: invalid skill name"
            matches = list((self.workspace / "skills").glob(f"**/{skill_name}/SKILL.md"))
            if not matches:
                matches = [p for p in (self.workspace / "skills").glob("**/SKILL.md") if p.parent.name == skill_name]
            return matches[0].read_text(encoding="utf-8") if matches else f"Error: unknown skill {skill_name}"
        return f"Error: unknown tool {name}"

    def _background_exec(self, task_id: str, command: str, timeout: int) -> None:
        try:
            result = subprocess.run(command, shell=True, cwd=self.workspace, capture_output=True, text=True, timeout=timeout)
            output = (result.stdout + result.stderr).strip() or "(no output)"
            state = {"status": "completed" if result.returncode == 0 else "failed",
                     "command": command, "result": f"exit_code={result.returncode}\n{output}"[:50_000]}
        except Exception as exc:
            state = {"status": "failed", "command": command, "result": str(exc)}
        with self._background_lock:
            self._background[task_id] = state
        self.events.emit("tool_finished", self.actor, {"tool": "background_run", "task_id": task_id,
                         "ok": state["status"] == "completed", "output": state["result"][:2000]})

    def diff(self) -> str:
        for relative in self.repo_map.build()["files"]:
            path = self.workspace / relative
            self._before.setdefault(path, None)
        chunks = []
        for path, before in sorted(self._before.items(), key=lambda item: str(item[0])):
            after = path.read_bytes() if path.exists() else None
            old_lines = (before or b"").decode("utf-8", errors="replace").splitlines(keepends=True)
            new_lines = (after or b"").decode("utf-8", errors="replace").splitlines(keepends=True)
            relative = path.relative_to(self.workspace).as_posix()
            chunks.extend(difflib.unified_diff(old_lines, new_lines, fromfile=f"a/{relative}", tofile=f"b/{relative}"))
        return "".join(chunks) or "No agent changes."

    def run_quality_gates(self) -> tuple[bool, str]:
        results, passed = [], True
        for kind, commands in (("lint", self.config.lint_commands), ("test", self.config.test_commands)):
            for command in commands:
                try:
                    result = subprocess.run(command, shell=True, cwd=self.workspace, capture_output=True, text=True,
                                            timeout=self.config.command_timeout)
                    output = (result.stdout + result.stderr).strip()
                    results.append(f"[{kind}] {command}\nexit_code={result.returncode}\n{output}")
                    passed = passed and result.returncode == 0
                except subprocess.TimeoutExpired:
                    results.append(f"[{kind}] {command}\nError: timeout")
                    passed = False
        summary = "\n\n".join(results) if results else "No quality gates configured; structural run completed."
        self.events.emit("validation_finished", self.actor, {"passed": passed, "summary": summary[:5000]})
        return passed, summary


def re_safe_name(value: str) -> bool:
    return bool(value and value.replace("-", "").replace("_", "").isalnum() and len(value) <= 64)
