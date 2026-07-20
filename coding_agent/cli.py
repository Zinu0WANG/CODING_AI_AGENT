from __future__ import annotations

import json
import os
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from .config import AgentConfig
from .events import EventStore
from .plans import PlanStore
from .policy import PolicyDecision, RiskLevel
from .runtime import AgentRuntime, AnthropicModel, RunMode, RunResult
from .state import MessageBus, TaskManager


class AgentCLI:
    def __init__(self, workspace: Path | None = None, console: Console | None = None):
        load_dotenv(override=True)
        self.workspace = (workspace or Path.cwd()).resolve()
        self.console = console or Console()
        self.config = AgentConfig.load(self.workspace)
        self.plan_store = PlanStore(self.workspace, self.config.ignore_patterns, self.config.max_file_bytes)
        self.last_runtime: AgentRuntime | None = None
        self.model = self._create_model()

    def _create_model(self) -> AnthropicModel:
        model_name = os.getenv("MODEL_ID")
        if not model_name:
            raise RuntimeError("MODEL_ID is required; copy .env.example to .env")
        kwargs = {}
        api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY or DASHSCOPE_API_KEY is required")
        kwargs["api_key"] = api_key
        if os.getenv("ANTHROPIC_BASE_URL"):
            os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)
            kwargs["base_url"] = os.environ["ANTHROPIC_BASE_URL"]
        return AnthropicModel(Anthropic(**kwargs), model_name)

    def approve(self, name: str, arguments: dict, decision: PolicyDecision) -> bool:
        self.console.print(Panel(
            f"Tool: [bold]{name}[/bold]\nRisk: [yellow]{decision.risk.value}[/yellow]\n"
            f"Reason: {decision.reason}\nArguments: {json.dumps(arguments, ensure_ascii=False, indent=2)}\n"
            f"Workspace: {self.workspace}",
            title="Application-level approval (not an OS sandbox)", border_style="yellow",
        ))
        answer = self.console.input("Allow? [y] once / [a] all writes this run / [N] deny: ").strip().lower()
        if answer == "a" and self.last_runtime:
            self.last_runtime.tools.approve_for_run(RiskLevel.WRITE)
            return True
        return answer == "y"

    def run_prompt(self, prompt: str) -> RunResult:
        runtime = AgentRuntime(self.workspace, self.config, self.model, self.approve)
        self.last_runtime = runtime
        self.console.print(f"[dim]run_id={runtime.events.run_id}[/dim]")
        try:
            result = runtime.run(prompt)
        except KeyboardInterrupt:
            runtime.abort()
            raise
        self.render_result(result)
        return result

    def create_plan(self, request: str) -> dict | None:
        if not request.strip():
            self.console.print("[red]Usage: /plan REQUIREMENT[/red]")
            return None
        runtime = AgentRuntime(
            self.workspace, self.config, self.model, self.approve,
            interactive=False, enable_team=False, mode=RunMode.PLAN,
        )
        self.last_runtime = runtime
        self.console.print(f"[dim]planning_run_id={runtime.events.run_id}[/dim]")
        result = runtime.run(request)
        if result.status != "planned" or not result.answer.strip():
            self.console.print(Panel(result.answer or "Plan generation failed", title="PLAN FAILED", border_style="red"))
            return None
        selected_files = [
            event.get("payload", {}).get("path") for event in runtime.events.read_events()
            if event.get("type") == "context_selected" and event.get("payload", {}).get("path")
        ]
        try:
            plan = self.plan_store.create(request, result.answer, result.run_id, selected_files)
        except ValueError as exc:
            runtime.events.emit("run_failed", "lead", {"reason": str(exc), "mode": RunMode.PLAN.value})
            self.console.print(f"[red]Plan generation failed: {exc}[/red]")
            return None
        runtime.events.emit("plan_created", "lead", {
            "plan_id": plan["plan_id"], "selected_files": plan["selected_files"],
            "workspace_fingerprint": plan["workspace_fingerprint"], "git_head": plan["git_head"],
        })
        self.console.print(Panel(Markdown(plan["plan"]), title=f"PLAN {plan['plan_id']}", border_style="cyan"))
        self.console.print(f"[dim]Execute with /implement {plan['plan_id']}[/dim]")
        return plan

    def implement_plan(self, plan_id: str) -> RunResult | None:
        try:
            plan = self.plan_store.begin(plan_id)
        except ValueError as exc:
            self.console.print(f"[red]{exc}[/red]")
            return None
        planning_events = EventStore(self.workspace, plan["planning_run_id"])
        if plan["status"] == "stale":
            planning_events.emit("plan_stale", "lead", {
                "plan_id": plan_id, "reason": "workspace or Git HEAD changed",
            })
            self.console.print(f"[red]Plan {plan_id} is stale. Generate a new plan with /plan.[/red]")
            return None
        prompt = (
            "Implement the approved plan below. Recheck every assumption against the current repository, "
            "then modify files, run quality gates, and report truthfully.\n\n"
            f"ORIGINAL REQUEST:\n{plan['original_request']}\n\nAPPROVED PLAN:\n{plan['plan']}"
        )
        runtime = AgentRuntime(self.workspace, self.config, self.model, self.approve, mode=RunMode.ACT)
        self.last_runtime = runtime
        runtime.events.emit("plan_implementation_started", "lead", {
            "plan_id": plan_id, "planning_run_id": plan["planning_run_id"],
        })
        self.console.print(f"[dim]implementation_run_id={runtime.events.run_id} plan_id={plan_id}[/dim]")
        result = runtime.run(prompt)
        final_status = "completed" if result.status == "completed" else "failed"
        self.plan_store.finish(plan_id, final_status, result.run_id)
        event_type = "plan_implementation_completed" if final_status == "completed" else "plan_implementation_failed"
        runtime.events.emit(event_type, "lead", {
            "plan_id": plan_id, "planning_run_id": plan["planning_run_id"], "status": result.status,
        })
        self.render_result(result)
        return result

    def show_plans(self) -> None:
        table = Table("Plan ID", "Status", "Request", "Planning Run", "Implementation Run")
        for plan in self.plan_store.list_all():
            table.add_row(
                plan.get("plan_id", ""), plan.get("status", ""), plan.get("original_request", "")[:60],
                plan.get("planning_run_id", "")[:8], (plan.get("implementation_run_id") or "")[:8],
            )
        self.console.print(table)

    def show_plan(self, plan_id: str) -> None:
        try:
            plan = self.plan_store.load(plan_id)
        except ValueError as exc:
            self.console.print(f"[red]{exc}[/red]")
            return
        self.console.print(Panel(Markdown(plan["plan"]), title=f"PLAN {plan_id} · {plan['status']}", border_style="cyan"))

    def render_result(self, result: RunResult) -> None:
        color = "green" if result.status == "completed" else "red"
        self.console.print(Panel(result.answer or "(no final answer)", title=f"{result.status.upper()} · {result.run_id}", border_style=color))
        self.console.print(Panel(result.validation, title="Quality gates", border_style=color))
        self.console.print(Panel(Syntax(result.diff, "diff", theme="ansi_dark", word_wrap=True), title="Agent changes"))
        self.console.print(f"[dim]Duration: {result.duration_seconds:.2f}s · Inspect with /inspect {result.run_id}[/dim]")

    def show_runs(self) -> None:
        table = Table("Run ID", "Status", "Events", "Tools", "Tokens", "Duration")
        for run in EventStore.list_runs(self.workspace):
            tokens = run["input_tokens"] + run["output_tokens"]
            table.add_row(run["run_id"], run["status"], str(run["events"]), str(run["tool_calls"]),
                          str(tokens), f"{run['duration_seconds']:.2f}s")
        self.console.print(table)

    def inspect(self, run_id: str, replay: bool = False) -> None:
        if not run_id or not (self.workspace / ".runs" / run_id / "events.jsonl").exists():
            self.console.print(f"[red]Unknown run: {run_id}[/red]")
            return
        store = EventStore(self.workspace, run_id)
        events = store.read_events()
        title = "Read-only replay" if replay else "Run inspection"
        table = Table("Time", "Actor", "Event", "Details", title=f"{title}: {run_id}")
        started = events[0]["timestamp"] if events else 0
        for event in events:
            payload = json.dumps(event.get("payload", {}), ensure_ascii=False, default=str)
            table.add_row(f"+{event.get('timestamp', 0) - started:.2f}s", event.get("actor", ""), event.get("type", ""), payload[:160])
        self.console.print(table)

    def show_team(self) -> None:
        if self.last_runtime and self.last_runtime.team:
            team_summary = self.last_runtime.team.list_all()
        else:
            config_path = self.workspace / ".team" / "config.json"
            team_config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {"team_name": "default", "members": []}
            bus = MessageBus(self.workspace / ".team" / "team.db", self.config.team_delivery_timeout_seconds)
            lines = [f"Team: {team_config.get('team_name', 'default')}"]
            for member in team_config.get("members", []):
                pending = len(bus.list_messages(member.get("name"), status="pending", limit=100))
                lines.append(f"- {member.get('name')} ({member.get('role')}): {member.get('status')} "
                             f"task={member.get('current_task')} scope={member.get('write_scope', [])} "
                             f"heartbeat={member.get('last_heartbeat')} pending={pending}")
            team_summary = "\n".join(lines)
        self.console.print(Panel(team_summary, title="Team"))
        self.console.print(Panel(TaskManager(self.workspace / ".tasks").list_all(), title="Tasks and write scopes"))

    def show_messages(self, status: str | None = None) -> None:
        bus = MessageBus(self.workspace / ".team" / "team.db", self.config.team_delivery_timeout_seconds)
        try:
            messages = bus.list_messages(status=status or None, limit=100)
        except ValueError as exc:
            self.console.print(f"[red]{exc}[/red]")
            return
        table = Table("ID", "Status", "Type", "From", "To", "Task", "Attempts", "Content", title="Team messages")
        for message in messages:
            content = json.dumps(message["content"], ensure_ascii=False, default=str)
            table.add_row(message["message_id"][:8], message["status"], message["type"], message["sender"],
                          message["recipient"], str(message.get("task_id") or ""),
                          str(message["delivery_attempts"]), content[:100])
        self.console.print(table)

    def retry_message(self, message_id: str) -> None:
        if not message_id:
            self.console.print("[red]Usage: /retry-message MESSAGE_ID[/red]")
            return
        bus = MessageBus(self.workspace / ".team" / "team.db", self.config.team_delivery_timeout_seconds)
        self.console.print("[green]Message queued for redelivery.[/green]" if bus.retry(message_id)
                           else "[red]Message not found or already acknowledged.[/red]")

    def handle_command(self, command: str) -> bool:
        parts = command.strip().split(maxsplit=1)
        name = parts[0].lower()
        argument = parts[1] if len(parts) > 1 else ""
        if name in {"/quit", "/exit", "/q"}:
            return False
        if name == "/plan":
            self.create_plan(argument)
        elif name == "/plans":
            self.show_plans()
        elif name == "/show-plan":
            self.show_plan(argument)
        elif name == "/implement":
            self.implement_plan(argument)
        elif name == "/runs":
            self.show_runs()
        elif name == "/team":
            self.show_team()
        elif name == "/messages":
            self.show_messages(argument or None)
        elif name == "/retry-message":
            self.retry_message(argument)
        elif name in {"/inspect", "/replay"}:
            self.inspect(argument, replay=name == "/replay")
        elif name == "/diff":
            self.console.print(Syntax(self.last_runtime.tools.diff() if self.last_runtime else "No run yet.", "diff"))
        elif name == "/test":
            if not self.last_runtime:
                self.last_runtime = AgentRuntime(self.workspace, self.config, self.model, self.approve)
            passed, summary = self.last_runtime.tools.run_quality_gates()
            self.console.print(Panel(summary, border_style="green" if passed else "red"))
        elif name == "/abort":
            if self.last_runtime:
                self.last_runtime.abort()
            self.console.print("[yellow]Current run marked aborted.[/yellow]")
        elif name == "/help":
            self.console.print("/plan REQUIREMENT · /plans · /show-plan ID · /implement ID · "
                               "/runs · /inspect ID · /replay ID · /team · /messages [STATUS] · "
                               "/retry-message ID · /diff · /test · /abort · /exit")
        else:
            self.console.print(f"[red]Unknown command: {name}[/red]")
        return True

    def repl(self) -> None:
        self.console.print(Panel(
            "Observable coding-agent CLI\nApplication-level approvals · trajectories · repo map · quality gates",
            title="Agent Harness", border_style="cyan",
        ))
        while True:
            try:
                query = self.console.input("[bold cyan]agent > [/bold cyan]").strip()
                if not query:
                    continue
                if query.startswith("/"):
                    if not self.handle_command(query):
                        break
                else:
                    self.run_prompt(query)
            except KeyboardInterrupt:
                if self.last_runtime:
                    self.last_runtime.abort()
                self.console.print("\n[yellow]Run aborted; trajectory preserved.[/yellow]")
            except EOFError:
                break


def main() -> None:
    AgentCLI().repl()
