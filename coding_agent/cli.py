from __future__ import annotations

import json
import os
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from .config import AgentConfig
from .events import EventStore
from .policy import PolicyDecision, RiskLevel
from .runtime import AgentRuntime, AnthropicModel, RunResult


class AgentCLI:
    def __init__(self, workspace: Path | None = None, console: Console | None = None):
        load_dotenv(override=True)
        self.workspace = (workspace or Path.cwd()).resolve()
        self.console = console or Console()
        self.config = AgentConfig.load(self.workspace)
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

    def handle_command(self, command: str) -> bool:
        parts = command.strip().split(maxsplit=1)
        name = parts[0].lower()
        argument = parts[1] if len(parts) > 1 else ""
        if name in {"/quit", "/exit", "/q"}:
            return False
        if name == "/runs":
            self.show_runs()
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
            self.console.print("/runs · /inspect ID · /replay ID · /diff · /test · /abort · /exit")
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
