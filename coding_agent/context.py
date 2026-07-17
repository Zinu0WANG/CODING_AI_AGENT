from __future__ import annotations

import ast
import fnmatch
import json
import subprocess
from collections import Counter
from pathlib import Path


LANGUAGES = {
    ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript", ".tsx": "TypeScript",
    ".rs": "Rust", ".go": "Go", ".java": "Java", ".md": "Markdown", ".yml": "YAML",
    ".yaml": "YAML", ".json": "JSON", ".toml": "TOML",
}
KEY_CONFIGS = {"pyproject.toml", "requirements.txt", "package.json", "Cargo.toml", "go.mod", ".agent.yml"}


class RepoMap:
    def __init__(self, workspace: Path, ignore_patterns: list[str] | None = None, max_file_bytes: int = 250_000):
        self.workspace = workspace.resolve()
        self.ignore_patterns = ignore_patterns or []
        self.max_file_bytes = max_file_bytes
        self.cache_path = self.workspace / ".runs" / "repo-map-cache.json"

    def _ignored(self, relative: str) -> bool:
        normalized = relative.replace("\\", "/")
        return any(fnmatch.fnmatch(normalized, pattern) or fnmatch.fnmatch(normalized + "/", pattern) for pattern in self.ignore_patterns)

    def _python_symbols(self, path: Path) -> list[str]:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, SyntaxError):
            return []
        return [node.name for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]

    def build(self) -> dict:
        cached = {}
        if self.cache_path.exists():
            try:
                cached = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                cached = {}
        files, languages, entries = [], Counter(), {}
        for path in sorted(self.workspace.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(self.workspace).as_posix()
            if self._ignored(relative) or path.stat().st_size > self.max_file_bytes:
                continue
            stat = path.stat()
            fingerprint = f"{stat.st_size}:{stat.st_mtime_ns}"
            entry = cached.get("entries", {}).get(relative)
            if not entry or entry.get("fingerprint") != fingerprint:
                entry = {"fingerprint": fingerprint, "symbols": self._python_symbols(path) if path.suffix == ".py" else []}
            entries[relative] = entry
            files.append(relative)
            if path.suffix in LANGUAGES:
                languages[LANGUAGES[path.suffix]] += 1
        git_status = "unavailable"
        try:
            result = subprocess.run(["git", "status", "--short"], cwd=self.workspace, capture_output=True, text=True, timeout=5)
            git_status = result.stdout.strip() or "clean"
        except (OSError, subprocess.TimeoutExpired):
            pass
        data = {"files": files, "languages": dict(languages), "entries": entries, "git_status": git_status}
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return data

    def render(self) -> str:
        data = self.build()
        lines = ["Repository map", f"Languages: {data['languages']}", f"Git: {data['git_status']}", "Files:"]
        for relative in data["files"]:
            symbols = data["entries"][relative].get("symbols", [])
            suffix = f"  symbols: {', '.join(symbols)}" if symbols else ""
            marker = " [config]" if Path(relative).name in KEY_CONFIGS else ""
            lines.append(f"- {relative}{marker}{suffix}")
        return "\n".join(lines)
