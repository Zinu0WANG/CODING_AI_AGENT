from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from .events import EventStore


CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
ARTIFACT_ID_PATTERN = re.compile(r"^[a-f0-9]{12}$")


def estimate_tokens(text: str) -> int:
    """Conservatively estimate tokens without a provider-specific tokenizer."""
    cjk = len(CJK_PATTERN.findall(text))
    non_cjk = len(text) - cjk
    return cjk + math.ceil(non_cjk / 4)


@dataclass(frozen=True, slots=True)
class ArtifactMetadata:
    artifact_id: str
    tool: str
    status: str
    created_at: float
    chars: int
    estimated_tokens: int
    sha256: str
    filename: str


class ArtifactStore:
    def __init__(self, run_dir: Path, events: EventStore):
        self.run_dir = run_dir.resolve()
        self.artifacts_dir = self.run_dir / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.artifacts_dir / "index.jsonl"
        self.events = events
        self._lock = threading.RLock()

    def artifact_path(self, artifact_id: str) -> Path:
        if not ARTIFACT_ID_PATTERN.fullmatch(artifact_id or ""):
            raise ValueError("unknown artifact")
        return self.artifacts_dir / f"{artifact_id}.txt"

    def _metadata(self) -> dict[str, ArtifactMetadata]:
        metadata = {}
        if not self.index_path.exists():
            return metadata
        for line in self.index_path.read_text(encoding="utf-8").splitlines():
            try:
                item = ArtifactMetadata(**json.loads(line))
            except (json.JSONDecodeError, TypeError):
                continue
            metadata[item.artifact_id] = item
        return metadata

    def list_metadata(self) -> list[ArtifactMetadata]:
        with self._lock:
            return list(self._metadata().values())

    def create(self, tool: str, content: str, status: str, estimated: int | None = None) -> ArtifactMetadata:
        encoded = content.encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        with self._lock:
            artifact_id = uuid.uuid4().hex[:12]
            while self.artifact_path(artifact_id).exists():
                artifact_id = uuid.uuid4().hex[:12]
            path = self.artifact_path(artifact_id)
            temporary = path.with_suffix(f".tmp-{uuid.uuid4().hex}")
            temporary.write_bytes(encoded)
            os.replace(temporary, path)
            metadata = ArtifactMetadata(
                artifact_id=artifact_id, tool=tool, status=status, created_at=time.time(),
                chars=len(content), estimated_tokens=estimated if estimated is not None else estimate_tokens(content),
                sha256=digest, filename=path.name,
            )
            with self.index_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(asdict(metadata), ensure_ascii=False) + "\n")
                stream.flush()
            self.events.emit("artifact_created", "runtime", asdict(metadata))
            return metadata

    def _verified_content(self, artifact_id: str, metadata: ArtifactMetadata | None = None) -> tuple[ArtifactMetadata, str]:
        metadata = metadata or self._metadata().get(artifact_id)
        if not metadata:
            raise ValueError("unknown artifact")
        path = self.artifact_path(artifact_id)
        if not path.exists():
            raise ValueError("artifact file missing")
        encoded = path.read_bytes()
        if hashlib.sha256(encoded).hexdigest() != metadata.sha256:
            raise ValueError("artifact integrity check failed")
        try:
            return metadata, encoded.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("artifact is not valid UTF-8") from exc

    def read(self, artifact_id: str, offset: int = 0, limit: int = 8000) -> dict:
        if offset < 0:
            raise ValueError("offset must be non-negative")
        if not 1 <= limit <= 12000:
            raise ValueError("limit must be between 1 and 12000")
        with self._lock:
            metadata, content = self._verified_content(artifact_id)
        page = content[offset:offset + limit]
        next_offset = offset + len(page) if offset + len(page) < len(content) else None
        result = {"artifact_id": artifact_id, "tool": metadata.tool, "offset": offset,
                  "total_chars": len(content), "next_offset": next_offset, "content": page}
        self.events.emit("artifact_accessed", "runtime", {"artifact_id": artifact_id, "operation": "read", "offset": offset, "chars": len(page)})
        return result

    def search(self, query: str, max_hits: int = 5) -> list[dict]:
        query = query.strip()
        if not query:
            raise ValueError("query is required")
        if len(query) > 200:
            raise ValueError("query must be at most 200 characters")
        if not 1 <= max_hits <= 20:
            raise ValueError("max_hits must be between 1 and 20")
        hits = []
        with self._lock:
            for metadata in self._metadata().values():
                try:
                    _, content = self._verified_content(metadata.artifact_id, metadata)
                except ValueError:
                    continue
                index = content.casefold().find(query.casefold())
                if index < 0:
                    continue
                start, end = max(0, index - 80), min(len(content), index + len(query) + 80)
                hits.append({"artifact_id": metadata.artifact_id, "tool": metadata.tool,
                             "chars": metadata.chars, "snippet": content[start:end]})
                if len(hits) >= max_hits:
                    break
        self.events.emit("artifact_accessed", "runtime", {"operation": "search", "query": query, "hits": len(hits)})
        return hits


@dataclass(slots=True)
class ToolResultReference:
    tool: str
    result: dict
    externalized: bool = False


class ContextManager:
    def __init__(self, store: ArtifactStore, events: EventStore, keep_batches: int = 3,
                 threshold_tokens: int = 1000):
        self.store = store
        self.events = events
        self.keep_batches = keep_batches
        self.threshold_tokens = threshold_tokens
        self.batches: list[list[ToolResultReference]] = []

    def register_batch(self, results: list[tuple[str, dict]]) -> None:
        self.batches.append([ToolResultReference(tool, result) for tool, result in results])

    def compact(self) -> int:
        eligible = self.batches[:-self.keep_batches] if self.keep_batches else self.batches
        count = 0
        for batch in eligible:
            for reference in batch:
                if reference.externalized:
                    continue
                content = reference.result.get("content", "")
                if not isinstance(content, str):
                    content = json.dumps(content, ensure_ascii=False, default=str)
                tokens = estimate_tokens(content)
                if tokens <= self.threshold_tokens:
                    continue
                failed_exit = content.startswith("exit_code=") and not content.startswith("exit_code=0\n")
                status = "error" if content.startswith("Error:") or failed_exit else "success"
                metadata = self.store.create(reference.tool, content, status, tokens)
                reference.result["content"] = (
                    f"[artifact: {metadata.artifact_id}]\n"
                    f"tool={reference.tool} status={status}\n"
                    f"full result externalized: estimated_tokens={tokens} chars={len(content)}\n"
                    "Use artifact_read or artifact_search to retrieve it."
                )
                reference.externalized = True
                count += 1
                self.events.emit("context_externalized", "runtime", {
                    "artifact_id": metadata.artifact_id, "tool": reference.tool,
                    "estimated_tokens": tokens, "chars": len(content),
                })
        return count
