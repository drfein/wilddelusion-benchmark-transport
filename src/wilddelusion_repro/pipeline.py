"""Small declarative pipeline runner with resumable, hashed stage records."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ALLOWED_KINDS = {"free", "network", "api", "gpu"}
FINGERPRINT_ROOTS = {
    "configs": {".toml"},
    "experiments": {".py"},
    "frozen": {".json", ".jsonl"},
    "provenance": {".toml"},
    "scripts": {".py"},
    "src": {".py"},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Stage:
    name: str
    description: str
    command: str
    deps: tuple[str, ...]
    outputs: tuple[str, ...]
    expected_sha256: tuple[tuple[str, str], ...]
    required_env: tuple[str, ...]
    kind: str = "free"
    cwd: str = "."


class Pipeline:
    def __init__(self, root: Path, config_path: Path, stages: dict[str, Stage]):
        self.root = root.resolve()
        self.config_path = config_path.resolve()
        self.stages = stages
        self.state_dir = self.root / ".repro" / "state"
        self._repository_input_hash: str | None = None
        self._validate()

    @classmethod
    def load(cls, root: Path, config_path: Path) -> Pipeline:
        raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
        stages = {}
        for name, item in raw.get("stages", {}).items():
            stages[name] = Stage(
                name=name,
                description=str(item.get("description", "")),
                command=str(item["command"]).strip(),
                deps=tuple(item.get("deps", [])),
                outputs=tuple(item.get("outputs", [])),
                expected_sha256=tuple(
                    sorted(item.get("expected_sha256", {}).items())
                ),
                required_env=tuple(item.get("required_env", [])),
                kind=str(item.get("kind", "free")),
                cwd=str(item.get("cwd", ".")),
            )
        return cls(root, config_path, stages)

    def _validate(self) -> None:
        for stage in self.stages.values():
            if stage.kind not in ALLOWED_KINDS:
                raise ValueError(f"{stage.name}: invalid kind {stage.kind!r}")
            missing = sorted(set(stage.deps) - self.stages.keys())
            if missing:
                raise ValueError(f"{stage.name}: unknown dependencies {missing}")
            unknown_outputs = sorted(
                set(dict(stage.expected_sha256)) - set(stage.outputs)
            )
            if unknown_outputs:
                raise ValueError(
                    f"{stage.name}: expected hashes for undeclared outputs "
                    f"{unknown_outputs}"
                )
            for path, digest in stage.expected_sha256:
                if len(digest) != 64 or any(
                    character not in "0123456789abcdef" for character in digest
                ):
                    raise ValueError(f"{stage.name}: invalid SHA-256 for {path}")
        self.order(self.stages)

    def order(self, targets: Iterable[str]) -> list[str]:
        requested = list(targets)
        unknown = sorted(set(requested) - self.stages.keys())
        if unknown:
            raise ValueError(f"unknown stages: {', '.join(unknown)}")
        visiting: set[str] = set()
        visited: set[str] = set()
        ordered: list[str] = []

        def visit(name: str) -> None:
            if name in visited:
                return
            if name in visiting:
                raise ValueError(f"pipeline cycle at {name}")
            visiting.add(name)
            for dependency in self.stages[name].deps:
                visit(dependency)
            visiting.remove(name)
            visited.add(name)
            ordered.append(name)

        for name in requested:
            visit(name)
        return ordered

    def render_command(self, stage: Stage) -> str:
        values = {
            "repo": shlex.quote(str(self.root)),
            "python": shlex.quote(sys.executable),
            "experiment": shlex.quote(
                str(
                    self.root
                    / "experiments"
                    / "lost_in_delusion_replication_260600975"
                )
            ),
        }
        return stage.command.format_map(values)

    def _state_path(self, stage: Stage) -> Path:
        return self.state_dir / f"{stage.name}.json"

    def repository_input_hash(self) -> str:
        """Bind resumable state to every published implementation/input file."""
        if self._repository_input_hash is not None:
            return self._repository_input_hash
        records = []
        for root_name, suffixes in FINGERPRINT_ROOTS.items():
            root = self.root / root_name
            if not root.exists():
                continue
            for path in sorted(root.rglob("*")):
                relative = path.relative_to(self.root)
                if (
                    not path.is_file()
                    or path.suffix not in suffixes
                    or "__pycache__" in relative.parts
                    or "artifacts" in relative.parts
                    or "results" in relative.parts
                    or "stochastic_results" in relative.parts
                    or "local_results" in relative.parts
                ):
                    continue
                records.append((relative.as_posix(), sha256_file(path)))
        payload = json.dumps(records, separators=(",", ":")).encode()
        self._repository_input_hash = hashlib.sha256(payload).hexdigest()
        return self._repository_input_hash

    def _output_record(self, relative: str) -> dict[str, Any]:
        path = self.root / relative
        record: dict[str, Any] = {"path": relative, "exists": path.exists()}
        if path.is_file():
            record.update({"bytes": path.stat().st_size, "sha256": sha256_file(path)})
        elif path.is_dir():
            record["kind"] = "directory"
        return record

    def is_complete(self, stage: Stage) -> bool:
        state_path = self._state_path(stage)
        if not state_path.exists() or not stage.outputs:
            return False
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("repository_input_sha256") != self.repository_input_hash():
            return False
        if state.get("command_sha256") != hashlib.sha256(
            self.render_command(stage).encode()
        ).hexdigest():
            return False
        dependency_states = state.get("dependency_states")
        if dependency_states is None or set(dependency_states) != set(stage.deps):
            return False
        for dependency, expected_hash in dependency_states.items():
            path = self._state_path(self.stages[dependency])
            if not path.exists() or sha256_file(path) != expected_hash:
                return False
        for expected in state.get("outputs", []):
            current = self._output_record(expected["path"])
            if not current["exists"]:
                return False
            if expected.get("sha256") != current.get("sha256"):
                return False
        for relative, digest in stage.expected_sha256:
            if self._output_record(relative).get("sha256") != digest:
                return False
        return True

    def run(
        self,
        targets: Iterable[str],
        *,
        dry_run: bool,
        resume: bool,
        allow_api: bool,
        allow_gpu: bool,
    ) -> None:
        ordered = self.order(targets)
        for name in ordered:
            stage = self.stages[name]
            command = self.render_command(stage)
            if resume and self.is_complete(stage):
                print(f"SKIP  {name} (verified state)")
                continue
            if stage.kind == "api" and not allow_api:
                raise RuntimeError(
                    f"{name} can incur API charges; rerun with --allow-api"
                )
            if stage.kind == "gpu" and not allow_gpu:
                raise RuntimeError(
                    f"{name} requires GPU compute; rerun with --allow-gpu"
                )
            missing_env = [key for key in stage.required_env if not os.getenv(key)]
            if missing_env and not dry_run:
                raise RuntimeError(f"{name} requires: {', '.join(missing_env)}")
            print(f"RUN   {name} [{stage.kind}]\n      {stage.description}")
            if dry_run:
                print(f"      {command.replace(chr(10), chr(10) + '      ')}")
                continue
            started = datetime.now(UTC)
            environment = os.environ.copy()
            environment.update(
                {
                    "REPO_ROOT": str(self.root),
                    "EXP_ROOT": str(
                        self.root
                        / "experiments"
                        / "lost_in_delusion_replication_260600975"
                    ),
                    "PYTHONUNBUFFERED": "1",
                }
            )
            subprocess.run(
                ["bash", "-euo", "pipefail", "-c", command],
                cwd=self.root / stage.cwd,
                env=environment,
                check=True,
            )
            missing_outputs = [
                output
                for output in stage.outputs
                if not (self.root / output).exists()
            ]
            if missing_outputs:
                raise RuntimeError(f"{name} did not create: {missing_outputs}")
            wrong_hashes = {
                relative: {
                    "expected": digest,
                    "actual": self._output_record(relative).get("sha256"),
                }
                for relative, digest in stage.expected_sha256
                if self._output_record(relative).get("sha256") != digest
            }
            if wrong_hashes:
                raise RuntimeError(
                    f"{name} deterministic output mismatch: "
                    f"{json.dumps(wrong_hashes, sort_keys=True)}"
                )
            self.state_dir.mkdir(parents=True, exist_ok=True)
            state = {
                "stage": name,
                "description": stage.description,
                "kind": stage.kind,
                "started_at": started.isoformat(),
                "finished_at": datetime.now(UTC).isoformat(),
                "command": command,
                "command_sha256": hashlib.sha256(command.encode()).hexdigest(),
                "repository_input_sha256": self.repository_input_hash(),
                "dependency_states": {
                    dependency: sha256_file(
                        self._state_path(self.stages[dependency])
                    )
                    for dependency in stage.deps
                },
                "outputs": [self._output_record(path) for path in stage.outputs],
            }
            self._state_path(stage).write_text(
                json.dumps(state, indent=2) + "\n", encoding="utf-8"
            )
