"""Durable, self-contained archives for the new preparation and diagnostic runs."""

import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
import time
import traceback
import uuid
import zipfile
from datetime import datetime, timezone


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with tmp.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def command_output(args, cwd=None):
    try:
        result = subprocess.run(args, cwd=cwd, text=True, capture_output=True, timeout=15)
        return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"unavailable": str(error)}


class Tee:
    def __init__(self, original, path):
        self.original = original
        self.file = Path(path).open("x", encoding="utf-8")

    def write(self, text):
        self.original.write(text)
        self.file.write(text)
        self.file.flush()
        return len(text)

    def flush(self):
        self.original.flush()
        self.file.flush()

    def close(self):
        self.file.flush()
        os.fsync(self.file.fileno())
        self.file.close()


class RunArchive:
    """Log before numerical work. Does not yet implement a training/resume engine."""

    required = {"schema_version", "purpose", "dataset", "geometry", "randomness", "computation"}

    def __init__(self, config, root, source_root, mirror_root=None, parent_run_id=None):
        missing = self.required.difference(config)
        if missing:
            raise ValueError(f"Missing required manifest fields: {sorted(missing)}")
        if config["schema_version"] != 1:
            raise ValueError("Unsupported manifest schema version")
        # Round-trip also rejects unserializable or non-finite configuration values.
        self.config = json.loads(json.dumps(config, allow_nan=False))
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:12]
        self.path = Path(root).resolve() / self.run_id
        self.source_root = Path(source_root).resolve()
        self.mirror = Path(mirror_root).resolve() / self.run_id if mirror_root else None
        self.parent_run_id = parent_run_id
        self.start = time.perf_counter()
        self._stack = contextlib.ExitStack()

    def event(self, kind, **fields):
        record = {"time_utc": utc_now(), "run_id": self.run_id, "event": kind, **fields}
        encoded = json.dumps(record, sort_keys=True, allow_nan=False)
        with (self.path / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(encoded + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _sources(self):
        sources = []
        excluded = {".git", "runs", ".venv", "venv", "__pycache__", ".pytest_cache", "local", "artifacts"}
        extensions = {".py", ".json", ".toml", ".yml", ".yaml", ".txt", ".lock"}
        with zipfile.ZipFile(self.path / "sources.zip", "x", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(self.source_root.rglob("*")):
                rel = path.relative_to(self.source_root)
                if excluded.intersection(rel.parts) or not path.is_file() or path.is_symlink():
                    continue
                if path.suffix not in extensions and path.name != ".gitignore":
                    continue
                content = path.read_bytes()
                archive.writestr(str(rel), content)
                sources.append({"path": str(rel), "sha256": hashlib.sha256(content).hexdigest()})
        write_json(self.path / "sources.json", sources)
        return sources

    def _mirror_metadata(self):
        if self.mirror is None:
            return
        self.mirror.mkdir(parents=True, exist_ok=True)
        for path in sorted(self.path.rglob("*")):
            if not path.is_file() or path.suffix not in {".json", ".jsonl", ".log", ".zip"}:
                continue
            # Nested test archives remain in the canonical parent; avoid duplicating them here.
            if "fixtures" in path.relative_to(self.path).parts:
                continue
            target = self.mirror / path.relative_to(self.path)
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".tmp")
            shutil.copyfile(path, tmp)
            if digest(tmp) != digest(path):
                raise OSError(f"Mirror hash mismatch: {path}")
            os.replace(tmp, target)

    def __enter__(self):
        self.path.mkdir(parents=True, exist_ok=False)
        try:
            sources = self._sources()
            packages = sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions()
                              if d.metadata["Name"])
            environment = {
                "python": sys.version, "executable": sys.executable,
                "platform": platform.platform(), "hostname": socket.gethostname(),
                "cpu_count": os.cpu_count(), "packages": packages,
                "nvidia_smi": command_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,driver_version", "--format=csv"]),
                "selected_environment": {k: os.environ[k] for k in (
                    "SLURM_JOB_ID", "SLURM_JOB_PARTITION", "CUDA_VISIBLE_DEVICES",
                    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "LD_LIBRARY_PATH", "LD_PRELOAD", "CUDA_MODULE_LOADING",
                    "CUBLAS_WORKSPACE_CONFIG", "CUSPARSE_LOG_LEVEL") if k in os.environ},
            }
            write_json(self.path / "environment.json", environment)
            git = {"head": command_output(["git", "rev-parse", "HEAD"], self.source_root),
                   "status": command_output(["git", "status", "--porcelain=v1"], self.source_root),
                   "diff": command_output(["git", "diff", "HEAD", "--"], self.source_root)}
            write_json(self.path / "git.json", git)
            manifest = {
                "schema_version": 1, "run_id": self.run_id, "parent_run_id": self.parent_run_id,
                "created_utc": utc_now(), "command": sys.argv, "cwd": str(Path.cwd()),
                "canonical_path": str(self.path), "metadata_mirror": str(self.mirror) if self.mirror else None,
                "configuration": self.config, "source_files": sources,
                "sources_zip_sha256": digest(self.path / "sources.zip"),
                "environment_sha256": digest(self.path / "environment.json"),
                "study_id": self.config.get("study_id"),
                "training": self.config.get("training", {"applicable": False, "reason": (
                    "Fixed untrained checkpoint diagnostics; no optimizer or training."
                    if self.config.get("model_evaluation") else
                    "Preparation/diagnostics only; no model or optimizer.")}),
                "model_evaluation": self.config.get("model_evaluation", {"applicable": False}),
                "replay": "Use configuration with this sources.zip and recorded Python/packages; create a new run.",
            }
            write_json(self.path / "manifest.json", manifest)
            self.event("started")
            self._mirror_metadata()  # Fail before numerical work if required mirror cannot be written.
            self.stdout = Tee(sys.stdout, self.path / "stdout.log")
            self.stderr = Tee(sys.stderr, self.path / "stderr.log")
            self._stack.enter_context(contextlib.redirect_stdout(self.stdout))
            self._stack.enter_context(contextlib.redirect_stderr(self.stderr))
            return self
        except BaseException as error:
            write_json(self.path / "completion.json", {"status": "setup_failed", "error": repr(error), "time_utc": utc_now()})
            raise

    def __exit__(self, exc_type, exc, tb):
        try:
            status = "completed" if exc_type is None else "failed"
            if exc_type:
                traceback.print_exception(exc_type, exc, tb)
            self.event("finished", status=status, elapsed_seconds=time.perf_counter() - self.start)
        finally:
            self._stack.close()
            self.stdout.close()
            self.stderr.close()
        write_json(self.path / "completion.json", {
            "status": status, "error": repr(exc) if exc else None, "time_utc": utc_now(),
            "elapsed_seconds": time.perf_counter() - self.start,
        })
        artifacts = [{"path": str(p.relative_to(self.path)), "bytes": p.stat().st_size, "sha256": digest(p)}
                     for p in sorted(self.path.rglob("*")) if p.is_file() and p.name != "artifacts.json"]
        write_json(self.path / "artifacts.json", {"canonical_path": str(self.path), "files": artifacts})
        self._mirror_metadata()
        return False
