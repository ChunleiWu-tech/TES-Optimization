"""Immutable archival bundles for completed V16 upstream runs.

The live ``results_v16`` directory is a convenient working location, but it is
not a stable provenance boundary because a later run may replace it.  This
module creates an independent, checksum-indexed run record only after a V16
result set has been promoted successfully.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterable


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_file_if_identical_or_new(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.is_file() and _sha256(source) == _sha256(destination):
            return
        raise FileExistsError(f"Archive destination already exists with different content: {destination}")
    shutil.copy2(source, destination)


def _copy_tree_if_identical_or_new(source: Path, destination: Path) -> None:
    for item in sorted(source.rglob("*")):
        relative = item.relative_to(source)
        target = destination / relative
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif item.is_file() and "__pycache__" not in item.parts and item.suffix != ".pyc":
            _copy_file_if_identical_or_new(item, target)


def _file_index(root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for item in sorted(root.rglob("*")):
        if not item.is_file() or item.name == "ARCHIVE_MANIFEST.json":
            continue
        records.append({
            "path": item.relative_to(root).as_posix(),
            "bytes": item.stat().st_size,
            "sha256": _sha256(item),
        })
    return records


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _validated_summary(results_dir: Path, not_before: datetime | None) -> tuple[Path, dict[str, object]]:
    summary_path = results_dir / "analysis_summary_v16.json"
    release_path = results_dir / "run_summary_v16.json"
    if not summary_path.is_file() or not release_path.is_file():
        raise FileNotFoundError("A promoted V16 result directory must contain analysis_summary_v16.json and run_summary_v16.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    release = json.loads(release_path.read_text(encoding="utf-8"))
    if summary.get("status") != "PASS" or release.get("status") != "PASS":
        raise RuntimeError("Only PASS V16 result sets may be archived")
    if not_before is not None:
        promoted_at = datetime.fromtimestamp(summary_path.stat().st_mtime, tz=not_before.tzinfo)
        if promoted_at < not_before:
            raise RuntimeError("Refusing to archive a result set promoted before this run started")
    return summary_path, summary


def archive_completed_run(
    package_root: str | Path,
    *,
    results_dir: str | Path | None = None,
    record_dir: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    log_paths: Iterable[str | Path] = (),
    not_before: datetime | None = None,
) -> Path:
    """Create or complete an immutable run record from a promoted PASS result set.

    Existing files are never overwritten.  For an in-progress record, identical
    checkpoint copies are accepted and only missing final deliverables are added.
    """
    root = Path(package_root).resolve()
    results = Path(results_dir).resolve() if results_dir else root / "results_v16"
    summary_path, summary = _validated_summary(results, not_before)

    if record_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        destination = root / "run_records" / f"V16_{stamp}_COMPLETE"
        if destination.exists():
            raise FileExistsError(f"Run record already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}_", dir=destination.parent))
        finalize_by_rename = True
    else:
        destination = Path(record_dir).resolve()
        destination.mkdir(parents=True, exist_ok=True)
        if (destination / "final_results").exists():
            raise FileExistsError(f"Run record already has final results: {destination}")
        staging = Path(tempfile.mkdtemp(prefix=".finalization_", dir=destination))
        finalize_by_rename = False

    try:
        _copy_tree_if_identical_or_new(results, staging / "final_results")
        snapshot = staging / "reproducibility_snapshot"
        for folder in ("config", "data", "tespub", "tests"):
            _copy_tree_if_identical_or_new(root / folder, snapshot / folder)
        for filename in ("run_v16.py", "pyproject.toml", "requirements.lock"):
            _copy_file_if_identical_or_new(root / filename, snapshot / filename)
        scripts = root / "scripts"
        for filename in (
            "run_v16_durable.ps1",
            "archive_v16_run.py",
            "run_v16_tests.py",
            "audit_v16_results.py",
            "audit_release_delta_v16.py",
        ):
            candidate = scripts / filename
            if candidate.is_file():
                _copy_file_if_identical_or_new(candidate, snapshot / "scripts" / filename)
        if checkpoint_dir is not None:
            checkpoints = Path(checkpoint_dir)
            if checkpoints.is_dir():
                _copy_tree_if_identical_or_new(checkpoints, staging / "validated_checkpoints")
        for log in log_paths:
            candidate = Path(log)
            if candidate.is_file():
                _copy_file_if_identical_or_new(candidate, staging / "finalization_logs" / candidate.name)

        if finalize_by_rename:
            manifest_root = staging
        else:
            for child in sorted(staging.iterdir(), key=lambda path: path.name):
                target = destination / child.name
                if target.exists():
                    if child.is_dir():
                        _copy_tree_if_identical_or_new(child, target)
                        shutil.rmtree(child)
                    else:
                        _copy_file_if_identical_or_new(child, target)
                        child.unlink()
                else:
                    child.replace(target)
            staging.rmdir()
            manifest_root = destination

        manifest = {
            "archive_schema": "V16_RUN_RECORD_V1",
            "archive_status": "PASS",
            "record_id": destination.name,
            "archived_at_local": datetime.now().astimezone().isoformat(timespec="seconds"),
            "source_results_directory": str(results),
            "analysis_summary_sha256": _sha256(summary_path),
            "study_id": summary.get("study_id"),
            "version": summary.get("version"),
            "result_status": summary.get("status"),
            "files": _file_index(manifest_root),
        }
        _atomic_json(manifest, manifest_root / "ARCHIVE_MANIFEST.json")
        if finalize_by_rename:
            staging.replace(destination)
        return destination
    except Exception:
        # The staging directory is retained for forensic recovery; no completed
        # record or live upstream file is overwritten on an archive failure.
        raise
