"""Build the independent upstream V16 result set using staged promotion."""
from __future__ import annotations

import json
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from tespub.codesign_v16 import run_all
from tespub.archive_v16 import archive_completed_run
from tespub.evidence import build_evidence_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-dir", help="Timestamped in-progress record to complete after promotion")
    parser.add_argument("--no-archive", action="store_true", help="Promote only; reserved for controlled recovery")
    args = parser.parse_args()
    stage = ROOT / ".results_v16_staging"
    final = ROOT / "results_v16"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    evidence = build_evidence_outputs(ROOT / "data", stage)
    analysis = run_all(
        ROOT / "config" / "study_v16.json",
        ROOT / "config" / "reference_control_volume_v16.json",
        stage,
    )
    release = {"evidence": evidence, "analysis": analysis, "status": "PASS", "version": "16.0.0"}
    (stage / "run_summary_v16.json").write_text(json.dumps(release, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    backup = ROOT / ".results_v16_previous"
    if backup.exists():
        shutil.rmtree(backup)
    if final.exists():
        final.replace(backup)
    stage.replace(final)
    if backup.exists():
        shutil.rmtree(backup)
    archive = None
    if not args.no_archive:
        archive = archive_completed_run(
            ROOT,
            results_dir=final,
            record_dir=args.record_dir,
            checkpoint_dir=ROOT / ".v16_restart_checkpoints",
            log_paths=(ROOT / "V16_durable_rebuild.log", ROOT / "V16_durable_rebuild.err.log"),
        )
    print(json.dumps({
        "status": "PASS", "version": "16.0.0", "results": str(final),
        "run_record": str(archive) if archive else None,
        "n_unit_cases": analysis["n_unit_cases"],
        "n_unit_admissible": analysis["n_unit_admissible"],
        "n_system_pareto": analysis["n_system_pareto"],
    }, indent=2))


if __name__ == "__main__":
    main()
