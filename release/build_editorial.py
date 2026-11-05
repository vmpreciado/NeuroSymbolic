"""Rebuild the final paper while preserving the verified scientific snapshot.

The parent ZIP is published unchanged. Only the manuscript and publication
documentation may differ from its inputs; scientific source, data, figures,
and original run evidence must still match exactly. This is an editorial
build, not a new eight-stage experiment run.
"""
import contextlib
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_all

PARENT_SHA256 = "c67b4247793c591125dfd71c1d86776a21b95ba1a59d116126c51d29cd934070"
RELEASE_URL = "https://github.com/vmpreciado/NeuroSymbolic/releases/tag/v1.0.0"
EDITORIAL_INPUTS = {"paper/nsp_contraction_paper_ieee.tex", "README.md",
                    "RESPONSE_TO_REFEREE.md"}
REBUILT_OUTPUTS = {"paper/nsp_contraction_paper_ieee.pdf", "paper/numbers.tex"}


def sha(content):
    return hashlib.sha256(content).hexdigest()


def verify_parent():
    archive = ROOT / "dist/neurosymbolic-reproduction.zip"
    if sha(archive.read_bytes()) != PARENT_SHA256:
        raise ValueError("verified parent archive checksum changed")
    prefix = "neurosymbolic-reproduction/"
    with zipfile.ZipFile(archive) as z:
        snapshot = json.loads(z.read(prefix + "SNAPSHOT.json"))
        for name, expected in snapshot["files"].items():
            if sha(z.read(prefix + name)) != expected:
                raise ValueError(f"invalid parent member: {name}")
        manifest = json.loads(z.read(prefix + "results/run_manifest.json"))
    if (manifest.get("status") != "pass" or len(manifest["stages"]) != 8
            or any(stage.get("status") != "pass" for stage in manifest["stages"])
            or manifest["sources_at_start"] != manifest["sources_at_finish"]):
        raise ValueError("parent is not a stable passing full reproduction")
    return snapshot, manifest


def check_unchanged(snapshot):
    unchanged = {}
    for name, expected in snapshot["files"].items():
        if name in EDITORIAL_INPUTS | REBUILT_OUTPUTS:
            continue
        if sha((ROOT / name).read_bytes()) != expected:
            raise ValueError(f"scientific input/output or run evidence changed: {name}")
        unchanged[name] = expected
    return unchanged


def main():
    snapshot, original_run = verify_parent()
    unchanged = check_unchanged(snapshot)
    started = datetime.now(timezone.utc).isoformat()
    log_path = ROOT / "results/editorial_build.log"
    with log_path.open("w", buffering=1) as log:
        with contextlib.redirect_stdout(run_all.Tee(sys.stdout, log)), \
                contextlib.redirect_stderr(run_all.Tee(sys.stderr, log)):
            print("Editorial-only build; verified scientific evidence is reused unchanged.")
            print("Release citation: " + RELEASE_URL)
            run_all.validate_traces(run_all.read_json("traces.json"))
            run_all.validate_residual(run_all.read_json("residual.json"))
            run_all.validate_benchmark(run_all.read_json("pendulum.json"))
            run_all.validate_benchmark(run_all.read_json("neural.json"), neural=True)
            run_all.validate_smoke(run_all.read_json("smoke_p1.json"))
            print("Saved-result validation PASS")
            run_all.run_script("fill_tex.py", outputs=("paper/numbers.tex",))
            detail = run_all.latex()
            print("PASS: " + detail)
    check_unchanged(snapshot)
    changed = {}
    for name in sorted(EDITORIAL_INPUTS):
        changed[name] = {"parent_sha256": snapshot["files"][name],
                         "final_sha256": sha((ROOT / name).read_bytes())}
    extra = {name: sha((ROOT / name).read_bytes()) for name in
             ("CITATION.cff", "release/build_editorial.py")}
    report = {
        "schema_version": 1, "kind": "editorial-only rebuild", "status": "pass",
        "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(),
        "release_tag": "v1.0.0", "release_url": RELEASE_URL,
        "parent_snapshot": {"path": "dist/neurosymbolic-reproduction.zip",
                            "sha256": PARENT_SHA256,
                            "run_manifest_sha256": snapshot["run_manifest_sha256"],
                            "full_run_status": original_run["status"]},
        "scientific_experiments_rerun": False,
        "description": "Scientific inputs, outputs and full-run evidence reused unchanged; final wording and release citation rebuilt separately.",
        "editorial_input_changes": changed, "additional_publication_sources": extra,
        "unchanged_parent_files": unchanged,
        "output_sha256": {name: sha((ROOT / name).read_bytes())
                          for name in sorted(REBUILT_OUTPUTS | {"results/editorial_build.log"})},
        "checks": {"parent_zip_and_members": "pass", "scientific_hashes": "pass",
                   "saved_result_validation": "pass", "fill_tex": "pass",
                   "two_latex_passes": detail},
    }
    output = ROOT / "results/editorial_rebuild.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print("Wrote " + str(output))


if __name__ == "__main__":
    main()
