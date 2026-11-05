"""Package a completed, hash-verified reproduction for reviewers.

This creates a local ZIP, not a public release. It refuses failed/incomplete
runs, changed sources, stale artifacts, and changed logs. The archive includes
the exact inputs, generated paper/results, and provenance of the passing run.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


HERE = Path(__file__).resolve().parent
PREFIX = "neurosymbolic-reproduction"
REQUIRED_SOURCES = {
    "run_all.py", "package_reproduction.py", "rounding.py", "bounds.py",
    "cegis_c.py", "experiments.py", "residual.py", "bench_pendulum.py",
    "bench_neural.py", "smoke_p1.py", "fill_tex.py", "README.md",
    "requirements.txt", "requirements-lock.txt",
    "paper/nsp_contraction_paper_ieee.tex",
    "revision_experiments.py", "revision_extensions.py", "revision_report.py",
    "RESPONSE_TO_CLAUDE.md", "release/build_arxiv.py",
}
REQUIRED_ARTIFACTS = {
    "results/traces.json", "results/gridcost.json", "results/residual.json",
    "results/pendulum.json", "results/neural.json", "results/smoke_p1.json",
    "results/rounding_checks.json", "paper/numbers.tex",
    "paper/nsp_contraction_paper_ieee.pdf",
    "results/revision_experiments.json", "results/revision_extensions.json",
    "results/revision_report.json", "results/timing_revision.json",
    "results/arxiv_bundle.json", "paper/revision_numbers.tex", "paper/revision_tables.tex",
    *("paper/fig_" + name + ".pdf" for name in
      ("ablation", "scale_chase", "extension3d", "surrogate")),
    *("paper/fig_" + name + ".pdf" for name in
      ("traces", "heatmaps", "gridcost", "residual", "pendulum", "neural")),
}


def digest(content):
    return hashlib.sha256(content).hexdigest()


def checked_bytes(root, relative, expected=None):
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or "\\" in relative:
        raise ValueError(f"unsafe snapshot path: {relative}")
    resolved = root / relative
    if not resolved.resolve().is_relative_to(root.resolve()) or resolved.is_symlink():
        raise ValueError(f"snapshot path escapes the project or is a symlink: {relative}")
    content = resolved.read_bytes()
    if expected is not None and digest(content) != expected:
        raise ValueError(f"content changed since the passing run: {relative}")
    return content


def snapshot_files(root):
    root = Path(root)
    manifest_path = "results/run_manifest.json"
    manifest_bytes = checked_bytes(root, manifest_path)
    manifest = json.loads(manifest_bytes)
    stages = manifest.get("stages", [])
    expected = manifest.get("expected_stage_count", 8)
    if (manifest.get("status") != "pass" or not isinstance(expected, int) or expected < 8
            or len(stages) != expected
            or any(stage.get("status") != "pass" for stage in stages)):
        raise ValueError("a complete passing reproduction with every declared stage is required")
    sources = manifest.get("sources_at_finish", {})
    artifacts = manifest.get("artifact_sha256", {})
    if sources != manifest.get("sources_at_start"):
        raise ValueError("sources changed during the pipeline; rerun before packaging")
    if not REQUIRED_SOURCES.issubset(sources) or not REQUIRED_ARTIFACTS.issubset(artifacts):
        raise ValueError("manifest omits required reproduction sources or artifacts")
    recorded = dict(sources)
    recorded.update(artifacts)
    recorded[manifest["run_log"]] = manifest["run_log_sha256"]
    for stage in stages:
        recorded[stage["log"]] = stage["log_sha256"]
    files = {name: checked_bytes(root, name, expected)
             for name, expected in sorted(recorded.items())}
    files[manifest_path] = manifest_bytes
    snapshot = {
        "schema_version": 1,
        "description": "Complete local reviewer archive; not a public release",
        "pipeline_finished_at": manifest["finished_at"],
        "run_manifest_sha256": digest(manifest_bytes),
        "files": {name: digest(content) for name, content in sorted(files.items())},
    }
    files["SNAPSHOT.json"] = (json.dumps(snapshot, indent=2, sort_keys=True) + "\n").encode()
    return files


def package(root=HERE, destination=None):
    root = Path(root)
    destination = (Path(destination) if destination is not None
                   else root / "dist" / (PREFIX + ".zip"))
    files = snapshot_files(root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as archive:
        for name, content in sorted(files.items()):
            info = zipfile.ZipInfo(PREFIX + "/" + name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, content, compresslevel=9)
    temporary.replace(destination)
    checksum = digest(destination.read_bytes())
    destination.with_suffix(".zip.sha256").write_text(checksum + "  " + destination.name + "\n")
    return destination, checksum, len(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    path, checksum, count = package(destination=args.output)
    print(f"Packaged {count} files: {path}\nSHA256 {checksum}")


if __name__ == "__main__":
    main()
