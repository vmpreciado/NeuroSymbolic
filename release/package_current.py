"""Package verified scientific evidence with a separately verified editorial overlay.

This does not rerun experiments or rewrite their manifest.  The original passing
snapshot is preserved byte-for-byte; the current manuscript and release metadata
are identified by a second, explicit inventory.  Only declared files are read.
The same standard-library script is included as the standalone archive verifier.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import tarfile
import tempfile
import zipfile


VERSION = "v1.1.1"
REPORT = "results/editorial_v1.1.1.json"
ORIGINAL = "provenance/original-scientific-snapshot.zip"
ROOT = (Path(__file__).resolve().parent if Path(__file__).name == "VERIFY_CURRENT_REPRODUCTION.py"
        else Path(__file__).resolve().parents[1])
# These are the only original run members an editorial/release overlay may change.
# In particular, numerical programs, generated numbers, figures, results, and the
# passing run's own logs and manifest must remain byte-for-byte unchanged.
EDITORIAL_MEMBERS = {
    "README.md", "CITATION.cff", "CLAUDE.md", "RESPONSE_TO_CLAUDE.md",
    "RESPONSE_TO_REFEREE.md", "paper/nsp_contraction_paper_ieee.tex",
    "paper/nsp_contraction_paper_ieee.pdf", "results/arxiv_bundle.json",
    "release/build_arxiv.py", "release/build_editorial.py", "release/test_arxiv_bundle.py",
}
OLD_BUNDLE = "dist/neurosymbolic-arxiv-claude-revision.tar.gz"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def safe_name(name):
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError(f"unsafe member path: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in ("", ".", "..") for part in name.split("/")):
        raise ValueError(f"unsafe member path: {name!r}")
    return name


def checked_bytes(root, name, expected=None):
    safe_name(name)
    root = Path(root).resolve()
    path = root / name
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError(f"symlink or escaping member: {name}")
    content = path.read_bytes()
    if expected is not None and digest(content) != expected:
        raise ValueError(f"hash mismatch: {name}")
    return content


def original_snapshot(content, expected_hash=None, expected_manifest_hash=None):
    """Validate the complete archived run without trusting the current worktree."""
    if expected_hash is not None and digest(content) != expected_hash:
        raise ValueError("original scientific snapshot hash mismatch")
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("duplicate member in original snapshot")
        for info in archive.infolist():
            safe_name(info.filename)
            if info.is_dir() or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError("nonregular original snapshot member")
        inventory = [name for name in names if name.endswith("/SNAPSHOT.json")]
        if len(inventory) != 1:
            raise ValueError("original snapshot must have exactly one inventory")
        prefix = inventory[0][:-len("SNAPSHOT.json")]
        snapshot = json.loads(archive.read(inventory[0]))
        hashes = snapshot["files"]
        for name in hashes:
            safe_name(name)
        if set(names) != {prefix + name for name in hashes} | {inventory[0]}:
            raise ValueError("original snapshot inventory is incomplete")
        files = {name: archive.read(prefix + name) for name in hashes}
        for name, expected in hashes.items():
            if digest(files[name]) != expected:
                raise ValueError(f"original snapshot member changed: {name}")
    manifest_bytes = files["results/run_manifest.json"]
    manifest_hash = digest(manifest_bytes)
    if (snapshot["run_manifest_sha256"] != manifest_hash
            or expected_manifest_hash is not None and expected_manifest_hash != manifest_hash):
        raise ValueError("original run manifest hash mismatch")
    manifest = json.loads(manifest_bytes)
    stages = manifest.get("stages", [])
    if (manifest.get("status") != "pass" or manifest.get("expected_stage_count") != 12
            or len(stages) != 12 or any(stage.get("status") != "pass" for stage in stages)
            or manifest.get("sources_at_start") != manifest.get("sources_at_finish")):
        raise ValueError("original snapshot needs a complete stable passing twelve-stage run")
    recorded = dict(manifest["sources_at_finish"])
    recorded.update(manifest["artifact_sha256"])
    recorded[manifest["run_log"]] = manifest["run_log_sha256"]
    for stage in stages:
        recorded[stage["log"]] = stage["log_sha256"]
    if not recorded or any(hashes.get(name) != expected for name, expected in recorded.items()):
        raise ValueError("original manifest does not match its archived evidence")
    return snapshot, manifest, files


def declared_hashes(report):
    """Every overlay member is explicitly declared by the editorial verifier."""
    declared = {}
    groups = [report.get("unchanged_parent_files", {}), report.get("output_sha256", {}),
              {name: item["final_sha256"]
               for name, item in report.get("editorial_input_changes", {}).items()}]
    for group in groups:
        for name, expected in group.items():
            safe_name(name)
            if not re.fullmatch(r"[0-9a-f]{64}", expected):
                raise ValueError(f"invalid SHA256: {name}")
            if name in declared and declared[name] != expected:
                raise ValueError(f"conflicting current hashes: {name}")
            declared[name] = expected
    return declared


def check_bundle(bundle_bytes, bundle_report, file_bytes):
    if (bundle_report.get("status") not in ("pass", "prepared")
            or bundle_report.get("clean_build", {}).get("status") != "pass"
            or bundle_report["sha256"] != digest(bundle_bytes)):
        raise ValueError("current arXiv bundle lacks a passing clean build")
    expected = bundle_report["member_sha256"]
    seen = set()
    with tarfile.open(fileobj=io.BytesIO(bundle_bytes), mode="r:gz") as archive:
        for member in archive:
            safe_name(member.name)
            if (not member.isfile() or "/" in member.name or member.name in seen
                    or member.name not in expected):
                raise ValueError("unsafe or unexpected arXiv source member")
            content = archive.extractfile(member).read()
            if (digest(content) != expected[member.name]
                    or content != file_bytes("paper/" + member.name)):
                raise ValueError(f"arXiv/current source mismatch: {member.name}")
            seen.add(member.name)
    if seen != set(expected) or len(seen) != bundle_report["member_count"]:
        raise ValueError("arXiv source inventory is incomplete")
    return len(seen)


def collect_files(root, report_name=REPORT, version=VERSION):
    root = Path(root)
    report_bytes = checked_bytes(root, report_name)
    report = json.loads(report_bytes)
    if report.get("status") != "pass" or report.get("scientific_experiments_rerun") is not False:
        raise ValueError("a passing, explicitly editorial-only verification is required")
    parent = report["parent_snapshot"]
    parent_path = parent["path"] if (root / parent["path"]).exists() else ORIGINAL
    original_bytes = checked_bytes(root, parent_path, parent["sha256"])
    original, run, _ = original_snapshot(original_bytes, parent["sha256"],
                                         parent["run_manifest_sha256"])
    declared = declared_hashes(report)
    files = {}
    for name, expected in original["files"].items():
        # The current archive is included by its current report, not its old name.
        if name == OLD_BUNDLE:
            continue
        final = declared.get(name, expected)
        if final != expected and name not in EDITORIAL_MEMBERS:
            raise ValueError(f"scientific input/output changed without a fresh run: {name}")
        files[name] = checked_bytes(root, name, final)
    for name, expected in declared.items():
        # Avoid recursive packages and unrelated historical archives. The only
        # nested ZIP is the separately named original scientific snapshot.
        if name.endswith(".zip") or name.endswith(".zip.sha256"):
            raise ValueError(f"recursive or undeclared nested archive: {name}")
        files[name] = checked_bytes(root, name, expected)
    files[report_name] = report_bytes
    own_script = "release/package_current.py"
    if own_script not in declared:
        raise ValueError("editorial verification must hash release/package_current.py")
    files["VERIFY_CURRENT_REPRODUCTION.py"] = files[own_script]
    files[ORIGINAL] = original_bytes
    bundle = json.loads(files["results/arxiv_bundle.json"])
    bundle_path = bundle["archive"]
    if bundle_path not in files or bundle_path + ".sha256" not in files:
        raise ValueError("editorial output inventory omits the current arXiv bundle/checksum")
    check_bundle(files[bundle_path], bundle, files.__getitem__)
    # Preserve the current clean-build evidence with portable archive paths.
    clean = bundle["clean_build"]
    clean_paths = {}
    for field in ("pdf", "log"):
        name = clean[field]
        target = "provenance/clean-arxiv-build/" + Path(name).name
        files[target] = checked_bytes(root, name, clean[field + "_sha256"])
        clean_paths[name] = target
    files["CURRENT_REPRODUCTION.txt"] = (
        f"Current verified reproduction {version}\n\n"
        "This archive contains runnable current source, saved scientific results,\n"
        "the final unnumbered manuscript, and its clean-tested arXiv source bundle.\n"
        "The experiments were not rerun for this editorial/release packaging.\n"
        "The original passing twelve-stage run and its manifest are preserved\n"
        f"byte-for-byte inside {ORIGINAL}.\n"
        "results/run_manifest.json belongs to that original scientific run; its\n"
        "earlier manuscript/metadata hashes must not be read as current hashes.\n"
        f"{report_name} identifies the editorial overlay.\n"
        "SNAPSHOT.json identifies every current payload; it does not certify the\n"
        "scientific correctness of those files. Obtain the outer ZIP checksum\n"
        "from the immutable release before extracting.\n\n"
        "Verify integrity with Python 3.9+ and no third-party dependencies:\n"
        "  python VERIFY_CURRENT_REPRODUCTION.py\n\n"
        "Install the locked Python requirements and the documented TeX packages\n"
        "to reproduce scientifically from this archive's root:\n"
        "  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        "VECLIB_MAXIMUM_THREADS=1 python run_all.py\n"
        "This fresh run replaces outputs and writes a new run manifest. After it\n"
        "passes, the strict python package_reproduction.py packages that fresh\n"
        "run. The strict packager intentionally rejects an editorial overlay\n"
        "before a fresh run. README.md explains paper-only rebuilding.\n"
    ).encode()
    snapshot = {
        "schema_version": 1,
        "kind": "verified scientific reproduction with an editorial overlay",
        "version": version,
        "scientific_experiments_rerun_for_packaging": False,
        "current_editorial_report": report_name,
        "current_editorial_report_sha256": digest(report_bytes),
        "original_scientific_snapshot": {"path": ORIGINAL, "sha256": parent["sha256"],
                                         "run_manifest_sha256": parent["run_manifest_sha256"],
                                         "passing_stage_count": len(run["stages"])},
        "clean_build_paths": clean_paths,
        "files": {name: digest(content) for name, content in sorted(files.items())},
    }
    files["SNAPSHOT.json"] = (json.dumps(snapshot, indent=2, sort_keys=True) + "\n").encode()
    files["SHA256SUMS"] = "".join(digest(content) + "  " + name + "\n"
                                     for name, content in sorted(files.items())).encode()
    return files


def verify_tree(root):
    """Check current inventory, the original run, overlay, and bundled TeX inputs."""
    root = Path(root)
    snapshot = json.loads(checked_bytes(root, "SNAPSHOT.json"))
    if (snapshot.get("kind") != "verified scientific reproduction with an editorial overlay"
            or snapshot.get("scientific_experiments_rerun_for_packaging") is not False):
        raise ValueError("unsupported current reproduction inventory")
    files = {name: checked_bytes(root, name, expected)
             for name, expected in snapshot["files"].items()}
    expected_sums = {**files, "SNAPSHOT.json": checked_bytes(root, "SNAPSHOT.json")}
    sums = "".join(digest(content) + "  " + name + "\n"
                   for name, content in sorted(expected_sums.items())).encode()
    if checked_bytes(root, "SHA256SUMS") != sums:
        raise ValueError("current checksum inventory is inconsistent")
    parent = snapshot["original_scientific_snapshot"]
    original, run, _ = original_snapshot(files[parent["path"]], parent["sha256"],
                                         parent["run_manifest_sha256"])
    report_name = snapshot["current_editorial_report"]
    if digest(files[report_name]) != snapshot["current_editorial_report_sha256"]:
        raise ValueError("editorial report hash mismatch")
    report = json.loads(files[report_name])
    if report.get("status") != "pass" or report.get("scientific_experiments_rerun") is not False:
        raise ValueError("editorial verification is not passing")
    if (report["parent_snapshot"]["sha256"] != parent["sha256"]
            or report["parent_snapshot"]["run_manifest_sha256"] != parent["run_manifest_sha256"]):
        raise ValueError("editorial report refers to a different scientific snapshot")
    declared = declared_hashes(report)
    for name, expected in declared.items():
        if name not in files or digest(files[name]) != expected:
            raise ValueError(f"editorial input/output mismatch: {name}")
    for name, expected in original["files"].items():
        if name == OLD_BUNDLE:
            continue
        final = declared.get(name, expected)
        if final != expected and name not in EDITORIAL_MEMBERS:
            raise ValueError(f"scientific input/output changed: {name}")
        if name not in files or digest(files[name]) != final:
            raise ValueError(f"original/current evidence mismatch: {name}")
    bundle = json.loads(files["results/arxiv_bundle.json"])
    source_count = check_bundle(files[bundle["archive"]], bundle, files.__getitem__)
    for field in ("pdf", "log"):
        location = snapshot["clean_build_paths"][bundle["clean_build"][field]]
        if digest(files[location]) != bundle["clean_build"][field + "_sha256"]:
            raise ValueError("clean-build evidence mismatch")
    if digest(files["results/run_manifest.json"]) != parent["run_manifest_sha256"]:
        raise ValueError("original run manifest was rewritten")
    summary = {"status": "pass", "version": snapshot["version"],
               "payload_files": len(files), "original_passing_stages": len(run["stages"]),
               "arxiv_source_files": source_count,
               "scientific_experiments_rerun_for_packaging": False}
    return summary


def extract_checked(archive_path, destination):
    destination = Path(destination)
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("duplicate current archive member")
        for member in archive.infolist():
            safe_name(member.filename)
            if member.is_dir() or stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError("nonregular current archive member")
            target = destination / member.filename
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError("current archive escapes extraction root")
        archive.extractall(destination)
        if len({name.split("/", 1)[0] for name in names}) != 1:
            raise ValueError("current archive needs exactly one root directory")
        root = destination / names[0].split("/", 1)[0]
        summary = verify_tree(root)
        expected = set(json.loads((root / "SNAPSHOT.json").read_text())["files"])
        expected |= {"SNAPSHOT.json", "SHA256SUMS"}
        prefix = root.name + "/"
        if set(names) != {prefix + name for name in expected}:
            raise ValueError("current archive has uninventoried or missing members")
    return root, summary


def package(root=ROOT, report=REPORT, version=VERSION, output=None):
    if not re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+", version):
        raise ValueError("version must be a literal vMAJOR.MINOR.PATCH")
    root = Path(root)
    destination = Path(output) if output is not None else root / f"dist/neurosymbolic-reproduction-{version}.zip"
    files = collect_files(root, report_name=report, version=version)
    prefix = f"neurosymbolic-reproduction-{version}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".zip.tmp")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=9) as archive:
            for name, content in sorted(files.items()):
                info = zipfile.ZipInfo(prefix + "/" + name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, content, compresslevel=9)
        with tempfile.TemporaryDirectory(prefix="current-reproduction-check-") as directory:
            extracted, summary = extract_checked(temporary, Path(directory))
            process = subprocess.run([sys.executable, "VERIFY_CURRENT_REPRODUCTION.py"],
                                     cwd=extracted, text=True, capture_output=True, check=True)
            if json.loads(process.stdout) != summary:
                raise ValueError("standalone and packager verification disagree")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    checksum = digest(destination.read_bytes())
    destination.with_suffix(".zip.sha256").write_text(checksum + "  " + destination.name + "\n")
    receipt = {
        "status": "pass", "version": version,
        "archive": str(destination.relative_to(root)) if destination.is_relative_to(root) else str(destination),
        "sha256": checksum, "size_bytes": destination.stat().st_size,
        "member_count": len(files), "payload_files": len(files) - 2,
        "editorial_report": report,
        "editorial_report_sha256": digest(files[report]),
        "scientific_experiments_rerun": False,
        "checks": ["original twelve-stage scientific snapshot", "unchanged scientific inputs and outputs",
                   "current editorial provenance", "current arXiv source equality and clean-build evidence",
                   "safe fresh extraction", "all payload hashes", "standalone standard-library verifier"],
    }
    (root / "results/current_reproduction.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return destination, checksum, len(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--report", default=REPORT)
    parser.add_argument("--version", default=VERSION)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path, metavar="EXTRACTED_ROOT")
    args = parser.parse_args()
    if args.verify is not None or Path(__file__).name == "VERIFY_CURRENT_REPRODUCTION.py":
        print(json.dumps(verify_tree(args.verify or args.root), indent=2))
    else:
        path, checksum, count = package(args.root, args.report, args.version, args.output)
        print(json.dumps({"status": "pass", "archive": str(path), "sha256": checksum,
                          "member_count": count}, indent=2))


if __name__ == "__main__":
    main()
