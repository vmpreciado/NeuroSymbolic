"""Verify this extracted review packet without third-party dependencies.

This checks file integrity and build provenance, not scientific correctness.
"""
from pathlib import Path, PurePosixPath
import hashlib
import json
import io
import stat
import subprocess
import sys
import tempfile
import re
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parent


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError("Unsafe package path: " + name)
    target = ROOT / name
    if target.is_symlink() or not target.resolve().is_relative_to(ROOT):
        raise ValueError("Package path escapes root: " + name)
    return target.read_bytes()


def main():
    manifest = json.loads(read("PACKAGE_MANIFEST.json"))
    files = manifest["files"]
    for name, info in files.items():
        content = read(name)
        if digest(content) != info["sha256"] or len(content) != info["size_bytes"]:
            raise ValueError("File integrity mismatch: " + name)
    checksums = {}
    for line in read("SHA256SUMS.txt").decode().splitlines():
        checksum, name = line.split("  ", 1)
        if name in checksums:
            raise ValueError("Duplicate checksum path: " + name)
        checksums[name] = checksum
        if digest(read(name)) != checksum:
            raise ValueError("Checksum mismatch: " + name)
    if set(checksums) != set(files) | {"PACKAGE_MANIFEST.json"}:
        raise ValueError("Checksum inventory is incomplete")
    origins = {info["origin"]: name for name, info in files.items()
               if "origin" in info}
    report = json.loads(read(origins[manifest["current_editorial_report"]]))
    if report["status"] != "pass" or report["scientific_experiments_rerun"]:
        raise ValueError("Unexpected editorial-build status")
    for origin, expected in report["output_sha256"].items():
        if digest(read(origins[origin])) != expected:
            raise ValueError("Current editorial output mismatch: " + origin)
    for origin, change in report["editorial_input_changes"].items():
        if digest(read(origins[origin])) != change["final_sha256"]:
            raise ValueError("Current manuscript input mismatch: " + origin)
    tex = read("source/nsp_contraction_paper_ieee.tex").decode()
    title = re.search(r"\\title\{([^}]+)\}", tex).group(1)
    if title != report["title"] or title != manifest["current_paper"]["title"]:
        raise ValueError("Current manuscript title differs from packet metadata")
    for name in ("START_HERE.txt", "REVIEW_PROMPT.txt"):
        if title not in " ".join(read(name).decode().split()):
            raise ValueError("Review instructions use an obsolete manuscript title")
    abstract = re.search(r"\\begin\{abstract\}\s*(.*?)\s*\\end\{abstract\}", tex, re.S).group(1)
    if abstract != read("ARXIV_ABSTRACT.txt").decode().strip():
        raise ValueError("Metadata abstract differs from manuscript abstract")
    if (not abstract.isascii() or len(abstract) > 1920
            or len(abstract) != report["abstract"]["characters"]
            or len(abstract.split()) != report["abstract"]["words"]):
        raise ValueError("Abstract violates the recorded metadata limit or counts")
    if "\\linenumbers" in tex or "{lineno}" in tex or "linenomath" in tex:
        raise ValueError("Reviewer line numbering has been reintroduced")
    if "did not demonstrate an advantage in certification counts within the tested budgets" not in abstract:
        raise ValueError("Main empirical finding missing from abstract")
    parent = report["parent_snapshot"]
    if digest(read(origins[parent["path"]])) != parent["sha256"]:
        raise ValueError("Original full-run archive mismatch")
    with zipfile.ZipFile(ROOT / origins[parent["path"]]) as archive:
        member = next(name for name in archive.namelist()
                      if name.endswith("/SNAPSHOT.json"))
        prefix = member[:-len("SNAPSHOT.json")]
        snapshot = json.loads(archive.read(member))
        for name, expected in snapshot["files"].items():
            if digest(archive.read(prefix + name)) != expected:
                raise ValueError("Original snapshot member mismatch: " + name)
        run_bytes = archive.read(prefix + "results/run_manifest.json")
        if digest(run_bytes) != parent["run_manifest_sha256"]:
            raise ValueError("Original run-manifest mismatch")
        run = json.loads(run_bytes)
        if (run["status"] != "pass" or len(run["stages"]) != 12
                or any(stage["status"] != "pass" for stage in run["stages"])
                or run["sources_at_start"] != run["sources_at_finish"]):
            raise ValueError("Original scientific run is not a stable passing run")
        if read(origins["results/run_manifest.json"]) != run_bytes:
            raise ValueError("Supplied original run evidence changed")
        for name, expected in report["unchanged_parent_files"].items():
            if snapshot["files"][name] != expected:
                raise ValueError("Editorial parent hash mismatch: " + name)
            if name in origins and digest(read(origins[name])) != expected:
                raise ValueError("Scientific evidence changed: " + name)
    bundle = json.loads(read(origins["results/arxiv_bundle.json"]))
    if bundle["status"] != "pass" or bundle["clean_build"]["warnings"]:
        raise ValueError("Current source bundle has no clean passing build")
    if digest(read(origins[bundle["archive"]])) != bundle["sha256"]:
        raise ValueError("Current source archive mismatch")
    with tarfile.open(ROOT / origins[bundle["archive"]], "r:gz") as archive:
        members = archive.getmembers()
        if len(members) != len(bundle["member_sha256"]) or len(members) != 13:
            raise ValueError("Wrong manuscript dependency inventory")
        names = set()
        for member in members:
            if (not member.isfile() or "/" in member.name or "\\" in member.name
                    or member.name in names):
                raise ValueError("Unexpected source archive member")
            names.add(member.name)
            content = archive.extractfile(member).read()
            if digest(content) != bundle["member_sha256"][member.name]:
                raise ValueError("Source archive content mismatch")
            if content != read("source/" + member.name):
                raise ValueError("Visible manuscript source differs from archive")
    for kind in ("pdf", "log"):
        name = bundle["clean_build"][kind]
        if digest(read(origins[name])) != bundle["clean_build"][kind + "_sha256"]:
            raise ValueError("Clean manuscript build evidence mismatch")
    audit = json.loads(read(origins["results/reference_audit.json"]))
    bibliography_keys = re.findall(r"\\bibitem\{([^}]+)\}", tex)
    citation_keys = {key.strip() for group in re.findall(r"\\cite\{([^}]+)\}", tex)
                     for key in group.split(",")}
    audit_keys = [entry["key"] for entry in audit["entries"]]
    if (audit["status"] != "pass" or audit["entry_count"] != len(bibliography_keys)
            or len(bibliography_keys) != len(set(bibliography_keys))
            or len(audit_keys) != len(set(audit_keys))
            or set(audit_keys) != set(bibliography_keys) or citation_keys != set(audit_keys)
            or audit["entry_count"] != report["reference_count"]
            or audit["entry_count"] != manifest["current_paper"]["reference_count"]
            or not all(entry["exists"] and entry["primary_sources"] for entry in audit["entries"])):
        raise ValueError("Reference audit record mismatch")
    current = json.loads(read(origins["results/current_reproduction.json"]))
    current_bytes = read(origins[current["archive"]])
    if current["status"] != "pass" or digest(current_bytes) != current["sha256"]:
        raise ValueError("Current reproduction archive mismatch")
    with tempfile.TemporaryDirectory(prefix="current-reproduction-review-check-") as directory:
        destination = Path(directory)
        with zipfile.ZipFile(io.BytesIO(current_bytes)) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError("Duplicate current reproduction member")
            for member in archive.infolist():
                path = PurePosixPath(member.filename)
                if (path.is_absolute() or ".." in path.parts or "\\" in member.filename
                        or member.is_dir() or stat.S_ISLNK(member.external_attr >> 16)):
                    raise ValueError("Unsafe current reproduction member")
            roots = {name.split("/", 1)[0] for name in names}
            if len(roots) != 1:
                raise ValueError("Current reproduction has multiple roots")
            archive.extractall(destination)
        project = destination / roots.pop()
        process = subprocess.run([sys.executable, "VERIFY_CURRENT_REPRODUCTION.py"],
                                 cwd=project, capture_output=True, text=True, check=True)
        summary = json.loads(process.stdout)
        if (summary["status"] != "pass" or summary["version"] != "v1.1.0"
                or summary["arxiv_source_files"] != 13):
            raise ValueError("Nested current reproduction verification failed")
        if ((project / "paper/nsp_contraction_paper_ieee.pdf").read_bytes() != read("PAPER_CURRENT.pdf")
                or (project / "paper/nsp_contraction_paper_ieee.tex").read_bytes() != read("source/nsp_contraction_paper_ieee.tex")):
            raise ValueError("Nested current reproduction uses a different manuscript")
    print("PASS: " + str(len(files)) + " payload files and checksum inventory.")
    print("PASS: original 12-stage scientific snapshot, v1.1.0 reproduction overlay, and current editorial provenance.")
    print("PASS: current PDF, 13 manuscript inputs, source archive, and build evidence.")
    print("PASS: identical manuscript/metadata abstract within 1,920 characters; no line-numbering hooks.")
    print("Integrity checks do not establish mathematical correctness or rerun experiments.")


if __name__ == "__main__":
    main()
