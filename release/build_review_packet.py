"""Assemble the current, explicitly versioned independent-review packet."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import subprocess
import shutil
import sys
import tarfile
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DOCS = Path(__file__).resolve().parent / "review_packet"
DATE = "2026-11-14"
EDITORIAL_REPORT = "results/editorial_v1.1.0.json"
NAME = "neurosymbolic-claude-deep-review-v1.1.0"
OUT = ROOT / "dist" / (NAME + ".zip")
PAYLOAD = {}
ORIGINS = {}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def add(name, content, origin=None):
    if name in PAYLOAD:
        raise ValueError("Duplicate package path: " + name)
    PAYLOAD[name] = content
    if origin:
        ORIGINS[name] = origin


def copy(name, origin):
    path = ROOT / origin
    if not path.exists() and origin == "dist/neurosymbolic-reproduction-claude-revision.zip":
        path = ROOT / "provenance/original-scientific-snapshot.zip"
    if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
        raise ValueError("Unsafe input path: " + origin)
    add(name, path.read_bytes(), origin)


def main():
    report = json.loads((ROOT / EDITORIAL_REPORT).read_text())
    if report["status"] != "pass":
        raise ValueError("Current editorial verification did not pass")
    parent = report["parent_snapshot"]
    parent_path = ROOT / parent["path"]
    if not parent_path.exists():
        parent_path = ROOT / "provenance/original-scientific-snapshot.zip"
    parent_bytes = parent_path.read_bytes()
    if digest(parent_bytes) != parent["sha256"]:
        raise ValueError("Original scientific snapshot changed")
    for name, expected in report["output_sha256"].items():
        if digest((ROOT / name).read_bytes()) != expected:
            raise ValueError("Current output changed: " + name)
    for name, info in report["editorial_input_changes"].items():
        if digest((ROOT / name).read_bytes()) != info["final_sha256"]:
            raise ValueError("Current input changed: " + name)
    with zipfile.ZipFile(parent_path) as archive:
        member = next(name for name in archive.namelist()
                      if name.endswith("/SNAPSHOT.json"))
        prefix = member[:-len("SNAPSHOT.json")]
        snapshot = json.loads(archive.read(member))
        for name, expected in snapshot["files"].items():
            if digest(archive.read(prefix + name)) != expected:
                raise ValueError("Original snapshot is inconsistent: " + name)
    for name, expected in report["unchanged_parent_files"].items():
        if digest((ROOT / name).read_bytes()) != expected:
            raise ValueError("Scientific input/output changed: " + name)
    for name in ("START_HERE.txt", "REVIEW_PROMPT.txt", "VERIFY_PACKAGE.py"):
        add(name, (DOCS / name).read_bytes())
    copy("PAPER_CURRENT.pdf", "paper/nsp_contraction_paper_ieee.pdf")
    copy("ARXIV_ABSTRACT.txt", "dist/arxiv_abstract.txt")
    pdftotext = shutil.which("pdftotext")
    if pdftotext is None:
        raise RuntimeError("Install Poppler and put pdftotext on PATH to create the review text export")
    process = subprocess.run([str(pdftotext), "-layout", "-enc", "UTF-8",
                              str(ROOT / "paper/nsp_contraction_paper_ieee.pdf"), "-"],
                             capture_output=True, check=True)
    header = ("TEXT EXTRACTION OF PAPER_CURRENT.pdf\n"
              "The PDF is authoritative and has no reviewer line numbers. Cite its\n"
              "pages and section or object labels. Equations/tables may lose\n"
              "structure in extraction. Consult the PDF for every mathematical claim.\n\n")
    add("PAPER_CURRENT.txt", header.encode() + process.stdout)
    bundle = json.loads((ROOT / "results/arxiv_bundle.json").read_text())
    with tarfile.open(ROOT / bundle["archive"], "r:gz") as archive:
        for member in archive:
            if not member.isfile() or "/" in member.name or "\\" in member.name:
                raise ValueError("Unsafe manuscript member")
            content = archive.extractfile(member).read()
            if content != (ROOT / "paper" / member.name).read_bytes():
                raise ValueError("Current source differs from tested bundle")
            add("source/" + member.name, content, "paper/" + member.name)
    for origin in sorted(snapshot["files"]):
        if origin.endswith(".py"):
            copy("code/" + origin, origin)
        elif origin in ("requirements.txt", "requirements-lock.txt"):
            copy("code/" + origin, origin)
        elif origin.startswith("results/"):
            copy("evidence/" + origin, origin)
    for origin in ("results/editorial_v1.1.0.json",
                   "results/editorial_v1.1.0.log",
                   "results/editorial_empirical_finding.json",
                   "results/editorial_empirical_finding.log",
                   "results/editorial_references_ai.json",
                   "results/editorial_references_ai.log",
                   "results/editorial_no_lines.json",
                   "results/editorial_no_lines.log",
                   "results/editorial_abstract.json",
                   "results/editorial_abstract.log",
                   "results/editorial_neurosymbolic.json",
                   "results/editorial_neurosymbolic.log",
                   "results/editorial_bullo_positioning.json",
                   "results/editorial_bullo_positioning.log",
                   "results/reference_audit_before_bullo.json",
                   "results/reference_audit.json"):
        copy("evidence/" + origin, origin)
    for origin in report["editorial_input_changes"]:
        if origin == "paper/nsp_contraction_paper_ieee.tex":
            continue
        if origin in ("README.md", "CITATION.cff", "RESPONSE_TO_CLAUDE.md"):
            continue
        target = ("optional_history/RESPONSE_TO_EARLIER_CLAUDE_REVIEW.txt"
                  if origin == "RESPONSE_TO_CLAUDE.md" else "code/" + origin)
        if target not in PAYLOAD:
            copy(target, origin)
    copy("evidence/project_metadata/README_FROM_PROJECT.txt", "README.md")
    copy("evidence/project_metadata/CITATION.cff", "CITATION.cff")
    copy("reproduction/neurosymbolic-reproduction-claude-revision.zip", parent["path"])
    add("reproduction/neurosymbolic-reproduction-claude-revision.zip.sha256",
        (parent["sha256"] + "  neurosymbolic-reproduction-claude-revision.zip\n").encode())
    for origin in (bundle["archive"], bundle["archive"] + ".sha256"):
        copy("reproduction/" + Path(origin).name, origin)
    for filename, origin in (
        ("EARLIER_CLAUDE_REVIEW.txt", "reviews/claude-2026-11-04.md"),
        ("RESPONSE_TO_EARLIER_CLAUDE_REVIEW.txt", "RESPONSE_TO_CLAUDE.md"),
        ("EARLIER_RESPONSE_RECORD.txt", "RESPONSE_TO_REFEREE.md")):
        copy("optional_history/" + filename, origin)
    current_reproduction = json.loads((ROOT / "results/current_reproduction.json").read_text())
    copy("evidence/results/current_reproduction.json", "results/current_reproduction.json")
    for origin in (current_reproduction["archive"], current_reproduction["archive"] + ".sha256"):
        copy("reproduction/" + Path(origin).name, origin)
    clean_directory = ROOT / bundle["clean_build"]["directory"]
    for path in sorted(clean_directory.glob("*.log")):
        copy("evidence/clean_arxiv_build/" + path.name, str(path.relative_to(ROOT)))
    copy("evidence/clean_arxiv_build/PAPER_COMPILED_FROM_CLEAN_SOURCE.pdf",
         bundle["clean_build"]["pdf"])
    for origin in report["output_sha256"]:
        if origin not in ORIGINS.values():
            copy("evidence/current_outputs/" + Path(origin).name, origin)
    manifest = {
        "schema_version": 1,
        "kind": "Independent deep pre-arXiv review packet for immutable release v1.1.0",
        "prepared_at": datetime.now(ZoneInfo("America/New_York")).isoformat(),
        "current_paper": {"pdf": "PAPER_CURRENT.pdf", "pages": report["pages"],
                          "reviewer_line_numbers": False,
                          "abstract_characters": report["abstract"]["characters"],
                          "abstract_words": report["abstract"]["words"],
                          "reference_count": report["reference_count"],
                          "citation_style": "PDF page and section or object label",
                          "title": report["title"]},
        "scientific_experiments_rerun_for_packaging": False,
        "current_editorial_report": EDITORIAL_REPORT,
        "release_tag": "v1.1.0",
        "release_url": "https://github.com/vmpreciado/NeuroSymbolic/releases/tag/v1.1.0",
        "files": {name: {"sha256": digest(content), "size_bytes": len(content),
                         **({"origin": ORIGINS[name]} if name in ORIGINS else {})}
                  for name, content in sorted(PAYLOAD.items())},
        "provenance_note": "Original full scientific snapshot is unchanged; current editorial report identifies the final manuscript. Prepared as an asset of immutable release v1.1.0. Prior reviews are optional history. Checksums establish integrity, not scientific correctness."
    }
    PAYLOAD["PACKAGE_MANIFEST.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    PAYLOAD["SHA256SUMS.txt"] = "".join(digest(content) + "  " + name + "\n"
                                          for name, content in sorted(PAYLOAD.items())).encode()
    OUT.parent.mkdir(exist_ok=True)
    temporary = OUT.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as archive:
        for name, content in sorted(PAYLOAD.items()):
            info = zipfile.ZipInfo(NAME + "/" + name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, content, compresslevel=9)
    temporary.replace(OUT)
    checksum = digest(OUT.read_bytes())
    OUT.with_suffix(".zip.sha256").write_text(checksum + "  " + OUT.name + "\n")
    (ROOT / "dist/neurosymbolic-claude-review-prompt.txt").write_bytes(PAYLOAD["REVIEW_PROMPT.txt"])
    with tempfile.TemporaryDirectory(prefix="packet-check-", dir=DOCS) as directory:
        destination = Path(directory)
        with zipfile.ZipFile(OUT) as archive:
            if archive.testzip() is not None:
                raise ValueError("ZIP CRC check failed")
            for member in archive.infolist():
                target = (destination / member.filename).resolve()
                if not target.is_relative_to(destination.resolve()):
                    raise ValueError("ZIP path escapes extraction root")
            archive.extractall(destination)
        process = subprocess.run([sys.executable, "VERIFY_PACKAGE.py"],
                                 cwd=destination / NAME, text=True,
                                 capture_output=True, check=True)
        print(process.stdout, end="")
    summary = {"status": "pass", "archive": str(OUT.relative_to(ROOT)),
               "sha256": checksum, "size_bytes": OUT.stat().st_size,
               "member_count": len(PAYLOAD), "payload_files": len(manifest["files"]),
               "current_pdf_sha256": digest(PAYLOAD["PAPER_CURRENT.pdf"]),
               "current_tex_sha256": digest(PAYLOAD["source/nsp_contraction_paper_ieee.tex"]),
               "scientific_experiments_rerun": False,
               "checks": ["ZIP CRC", "safe fresh extraction", "all payload hashes",
                          "original scientific snapshot", "current reproduction overlay", "current editorial provenance",
                          "13 current source dependencies", "clean-build evidence"]}
    (ROOT / "results/review_packet.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
