"""Create and optionally compile the current minimal arXiv source archive.

The archive contains only the main TeX, its generated inputs, and referenced
PDF figures. It excludes the compiled paper, logs, scientific data, programs,
and third-party papers. Archive metadata is normalized for repeatable bytes.
"""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
VERSION = "application-comparison"
MAIN = "nsp_contraction_paper_ieee.tex"
FIGURES = {"fig_" + name + ".pdf" for name in
           ("traces", "heatmaps", "gridcost", "residual", "pendulum", "neural")}
EXPECTED = {MAIN, "numbers.tex"} | FIGURES
ALLOWED = EXPECTED | {"revision_numbers.tex", "revision_tables.tex",
                      "application_numbers.tex", "application_tables.tex"} | {
    "fig_" + name + ".pdf" for name in
    ("ablation", "scale_chase", "extension3d", "surrogate", "application")}
SYSTEM_CLASSES = {"IEEEtran"}
SYSTEM_PACKAGES = {"amsmath", "amssymb", "amsthm", "etoolbox", "lineno",
                   "algorithm", "algpseudocode", "graphicx", "hyperref",
                   "xcolor", "placeins"}


def sha(content):
    return hashlib.sha256(content).hexdigest()


def literal_name(value, suffix):
    """Require a portable flat filename, never an external or parent path."""
    value = value.strip()
    if not re.fullmatch(r"[A-Za-z0-9_+.,=-]+", value) or value in (".", ".."):
        raise ValueError(f"TeX dependency must be a literal flat filename: {value!r}")
    if not Path(value).suffix:
        value += suffix
    if Path(value).suffix != suffix:
        raise ValueError(f"unexpected TeX dependency extension: {value}")
    return value


def uncomment(source):
    return re.sub(r"(?<!\\)%[^\n]*", "", source)


def collect_sources(paper):
    """Collect exactly the transitive literal dependencies from a reviewed allowlist."""
    paper = Path(paper)
    pending = [MAIN]
    files = {}
    while pending:
        name = pending.pop()
        if name not in ALLOWED:
            raise ValueError(f"unreviewed manuscript dependency: {name}")
        if name in files:
            continue
        path = paper / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"missing or symlinked manuscript dependency: {name}")
        files[name] = path.read_bytes()
        if not name.endswith(".tex"):
            continue
        source = uncomment(files[name].decode("utf-8"))
        if re.search(r"\\(?:bibliography|addbibresource|graphicspath|externaldocument)\b", source):
            raise ValueError("external bibliography, graphics paths, or document dependencies need explicit bundling")
        for raw in re.findall(r"\\(?:input|include)\s*\{([^{}]+)\}", source):
            pending.append(literal_name(raw, ".tex"))
        for raw in re.findall(r"\\includegraphics\*?(?:\s*\[[^\]]*\])?\s*\{([^{}]+)\}", source):
            pending.append(literal_name(raw, ".pdf"))
        for kind, raw in re.findall(r"\\(documentclass|usepackage)(?:\s*\[[^\]]*\])?\s*\{([^{}]+)\}", source):
            allowed = SYSTEM_CLASSES if kind == "documentclass" else SYSTEM_PACKAGES
            if not set(part.strip() for part in raw.split(",")).issubset(allowed):
                raise ValueError(f"nonstandard TeX {kind} dependency needs explicit bundling: {raw}")
    if not {MAIN, "numbers.tex"}.issubset(files):
        raise ValueError("manuscript omits its main source or generated numerical input")
    return files


def archive_bytes(files):
    """Stable sorted USTAR members inside a gzip stream without date/name metadata."""
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for name, content in sorted(files.items()):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            info.mode = 0o644
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            archive.addfile(info, io.BytesIO(content))
    gzip_buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=gzip_buffer, mtime=0, compresslevel=9) as compressed:
        compressed.write(tar_buffer.getvalue())
    return gzip_buffer.getvalue()


def extract_checked(archive_path, destination, expected):
    """Extract only the known regular flat members and verify their full hashes."""
    destination = Path(destination)
    seen = set()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            if (not member.isfile() or not re.fullmatch(r"[A-Za-z0-9_+.,=-]+", member.name)
                    or member.name in (".", "..") or member.name not in expected
                    or member.name in seen):
                raise ValueError(f"unexpected arXiv archive member: {member.name}")
            seen.add(member.name)
            content = archive.extractfile(member).read()
            if sha(content) != expected[member.name]:
                raise ValueError(f"arXiv archive hash mismatch: {member.name}")
            (destination / member.name).write_bytes(content)
    if seen != set(expected):
        raise ValueError("arXiv archive omits a required manuscript input")


def validate_latex_log(log):
    errors = re.findall(r"Undefined control sequence|LaTeX Error|Emergency stop|Fatal error|"
                        r"(?:Reference|Citation)[^\n]*undefined|There were undefined (?:references|citations)|"
                        r"multiply[- ]defined|Label[^\n]*multiply defined", log, flags=re.IGNORECASE)
    if errors:
        raise ValueError("clean arXiv compilation failed: " + ", ".join(errors))
    match = re.search(r"Output written on .*\((\d+) pages", log)
    if not match:
        raise ValueError("clean arXiv compilation did not report a completed PDF")
    warnings = re.findall(r"^.*(?:Warning:|Overfull \\[hv]box|Underfull \\[hv]box).*$", log, flags=re.MULTILINE)
    return int(match.group(1)), warnings


def compile_clean(archive, member_hashes, root):
    """Compile extracted source twice in a fresh directory, retaining the evidence."""
    temporary_root = Path(root) / "tmp"
    temporary_root.mkdir(exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=f"arxiv-bundle-check-{VERSION}-", dir=temporary_root))
    extract_checked(archive, directory, member_hashes)
    environment = os.environ.copy()
    for variable in ("TEXINPUTS", "BIBINPUTS", "BSTINPUTS"):
        environment[variable] = ""
    command = ["pdflatex", "-no-shell-escape", "-interaction=nonstopmode", "-halt-on-error", Path(MAIN).stem]
    for attempt in range(1, 5):
        with (directory / f"build-pass-{attempt}.log").open("w") as stream:
            process = subprocess.run(command, cwd=directory, env=environment,
                                     stdout=stream, stderr=subprocess.STDOUT, text=True)
        if process.returncode != 0:
            raise ValueError(f"clean arXiv LaTeX pass {attempt} exited {process.returncode}; see {directory}")
        if attempt >= 2:
            current=(directory / Path(MAIN).with_suffix(".log")).read_text(errors="replace")
            if not re.search(r"Label\(s\) may have changed|rerunfilecheck Warning|Rerun to get",current):
                break
    log_path = directory / Path(MAIN).with_suffix(".log")
    pdf_path = directory / Path(MAIN).with_suffix(".pdf")
    pages, warnings = validate_latex_log(log_path.read_text(errors="replace"))
    if not pdf_path.read_bytes().startswith(b"%PDF-"):
        raise ValueError("clean arXiv build did not produce a PDF")
    return {"status": "pass", "passes": attempt, "pages": pages, "warnings": warnings,
            "directory": str(directory.relative_to(root)),
            "pdf": str(pdf_path.relative_to(root)), "pdf_sha256": sha(pdf_path.read_bytes()),
            "log": str(log_path.relative_to(root)), "log_sha256": sha(log_path.read_bytes()),
            "command": command}


def build(root=ROOT, check=False, output=None):
    root = Path(root)
    files = collect_sources(root / "paper")
    member_hashes = {name: sha(content) for name, content in sorted(files.items())}
    output = Path(output) if output is not None else root / f"dist/neurosymbolic-arxiv-{VERSION}.tar.gz"
    output.parent.mkdir(parents=True, exist_ok=True)
    content = archive_bytes(files)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(output)
    output.with_name(output.name + ".sha256").write_text(sha(content) + "  " + output.name + "\n")
    report = {"schema_version": 2, "status": "prepared", "release_tag": None,
              "manuscript_revision": VERSION, "publication_status": "local revision; not a published release",
              "kind": "minimal arXiv TeX source bundle",
              "created_at": datetime.now(timezone.utc).isoformat(),
              "archive": str(output.relative_to(root)) if output.is_relative_to(root) else str(output),
              "sha256": sha(content), "main_tex": MAIN, "member_count": len(files),
              "member_sha256": member_hashes, "deterministic_archive_metadata": True,
              "system_tex_class": sorted(SYSTEM_CLASSES), "system_tex_packages": sorted(SYSTEM_PACKAGES),
              "clean_build": {"status": "not run"}}
    results = root / "results"
    results.mkdir(exist_ok=True)
    manifest = results / "arxiv_bundle.json"
    manifest.write_text(json.dumps(report, indent=2) + "\n")
    if check:
        report["status"] = "running"
        manifest.write_text(json.dumps(report, indent=2) + "\n")
        try:
            report["clean_build"] = compile_clean(output, member_hashes, root)
            report["status"] = "pass"
        except Exception as error:
            report.update(status="fail", error=f"{type(error).__name__}: {error}")
            manifest.write_text(json.dumps(report, indent=2) + "\n")
            raise
        manifest.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compile extracted source twice in a clean directory")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = build(check=arguments.check, output=arguments.output)
    print(f"Wrote {report['archive']}: {report['member_count']} files, SHA256 {report['sha256']}")
    if arguments.check:
        check = report["clean_build"]
        print(f"Clean build PASS: {check['pages']} pages, {len(check['warnings'])} warning(s); {check['directory']}")


if __name__ == "__main__":
    main()
