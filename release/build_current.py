"""Rebuild the v1.1.0 editorial paper from the unchanged verified science.

This validates saved results and replays adaptive certificates. It does not
rerun learning or replace the original twelve-stage run manifest.
"""
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
VERSION = "v1.1.0"
REPORT = "results/editorial_v1.1.0.json"
LOG = "results/editorial_v1.1.0.log"
PARENT = "dist/neurosymbolic-reproduction-claude-revision.zip"
PARENT_SHA = "c1cdea042cca4f5495a3ed025174570bb68e494df2695641af2e97d91437eb39"
CURRENT_INPUTS = (
    "paper/nsp_contraction_paper_ieee.tex", "README.md", "CITATION.cff",
    "CLAUDE.md", "RESPONSE_TO_CLAUDE.md", "release/build_current.py",
    "release/package_current.py", "release/test_current_package.py",
    "release/build_review_packet.py", "release/RELEASE_NOTES_v1.1.0.md",
    "release/review_packet/START_HERE.txt", "release/review_packet/REVIEW_PROMPT.txt",
    "release/review_packet/VERIFY_PACKAGE.py",
)
CHANGED_PARENT_OUTPUTS = {
    "paper/nsp_contraction_paper_ieee.pdf", "results/arxiv_bundle.json",
    "dist/neurosymbolic-arxiv-claude-revision.tar.gz",
}


def sha(content):
    return hashlib.sha256(content).hexdigest()


def read(name):
    path = ROOT / name
    if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
        raise ValueError("Unsafe project path: " + name)
    return path.read_bytes()


def check_parent():
    parent_path = PARENT if (ROOT / PARENT).is_file() else "provenance/original-scientific-snapshot.zip"
    if sha(read(parent_path)) != PARENT_SHA:
        raise ValueError("The original verified scientific archive changed")
    with zipfile.ZipFile(ROOT / parent_path) as archive:
        member = next(name for name in archive.namelist() if name.endswith("/SNAPSHOT.json"))
        prefix = member[:-len("SNAPSHOT.json")]
        snapshot = json.loads(archive.read(member))
        for name, expected in snapshot["files"].items():
            if sha(archive.read(prefix + name)) != expected:
                raise ValueError("Original archive member changed: " + name)
        original_run = archive.read(prefix + "results/run_manifest.json")
    run = json.loads(original_run)
    if (run["status"] != "pass" or len(run["stages"]) != 12
            or any(stage["status"] != "pass" for stage in run["stages"])
            or run["sources_at_start"] != run["sources_at_finish"]
            or read("results/run_manifest.json") != original_run):
        raise ValueError("Original twelve-stage evidence is not intact")
    unchanged = {}
    for name, expected in snapshot["files"].items():
        if name in CURRENT_INPUTS or name in CHANGED_PARENT_OUTPUTS:
            continue
        if sha(read(name)) != expected:
            raise ValueError("Scientific input or output changed: " + name)
        unchanged[name] = expected
    return snapshot, run, unchanged, parent_path


def main():
    os.environ["PATH"] = "/Library/TeX/texbin:" + os.environ.get("PATH", "")
    started = datetime.now(timezone.utc).isoformat()
    snapshot, run, unchanged, parent_path = check_parent()
    input_changes = {
        name: {"parent_sha256": snapshot["files"].get(name), "final_sha256": sha(read(name))}
        for name in CURRENT_INPUTS
    }
    tex = read("paper/nsp_contraction_paper_ieee.tex").decode()
    if any(hook in tex for hook in ("\\linenumbers", "{lineno}", "linenomath")):
        raise ValueError("Reviewer line numbers are enabled")
    title = re.search(r"\\title\{([^}]+)\}", tex).group(1)
    abstract = re.search(r"\\begin\{abstract\}\s*(.*?)\s*\\end\{abstract\}", tex, re.S).group(1)
    if not abstract.isascii() or len(abstract) > 1920:
        raise ValueError("The arXiv metadata abstract is invalid")
    if "did not demonstrate an advantage in certification counts within the tested budgets" not in abstract:
        raise ValueError("Main empirical finding is missing")
    audit = json.loads(read("results/reference_audit.json"))
    keys = re.findall(r"\\bibitem\{([^}]+)\}", tex)
    citations = {key.strip() for group in re.findall(r"\\cite\{([^}]+)\}", tex) for key in group.split(",")}
    if (audit["status"] != "pass" or len(keys) != len(set(keys))
            or set(keys) != citations or citations != {item["key"] for item in audit["entries"]}):
        raise ValueError("Bibliographic keys and audit disagree")
    import run_all
    from release import build_arxiv
    run_all.SCIENCE_REPLAY_WORKERS = max(1, min(3, os.cpu_count() or 1))
    validations = {}
    log_path = ROOT / LOG
    with log_path.open("w") as stream, redirect_stdout(stream), redirect_stderr(stream):
        print("Editorial rebuild of", VERSION, "from an unchanged twelve-stage scientific run", flush=True)
        print("No learning or experiment producers are rerun.", flush=True)
        for label, action in (
            ("main outcomes", lambda: run_all.validate_traces(run_all.read_json("traces.json"))),
            ("residual budget", lambda: run_all.validate_residual(run_all.read_json("residual.json"))),
            ("pendulum", lambda: run_all.validate_benchmark(run_all.read_json("pendulum.json"))),
            ("neural", lambda: run_all.validate_benchmark(run_all.read_json("neural.json"), neural=True)),
            ("smoke", lambda: run_all.validate_smoke(run_all.read_json("smoke_p1.json"))),
            ("controlled revision studies", run_all.validate_revision_experiments),
            ("adaptive certificate replay and surrogate transfer", run_all.validate_revision_extensions),
        ):
            outcome = action()
            validations[label] = str(outcome) if outcome is not None else "pass"
            print(label + ": " + validations[label], flush=True)
        generated_names = ("paper/numbers.tex", "paper/revision_numbers.tex", "paper/revision_tables.tex")
        original_numbers = {name: read(name) for name in generated_names}
        run_all.run_command([sys.executable, "fill_tex.py"])
        for name, original in original_numbers.items():
            regenerated = read(name)
            strip_git_comment = lambda value: re.sub(
                rb"^% Baseline Git HEAD[^\n]*$", b"", value, flags=re.M)
            if regenerated != original:
                if name != "paper/numbers.tex" or strip_git_comment(regenerated) != strip_git_comment(original):
                    raise ValueError("Generated scientific inputs changed: " + name)
                (ROOT / name).write_bytes(original)
                print("Generated macros and source hashes identical; preserved original Git source-freeze comment.", flush=True)
        latex_status = run_all.latex()
        print(latex_status, flush=True)
        bundle = build_arxiv.build(root=ROOT, check=True,
                                   output=ROOT / "dist/neurosymbolic-arxiv-v1.1.0.tar.gz")
        bundle.update(release_tag=VERSION, manuscript_revision=VERSION,
                      publication_status="prepared release asset; public publication is verified separately")
        (ROOT / "results/arxiv_bundle.json").write_text(json.dumps(bundle, indent=2) + "\n")
        print("Clean source compilation:", bundle["clean_build"]["status"], flush=True)
    pages, warnings = run_all.validate_latex_log((ROOT / "paper/nsp_contraction_paper_ieee.log").read_text())
    if warnings or bundle["status"] != "pass" or bundle["clean_build"]["warnings"]:
        raise ValueError("A warning-free paper and clean source build are required")
    _, _, final_unchanged, _ = check_parent()
    if final_unchanged != unchanged:
        raise ValueError("Science changed during the editorial build")
    for name, info in input_changes.items():
        if sha(read(name)) != info["final_sha256"]:
            raise ValueError("Editorial input changed during the build: " + name)
    (ROOT / "dist/arxiv_abstract.txt").write_text(abstract + "\n")
    for name in ("dist/neurosymbolic-paper-v1.1.0.pdf", "dist/neurosymbolic_paper_without_line_numbers.pdf",
                 "dist/paper_without_line_numbers.pdf"):
        shutil.copyfile(ROOT / "paper/nsp_contraction_paper_ieee.pdf", ROOT / name)
    outputs = (
        "paper/nsp_contraction_paper_ieee.pdf", "dist/neurosymbolic-paper-v1.1.0.pdf",
        "dist/arxiv_abstract.txt", "dist/neurosymbolic-arxiv-v1.1.0.tar.gz",
        "dist/neurosymbolic-arxiv-v1.1.0.tar.gz.sha256", "results/arxiv_bundle.json",
        "results/reference_audit.json", "results/reference_audit_before_v1.1.0.json", LOG,
        "results/reference_audit_before_bullo.json",
        *("results/editorial_" + name + suffix for name in (
            "empirical_finding", "references_ai", "no_lines", "abstract", "neurosymbolic", "bullo_positioning"
        ) for suffix in (".json", ".log")),
    )
    report = {
        "schema_version": 1, "kind": "editorial-only rebuild", "status": "pass",
        "built_at": started, "verified_at": datetime.now(timezone.utc).isoformat(),
        "title": title, "release_tag": VERSION,
        "publication_status": "prepared for immutable release; publication is checked separately",
        "scientific_experiments_rerun": False,
        "parent_snapshot": {"path": parent_path, "sha256": PARENT_SHA,
                            "run_manifest_sha256": sha(read("results/run_manifest.json")),
                            "full_run_status": "pass", "stage_count": 12},
        "editorial_input_changes": input_changes, "unchanged_parent_files": unchanged,
        "output_sha256": {name: sha(read(name)) for name in outputs},
        "pages": pages, "reviewer_line_numbers": False, "reference_count": len(keys),
        "abstract": {"characters": len(abstract), "words": len(abstract.split()),
                     "limit_characters": 1920, "standalone_file": "dist/arxiv_abstract.txt",
                     "empirical_finding_preserved": True},
        "arxiv_bundle": bundle["archive"], "saved_result_validation": validations,
        "adaptive_replay_workers": run_all.SCIENCE_REPLAY_WORKERS,
        "changes": ["dual CCM and direct primal controller synthesis distinguished",
                    "novelty hierarchy clarified", "Davydov training objective compared explicitly",
                    "availability and citation identify current immutable v1.1.0 assets"],
        "checks": {"latex": latex_status, "clean_arxiv_build": "pass",
                   "original_scientific_snapshot_unchanged": True,
                   "visual_inspection": "Required separately before publication; not an automated build assertion"},
    }
    (ROOT / REPORT).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "pass", "report": REPORT, "pages": pages,
                      "scientific_experiments_rerun": False, "latex": latex_status}, indent=2))


if __name__ == "__main__":
    main()
