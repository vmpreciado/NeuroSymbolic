"""Verify an extracted v1.2.2 review packet using only stdlib.

Integrity and provenance checks are not mathematical verification or a rerun.
No code from the nested scientific reproduction is executed.
"""
from pathlib import Path, PurePosixPath
import hashlib
import io
import json
import re
import stat
import tarfile
import zipfile

VERSION = "v1.2.2"
RELEASE_URL = "https://github.com/vmpreciado/NeuroSymbolic/releases/tag/" + VERSION
PUBLICATION_STATUS = "release artifact; publication and immutability are established by the GitHub release record"


def release_metadata(record):
    if (record.get("release_tag") != VERSION or record.get("release_url") != RELEASE_URL
            or record.get("publication_status") != PUBLICATION_STATUS):
        raise ValueError("Release metadata mismatch")


def check_release_sources(project):
    """Check release/citation and baseline identity without executing project code."""
    citation = project["CITATION.cff"].decode()
    version = re.search(r"^version:\s*['\"]?(v?1\.2\.2)['\"]?\s*$", citation, re.M)
    url = re.search(r"^url:\s*['\"]?([^'\"\s]+)['\"]?\s*$", citation, re.M)
    tex = project["paper/nsp_contraction_paper_ieee.tex"].decode()
    if not version or not url or url.group(1) != RELEASE_URL or RELEASE_URL not in tex:
        raise ValueError("Current release and manuscript citation disagree")
    baseline = json.loads(project["results/application_constant_baseline.json"])
    if (baseline.get("status") != "pass"
            or baseline.get("method") != "fixed_constant_metric_analytical_baseline"
            or baseline.get("model_id") != "two_machine_swing_exact_pi_over_3_v1"
            or baseline.get("post_hoc") is not True
            or baseline.get("training_performed") is not False
            or baseline.get("seeds") is not None
            or baseline.get("requested_rate") != 0.05
            or baseline.get("requested_raw_margin") != 0.0001
            or baseline.get("metric") != [[1.0, 0.5], [0.5, 2.0]]):
        raise ValueError("Constant-metric baseline identity or scope mismatch")
    backend = baseline.get("verification_backend", {})
    if (backend.get("outward_rounded") is not True
            or backend.get("exact_rational_matrix_checks") is not True
            or backend.get("transcendental_backend") != "MPFR"):
        raise ValueError("Constant-metric baseline verification metadata missing")
    expected = [(0.25, True, "certified"),
                (0.5, False, "fixed_candidate_violates_requested_rate"),
                (0.75, False, "fixed_candidate_violates_requested_rate")]
    if [(row.get("box_half_width"), row.get("certified"), row.get("status"))
            for row in baseline.get("domains", [])] != expected:
        raise ValueError("Constant-metric baseline domain outcomes mismatch")
    polynomial = json.loads(project["results/polynomial_constant_baseline.json"])
    if (polynomial.get("status") != "pass" or polynomial.get("certified") is not True
            or polynomial.get("method") != "fixed_constant_metric_analytical_baseline"
            or polynomial.get("model_id") != "triangular_polynomial_exact_v1"
            or polynomial.get("post_hoc") is not True
            or polynomial.get("training_performed") is not False
            or polynomial.get("seeds") is not None
            or polynomial.get("domain", {}).get("box_rational") != [["-2/1", "2/1"], ["-2/1", "2/1"]]
            or polynomial.get("metric_rational") != [["12/1", "0/1"], ["0/1", "1/1"]]):
        raise ValueError("Polynomial constant-metric baseline identity or scope mismatch")
    if polynomial.get("verification_backend", {}).get("exact_rational_matrix_checks") is not True:
        raise ValueError("Polynomial constant-metric baseline verification metadata missing")
    for key, rate in [("decimal_target_check", "2/5"),
                      ("shared_binary64_target_check", "3602879701896397/9007199254740992")]:
        check = polynomial.get(key, {})
        if (check.get("certified") is not True or check.get("rate_rational") != rate
                or check.get("raw_margin_rational") != "1/20"
                or check.get("normalized_margin_rational") != "1/260"
                or len(check.get("endpoint_checks", [])) != 2):
            raise ValueError("Polynomial constant-metric baseline target metadata mismatch")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def safe_name(name):
    path = PurePosixPath(name)
    if (path.is_absolute() or ".." in path.parts or "\\" in name
            or not name or str(path) != name):
        raise ValueError("Unsafe package path: " + name)
    return path


def read(root, name):
    safe_name(name)
    path = root / name
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Package path escapes root: " + name)
    return path.read_bytes()


def snapshot_from_zip(content):
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        members = archive.infolist()
        names = [member.filename for member in members]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate reproduction member")
        roots = set()
        for member in members:
            path = safe_name(member.filename)
            if (member.is_dir() or stat.S_ISLNK(member.external_attr >> 16)
                    or len(path.parts) < 2):
                raise ValueError("Unsafe reproduction member")
            roots.add(path.parts[0])
        if len(roots) != 1:
            raise ValueError("Reproduction must have exactly one root")
        prefix = roots.pop() + "/"
        snapshot = json.loads(archive.read(prefix + "SNAPSHOT.json"))
        files = {name: archive.read(prefix + name) for name in snapshot["files"]}
        if set(names) != {prefix + name for name in files} | {prefix + "SNAPSHOT.json"}:
            raise ValueError("Reproduction inventory differs from snapshot")
    for name, expected in snapshot["files"].items():
        safe_name(name)
        if digest(files[name]) != expected:
            raise ValueError("Reproduction hash mismatch: " + name)
    run_bytes = files["results/run_manifest.json"]
    if digest(run_bytes) != snapshot["run_manifest_sha256"]:
        raise ValueError("Run manifest hash mismatch")
    run = json.loads(run_bytes)
    stages = run["stages"]
    if (run["status"] != "pass" or run["expected_stage_count"] < 15
            or len(stages) != run["expected_stage_count"]
            or any(stage["status"] != "pass" for stage in stages)
            or run["sources_at_start"] != run["sources_at_finish"]):
        raise ValueError("A complete stable passing application and baseline run is required")
    release_metadata(snapshot)
    required_sources = {"application_comparison.py", "application_protocol.json",
                        "application_baseline.py", "application_verifier.py", "application_report.py",
                        "application_constant_baseline.py", "tests/test_application_constant_baseline.py",
                        "polynomial_constant_baseline.py", "tests/test_polynomial_constant_baseline.py",
                        "CITATION.cff",
                        "comparison_sources/fitzsimmons_liu_2024.json",
                        "comparison_sources/preliminary_application_summary.json",
                        "release/build_application_packet.py", "release/verify_application_packet.py"}
    required_artifacts = {"results/application_comparison.json", "paper/application_numbers.tex",
                          "paper/application_tables.tex", "paper/fig_application.pdf",
                          "results/application_constant_baseline.json",
                          "results/polynomial_constant_baseline.json", "paper/polynomial_constant_numbers.tex"}
    if (not required_sources.issubset(run["sources_at_finish"])
            or not required_artifacts.issubset(run["artifact_sha256"])):
        raise ValueError("Application provenance is missing from the full run")
    recorded = dict(run["sources_at_finish"])
    recorded.update(run["artifact_sha256"])
    recorded[run["run_log"]] = run["run_log_sha256"]
    for stage in stages:
        recorded[stage["log"]] = stage["log_sha256"]
    recorded["results/run_manifest.json"] = digest(run_bytes)
    if set(recorded) != set(files):
        raise ValueError("Snapshot payload differs from recorded run")
    for name, expected in recorded.items():
        if digest(files[name]) != expected:
            raise ValueError("Recorded run hash mismatch: " + name)
    return snapshot, files, run


def verify(root):
    root = Path(root).resolve()
    manifest = json.loads(read(root, "PACKAGE_MANIFEST.json"))
    files = manifest["files"]
    for name, info in files.items():
        content = read(root, name)
        if digest(content) != info["sha256"] or len(content) != info["size_bytes"]:
            raise ValueError("Packet hash or size mismatch: " + name)
    checksums = {}
    for line in read(root, "SHA256SUMS.txt").decode().splitlines():
        expected, name = line.split("  ", 1)
        if name in checksums or digest(read(root, name)) != expected:
            raise ValueError("Invalid checksum inventory: " + name)
        checksums[name] = expected
    if set(checksums) != set(files) | {"PACKAGE_MANIFEST.json"}:
        raise ValueError("Checksum inventory is incomplete")
    actual = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
    if actual != set(checksums) | {"SHA256SUMS.txt"}:
        raise ValueError("Packet contains missing or uninventoried files")
    release_metadata(manifest)
    if (manifest.get("manuscript_revision") != VERSION
            or not manifest["full_scientific_pipeline_rerun"]):
        raise ValueError("Unexpected publication or run metadata")
    snapshot, project, run = snapshot_from_zip(read(root, manifest["reproduction"]["path"]))
    if digest(read(root, manifest["reproduction"]["path"])) != manifest["reproduction"]["sha256"]:
        raise ValueError("Reproduction archive hash mismatch")
    for name, content in project.items():
        if read(root, "project/" + name) != content:
            raise ValueError("Visible project differs from reproduction: " + name)
    if json.loads(read(root, "project/SNAPSHOT.json")) != snapshot:
        raise ValueError("Visible snapshot metadata differs from reproduction")
    if read(root, "PAPER_CURRENT.pdf") != project["paper/nsp_contraction_paper_ieee.pdf"]:
        raise ValueError("Current PDF differs from the fresh full-run PDF")
    check_release_sources(project)
    constant = manifest["constant_metric_baseline"]
    if (constant.get("path") != "project/results/application_constant_baseline.json"
            or constant.get("sha256") != digest(project["results/application_constant_baseline.json"])):
        raise ValueError("Constant-metric baseline release inventory mismatch")
    polynomial = manifest["polynomial_constant_metric_baseline"]
    if (polynomial.get("path") != "project/results/polynomial_constant_baseline.json"
            or polynomial.get("sha256") != digest(project["results/polynomial_constant_baseline.json"])):
        raise ValueError("Polynomial constant-metric baseline release inventory mismatch")
    tex = project["paper/nsp_contraction_paper_ieee.tex"].decode()
    title = re.search(r"\\title\{([^}]+)\}", tex).group(1)
    abstract = re.search(r"\\begin\{abstract\}\s*(.*?)\s*\\end\{abstract\}", tex, re.S).group(1)
    metadata = manifest["current_paper"]
    if (title != metadata["title"] or abstract != read(root, "ARXIV_ABSTRACT.txt").decode().strip()
            or not abstract.isascii() or "\\" in abstract or len(abstract) > 1920
            or len(abstract) != metadata["abstract_characters"]
            or len(abstract.split()) != metadata["abstract_words"]):
        raise ValueError("Manuscript title or abstract metadata mismatch")
    if any(hook in tex for hook in ("\\linenumbers", "{lineno}", "linenomath")):
        raise ValueError("Reviewer line numbers are enabled")
    if metadata["reference_count"] != len(re.findall(r"\\bibitem\{", tex)):
        raise ValueError("Reference count mismatch")
    bundle = json.loads(project["results/arxiv_bundle.json"])
    release_metadata(bundle)
    if bundle.get("manuscript_revision") != VERSION:
        raise ValueError("arXiv revision does not match the release")
    if (bundle["status"] != "pass" or bundle["clean_build"]["status"] != "pass"
            or bundle["clean_build"]["warnings"]
            or metadata["pages"] != bundle["clean_build"]["pages"]):
        raise ValueError("The arXiv source lacks a clean passing compilation")
    arxiv_bytes = read(root, manifest["arxiv_source"])
    if digest(arxiv_bytes) != bundle["sha256"] or arxiv_bytes != project[bundle["archive"]]:
        raise ValueError("arXiv source archive differs from the full run")
    with tarfile.open(fileobj=io.BytesIO(arxiv_bytes), mode="r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)) or set(names) != set(bundle["member_sha256"]):
            raise ValueError("arXiv source inventory mismatch")
        for member in members:
            safe_name(member.name)
            if not member.isfile() or "/" in member.name:
                raise ValueError("arXiv source must contain flat regular files")
            content = archive.extractfile(member).read()
            if (digest(content) != bundle["member_sha256"][member.name]
                    or content != project["paper/" + member.name]):
                raise ValueError("arXiv member differs from the current manuscript")
    for kind in ("pdf", "log"):
        if digest(read(root, "clean_arxiv_build/" + kind)) != bundle["clean_build"][kind + "_sha256"]:
            raise ValueError("Clean arXiv build evidence differs from its receipt")
    return {"status": "pass", "payload_files": len(files), "scientific_stages": len(run["stages"]),
            "arxiv_source_files": len(bundle["member_sha256"]), "pages": metadata["pages"],
            "release_tag": VERSION, "release_url": RELEASE_URL,
            "publication_status": manifest["publication_status"],
            "note": "Integrity/provenance only; no experiment rerun or proof of mathematical correctness."}


if __name__ == "__main__":
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2))
