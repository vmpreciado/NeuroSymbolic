"""Regression checks for fresh-run review packaging and standalone verification."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_application_packet as packet
import build_arxiv
import package_reproduction
import verify_application_packet


class ApplicationPacketTests(unittest.TestCase):
    def write(self, root, name, content):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return hashlib.sha256(content).hexdigest()

    def fixture(self, root):
        sources = {name: self.write(root, name, b"fixture source\n")
                   for name in package_reproduction.REQUIRED_SOURCES}
        artifacts = {name: self.write(root, name, b"fixture artifact\n")
                     for name in package_reproduction.REQUIRED_ARTIFACTS}
        tex = (r"\documentclass{IEEEtran}\title{Fixture application paper}" + "\n"
               + r"\begin{document}\begin{abstract}A fixture abstract.\end{abstract}" + "\n"
               + r"\input{numbers.tex}\input{application_numbers.tex}\input{application_tables.tex}" + "\n"
               + packet.RELEASE_URL + "\n" + r"\end{document}" + "\n").encode()
        sources["paper/nsp_contraction_paper_ieee.tex"] = self.write(root, "paper/nsp_contraction_paper_ieee.tex", tex)
        citation = ('version: 1.2.0\nurl: "' + packet.RELEASE_URL + '"\n').encode()
        sources["CITATION.cff"] = self.write(root, "CITATION.cff", citation)
        baseline = {"schema_version": 1, "status": "pass",
                    "method": "fixed_constant_metric_analytical_baseline",
                    "model_id": "two_machine_swing_exact_pi_over_3_v1",
                    "post_hoc": True, "training_performed": False, "seeds": None,
                    "requested_rate": 0.05, "requested_raw_margin": 0.0001,
                    "metric": [[1.0, 0.5], [0.5, 2.0]],
                    "verification_backend": {"outward_rounded": True,
                        "exact_rational_matrix_checks": True, "transcendental_backend": "MPFR"},
                    "domains": [{"box_half_width": width, "certified": width == 0.25,
                        "status": "certified" if width == 0.25 else "fixed_candidate_violates_requested_rate"}
                        for width in [0.25, 0.5, 0.75]]}
        artifacts["results/application_constant_baseline.json"] = self.write(
            root, "results/application_constant_baseline.json", json.dumps(baseline).encode())
        verifier = Path(verify_application_packet.__file__).read_bytes()
        sources["release/verify_application_packet.py"] = self.write(root, "release/verify_application_packet.py", verifier)
        artifacts["paper/numbers.tex"] = self.write(root, "paper/numbers.tex", b"numbers\n")
        artifacts["paper/application_numbers.tex"] = self.write(root, "paper/application_numbers.tex", b"application numbers\n")
        artifacts["paper/application_tables.tex"] = self.write(root, "paper/application_tables.tex", b"application tables\n")
        report = build_arxiv.build(root)
        pdf = "tmp/clean/manuscript.pdf"
        log = "tmp/clean/manuscript.log"
        report.update(status="pass", clean_build={"status": "pass", "pages": 1, "warnings": [],
                      "pdf": pdf, "pdf_sha256": self.write(root, pdf, b"%PDF-fixture\n"),
                      "log": log, "log_sha256": self.write(root, log, b"fixture build log\n")})
        artifacts["results/arxiv_bundle.json"] = self.write(root, "results/arxiv_bundle.json", json.dumps(report).encode())
        artifacts[report["archive"]] = hashlib.sha256((root / report["archive"]).read_bytes()).hexdigest()
        stages = []
        for index in range(14):
            name = f"results/stage_logs/{index}.log"
            stages.append({"status": "pass", "log": name,
                           "log_sha256": self.write(root, name, b"fixture stage\n")})
        run = {"status": "pass", "expected_stage_count": 14, "stages": stages,
               "sources_at_start": sources, "sources_at_finish": dict(sources),
               "artifact_sha256": artifacts, "finished_at": "fixture time",
               "run_log": "results/run_all.log",
               "run_log_sha256": self.write(root, "results/run_all.log", b"fixture run\n")}
        self.write(root, "results/run_manifest.json", json.dumps(run).encode())

    def build_fixture(self, root):
        self.fixture(root)
        real_run = subprocess.run

        def run(command, **kwargs):
            if command[0] == "fixture-pdftotext":
                return subprocess.CompletedProcess(command, 0, stdout=b"Fixture paper text\n", stderr=b"")
            return real_run(command, **kwargs)

        with patch.object(packet.shutil, "which", return_value="fixture-pdftotext"), patch.object(packet.subprocess, "run", side_effect=run):
            return packet.build(root)

    def test_roundtrip_and_standalone_verifier_with_no_nested_code_execution(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            report = self.build_fixture(root)
            self.assertEqual(report["scientific_stages"], 14)
            self.assertEqual(report["publication_status"], packet.PUBLICATION_STATUS)
            self.assertEqual(report["release_tag"], "v1.2.0")
            with zipfile.ZipFile(root / report["archive"]) as archive:
                archive.extractall(root / "extracted")
            destination = root / "extracted" / packet.NAME
            checked = verify_application_packet.verify(destination)
            self.assertEqual(checked["status"], "pass")
            (destination / "project/application_protocol.json").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                verify_application_packet.verify(destination)

    def test_old_or_changed_run_cannot_be_packaged(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            run = json.loads((root / "results/run_manifest.json").read_text())
            run.update(expected_stage_count=12, stages=run["stages"][:12])
            (root / "results/run_manifest.json").write_text(json.dumps(run))
            with self.assertRaisesRegex(ValueError, "complete passing reproduction"):
                packet.build(root)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            (root / "application_protocol.json").write_text("changed protocol")
            with self.assertRaisesRegex(ValueError, "changed since"):
                packet.build(root)

    def test_citation_and_baseline_scope_must_match_release(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            names = ["CITATION.cff", "paper/nsp_contraction_paper_ieee.tex",
                     "results/application_constant_baseline.json"]
            project = {name: (root / name).read_bytes() for name in names}
            verify_application_packet.check_release_sources(project)
            changed = dict(project)
            changed["CITATION.cff"] = project["CITATION.cff"].replace(b"v1.2.0", b"v1.1.1")
            with self.assertRaisesRegex(ValueError, "citation disagree"):
                verify_application_packet.check_release_sources(changed)
            for key, value in [("post_hoc", False), ("seeds", 20), ("requested_raw_margin", 0.0)]:
                with self.subTest(key=key):
                    changed = dict(project)
                    baseline = json.loads(project[names[2]])
                    baseline[key] = value
                    changed[names[2]] = json.dumps(baseline).encode()
                    with self.assertRaisesRegex(ValueError, "identity or scope"):
                        verify_application_packet.check_release_sources(changed)

    def test_release_metadata_cannot_claim_a_different_tag(self):
        metadata = {"release_tag": packet.VERSION, "release_url": packet.RELEASE_URL,
                    "publication_status": packet.PUBLICATION_STATUS}
        verify_application_packet.release_metadata(metadata)
        metadata["release_tag"] = "v1.1.1"
        with self.assertRaisesRegex(ValueError, "Release metadata"):
            verify_application_packet.release_metadata(metadata)


if __name__ == "__main__":
    unittest.main()
