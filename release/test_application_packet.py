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
               + r"\end{document}" + "\n").encode()
        sources["paper/nsp_contraction_paper_ieee.tex"] = self.write(root, "paper/nsp_contraction_paper_ieee.tex", tex)
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
        for index in range(13):
            name = f"results/stage_logs/{index}.log"
            stages.append({"status": "pass", "log": name,
                           "log_sha256": self.write(root, name, b"fixture stage\n")})
        run = {"status": "pass", "expected_stage_count": 13, "stages": stages,
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
            self.assertEqual(report["scientific_stages"], 13)
            self.assertEqual(report["publication_status"], "local revision; not a published release")
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
            with self.assertRaisesRegex(ValueError, "new complete scientific run"):
                packet.build(root)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            (root / "application_protocol.json").write_text("changed protocol")
            with self.assertRaisesRegex(ValueError, "changed since"):
                packet.build(root)


if __name__ == "__main__":
    unittest.main()
