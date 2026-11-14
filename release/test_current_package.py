"""Protect scientific provenance and safe extraction of editorial release packages."""
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_arxiv
import package_current as current


class CurrentReproductionTests(unittest.TestCase):
    def write(self, root, name, content):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return current.digest(content)

    def fixture(self, root):
        sources = {"cegis_c.py": b"unchanged scientific program\n",
                   "README.md": b"original readme\n",
                   "paper/nsp_contraction_paper_ieee.tex": b"original manuscript\n"}
        artifacts = {"results/science.json": b'{"scientific_value": 7}\n',
                     "paper/numbers.tex": b"generated scientific numbers\n",
                     "paper/nsp_contraction_paper_ieee.pdf": b"%PDF-1.4\noriginal\n"}
        files = {**sources, **artifacts, "results/run_all.log": b"passing full run\n"}
        stages = []
        for index in range(12):
            name = f"results/stage_logs/{index:02d}.log"
            files[name] = b"pass\n"
            stages.append({"status": "pass", "log": name,
                           "log_sha256": current.digest(files[name])})
        manifest = {"status": "pass", "expected_stage_count": 12, "stages": stages,
                    "sources_at_start": {name: current.digest(value) for name, value in sources.items()},
                    "sources_at_finish": {name: current.digest(value) for name, value in sources.items()},
                    "artifact_sha256": {name: current.digest(value) for name, value in artifacts.items()},
                    "run_log": "results/run_all.log", "run_log_sha256": current.digest(files["results/run_all.log"])}
        files["results/run_manifest.json"] = json.dumps(manifest).encode()
        original = {"files": {name: current.digest(value) for name, value in files.items()},
                    "run_manifest_sha256": current.digest(files["results/run_manifest.json"])}
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for name, content in files.items():
                archive.writestr("original/" + name, content)
            archive.writestr("original/SNAPSHOT.json", json.dumps(original))
        parent_path = "dist/original.zip"
        self.write(root, parent_path, buffer.getvalue())
        for name, content in files.items():
            self.write(root, name, content)
        self.write(root, "README.md", b"current editorial documentation\n")
        self.write(root, "release/package_current.py", Path(current.__file__).read_bytes())
        self.write(root, "tmp/clean/paper.pdf", b"%PDF-1.4\nclean\n")
        self.write(root, "tmp/clean/paper.log", b"clean build pass\n")
        members = {name.removeprefix("paper/"): (root / name).read_bytes()
                   for name in ("paper/nsp_contraction_paper_ieee.tex", "paper/numbers.tex")}
        bundle_name = "dist/neurosymbolic-arxiv-v1.1.0.tar.gz"
        bundle_bytes = build_arxiv.archive_bytes(members)
        self.write(root, bundle_name, bundle_bytes)
        self.write(root, bundle_name + ".sha256", (current.digest(bundle_bytes) + "  " + Path(bundle_name).name + "\n").encode())
        bundle = {"status": "pass", "archive": bundle_name,
                  "sha256": current.digest(bundle_bytes), "member_count": len(members),
                  "member_sha256": {name: current.digest(value) for name, value in members.items()},
                  "clean_build": {"status": "pass", "pdf": "tmp/clean/paper.pdf",
                                  "pdf_sha256": current.digest((root / "tmp/clean/paper.pdf").read_bytes()),
                                  "log": "tmp/clean/paper.log",
                                  "log_sha256": current.digest((root / "tmp/clean/paper.log").read_bytes())}}
        self.write(root, "results/arxiv_bundle.json", json.dumps(bundle).encode())
        report = {"status": "pass", "scientific_experiments_rerun": False,
                  "parent_snapshot": {"path": parent_path, "sha256": current.digest(buffer.getvalue()),
                                      "run_manifest_sha256": original["run_manifest_sha256"]},
                  "editorial_input_changes": {name: {"final_sha256": current.digest((root / name).read_bytes())}
                                              for name in ("README.md", "release/package_current.py")},
                  "unchanged_parent_files": {name: expected for name, expected in original["files"].items()
                                             if name != "README.md"},
                  "output_sha256": {name: current.digest((root / name).read_bytes())
                                    for name in (bundle_name, bundle_name + ".sha256", "results/arxiv_bundle.json")}}
        self.write(root, current.REPORT, json.dumps(report).encode())
        return report

    def test_current_archive_is_repeatable_and_standalone_verifier_passes(self):
        with tempfile.TemporaryDirectory() as folder, tempfile.TemporaryDirectory() as extracted:
            root = Path(folder)
            self.fixture(root)
            path, first_hash, count = current.package(root)
            self.assertEqual(current.package(root)[1], first_hash)
            tree, summary = current.extract_checked(path, Path(extracted))
            self.assertEqual(summary["original_passing_stages"], 12)
            self.assertFalse(summary["scientific_experiments_rerun_for_packaging"])
            self.assertEqual(count, summary["payload_files"] + 2)
            self.assertEqual((tree / "results/run_manifest.json").read_bytes(),
                             (root / "results/run_manifest.json").read_bytes())
            (tree / "results/science.json").write_bytes(b"corrupted saved result\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                current.verify_tree(tree)

    def test_editorial_report_cannot_relabel_changed_science_as_an_overlay(self):
        for name in ("cegis_c.py", "results/science.json", "results/stage_logs/00.log",
                     "paper/numbers.tex", "results/run_manifest.json"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                report = self.fixture(root)
                changed_hash = self.write(root, name, b"changed but no fresh run\n")
                report["unchanged_parent_files"].pop(name)
                report["editorial_input_changes"][name] = {"final_sha256": changed_hash}
                self.write(root, current.REPORT, json.dumps(report).encode())
                with self.assertRaisesRegex(ValueError, "scientific input/output changed"):
                    current.package(root)

    def test_changed_parent_or_failed_clean_build_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            report = self.fixture(root)
            self.write(root, report["parent_snapshot"]["path"], b"wrong original snapshot\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                current.package(root)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            report = self.fixture(root)
            bundle_path = root / "results/arxiv_bundle.json"
            bundle = json.loads(bundle_path.read_text())
            bundle["clean_build"]["status"] = "fail"
            bundle_path.write_text(json.dumps(bundle))
            report["output_sha256"]["results/arxiv_bundle.json"] = current.digest(bundle_path.read_bytes())
            self.write(root, current.REPORT, json.dumps(report).encode())
            with self.assertRaisesRegex(ValueError, "passing clean build"):
                current.package(root)

    def test_extraction_rejects_parent_paths_symlinks_and_uninventoried_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("../outside", "/absolute", "root/../outside", "root\\outside"):
                archive = root / "unsafe.zip"
                with zipfile.ZipFile(archive, "w") as packed:
                    packed.writestr(name, b"unsafe")
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "unsafe"):
                    current.extract_checked(archive, root / "extract")
            self.fixture(root)
            archive, _, _ = current.package(root)
            with zipfile.ZipFile(archive, "a") as packed:
                packed.writestr("neurosymbolic-reproduction-v1.1.0/uninventoried.txt", b"untracked")
            with self.assertRaisesRegex(ValueError, "uninventoried"):
                current.extract_checked(archive, root / "extract")


if __name__ == "__main__":
    unittest.main()
