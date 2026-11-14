"""Integrity and completeness checks for the reviewer reproduction archive."""
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import package_reproduction as packaging


class ReproductionPackageTests(unittest.TestCase):
    def fixture(self, root):
        sources = {name: self.write(root, name, b"source\n")
                   for name in packaging.REQUIRED_SOURCES}
        artifacts = {name: self.write(root, name, b"artifact\n")
                     for name in packaging.REQUIRED_ARTIFACTS}
        stages = []
        for index in range(8):
            name = f"results/stage_logs/{index}.log"
            stages.append({"status": "pass", "log": name,
                           "log_sha256": self.write(root, name, b"pass\n")})
        manifest = {"status": "pass", "stages": stages,
                    "sources_at_start": sources, "sources_at_finish": dict(sources),
                    "artifact_sha256": artifacts, "finished_at": "2026-11-04",
                    "run_log": "results/run_all.log",
                    "run_log_sha256": self.write(root, "results/run_all.log", b"pass\n")}
        self.write(root, "results/run_manifest.json", json.dumps(manifest).encode())
        return manifest

    def write(self, root, name, content):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return packaging.digest(content)

    def test_complete_archive_contains_verifiable_snapshot_and_is_repeatable(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            path, first_hash, count = packaging.package(root)
            with zipfile.ZipFile(path) as archive:
                prefix = packaging.PREFIX + "/"
                snapshot = json.loads(archive.read(prefix + "SNAPSHOT.json"))
                self.assertEqual(count, len(snapshot["files"]) + 1)
                for name, expected in snapshot["files"].items():
                    self.assertEqual(packaging.digest(archive.read(prefix + name)), expected)
            self.assertEqual(packaging.package(root)[1], first_hash)
            summary, contents = packaging.verify_archive(path)
            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["stages"], 8)
            self.assertEqual(set(contents), set(snapshot["files"]))

    def test_changed_source_artifact_or_log_is_rejected(self):
        for name in ("rounding.py", "results/neural.json", "results/stage_logs/4.log"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                self.fixture(root)
                (root / name).write_bytes(b"changed\n")
                with self.assertRaisesRegex(ValueError, "changed since"):
                    packaging.package(root)

    def test_incomplete_failed_or_mutating_run_is_rejected(self):
        for mutation in (lambda m: m.update(status="fail"),
                         lambda m: m["stages"].pop(),
                         lambda m: m["artifact_sha256"].pop("results/neural.json"),
                         lambda m: m["sources_at_finish"].update({"rounding.py": "changed"})):
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                manifest = self.fixture(root)
                mutation(manifest)
                (root / "results/run_manifest.json").write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    packaging.package(root)

    def test_snapshot_paths_cannot_escape_project(self):
        with tempfile.TemporaryDirectory() as folder:
            for name in ("../outside", "/absolute", "..\\outside"):
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "unsafe"):
                    packaging.checked_bytes(Path(folder), name)

    def test_archive_verification_rejects_tampering_and_extra_members(self):
        for extra in (False, True):
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                self.fixture(root)
                path, _, _ = packaging.package(root)
                with zipfile.ZipFile(path) as archive:
                    members = {name: archive.read(name) for name in archive.namelist()}
                if extra:
                    members[packaging.PREFIX + "/unrecorded.txt"] = b"extra"
                else:
                    members[packaging.PREFIX + "/results/neural.json"] = b"tampered"
                with zipfile.ZipFile(path, "w") as archive:
                    for name, content in members.items():
                        archive.writestr(name, content)
                with self.assertRaises(ValueError):
                    packaging.verify_archive(path)


if __name__ == "__main__":
    unittest.main()
