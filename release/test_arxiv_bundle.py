"""Integrity checks for the minimal arXiv source archive, using temporary inputs."""
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_arxiv as bundle


class ArxivBundleTests(unittest.TestCase):
    def fixture(self, root):
        paper = root / "paper"
        paper.mkdir()
        main = (r"\documentclass[12pt,onecolumn,journal]{IEEEtran}" + "\n"
                + r"\usepackage{graphicx}" + "\n"
                + r"\begin{document}\input{numbers.tex}" + "\n"
                + "\n".join(r"\includegraphics[width=.5\textwidth]{" + name + "}"
                            for name in sorted(bundle.FIGURES))
                + "\n" + r"\end{document}" + "\n")
        (paper / bundle.MAIN).write_text(main)
        (paper / "numbers.tex").write_text(r"\providecommand{\Count}{20}" + "\n")
        for name in bundle.FIGURES:
            (paper / name).write_bytes(b"%PDF-1.4\nfixture figure\n")
        return paper

    def test_archive_is_complete_minimal_and_repeatable_after_timestamp_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paper = self.fixture(root)
            for name in ("secret.txt", "unused.pdf", Path(bundle.MAIN).with_suffix(".pdf").name,
                         Path(bundle.MAIN).with_suffix(".aux").name):
                (paper / name).write_text("must not enter source archive")
            first = bundle.build(root)
            archive = root / first["archive"]
            content = archive.read_bytes()
            with tarfile.open(archive, "r:gz") as packed:
                self.assertEqual(set(packed.getnames()), bundle.EXPECTED)
                for member in packed.getmembers():
                    self.assertEqual(packed.extractfile(member).read(), (paper / member.name).read_bytes())
            for path in paper.iterdir():
                os.utime(path, (1, 1))
            second = bundle.build(root)
            self.assertEqual(second["sha256"], first["sha256"])
            self.assertEqual(archive.read_bytes(), content)
            self.assertEqual(first["member_count"], 8)

    def test_missing_input_or_figure_and_external_dependencies_are_rejected(self):
        for name in ("numbers.tex", sorted(bundle.FIGURES)[0]):
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                paper = self.fixture(root)
                (paper / name).unlink()
                with self.assertRaisesRegex(ValueError, "missing"):
                    bundle.build(root)
        for addition in (r"\input{../outside.tex}", r"\input{/outside.tex}",
                         r"\bibliography{private}", r"\usepackage{my_private_style}"):
            with self.subTest(addition=addition), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                paper = self.fixture(root)
                with (paper / bundle.MAIN).open("a") as stream:
                    stream.write(addition)
                with self.assertRaises(ValueError):
                    bundle.build(root)

    def test_extraction_rejects_unsafe_members_and_content_tampering(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = root / "bad.tar.gz"
            content = b"outside"
            with tarfile.open(archive, "w:gz") as packed:
                info = tarfile.TarInfo("../outside.tex")
                info.size = len(content)
                packed.addfile(info, io.BytesIO(content))
            with self.assertRaisesRegex(ValueError, "unexpected"):
                bundle.extract_checked(archive, root, {"../outside.tex": bundle.sha(content)})
            paper = self.fixture(root)
            report = bundle.build(root)
            hashes = dict(report["member_sha256"])
            hashes[bundle.MAIN] = "bad hash"
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                bundle.extract_checked(root / report["archive"], root, hashes)

    def test_failed_clean_build_cannot_leave_a_passing_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            with patch.object(bundle, "compile_clean", side_effect=ValueError("TeX failure")):
                with self.assertRaisesRegex(ValueError, "TeX failure"):
                    bundle.build(root, check=True)
            report = json.loads((root / "results/arxiv_bundle.json").read_text())
            self.assertEqual(report["status"], "fail")

    def test_revision_dependencies_are_transitive_and_unused_files_are_excluded(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            paper=self.fixture(root)
            with (paper/bundle.MAIN).open("a") as stream:
                stream.write(r"\input{revision_numbers.tex}\input{revision_tables.tex}")
            (paper/"revision_numbers.tex").write_text(r"\providecommand{\RevCount}{3}")
            (paper/"revision_tables.tex").write_text(r"\includegraphics{fig_extension3d.pdf}")
            (paper/"fig_extension3d.pdf").write_bytes(b"%PDF-1.4\nfigure\n")
            (paper/"fig_surrogate.pdf").write_bytes(b"unused\n")
            files=bundle.collect_sources(paper)
            self.assertIn("fig_extension3d.pdf",files)
            self.assertNotIn("fig_surrogate.pdf",files)
            self.assertEqual(len(files),11)

    def test_application_dependencies_and_release_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paper = self.fixture(root)
            with (paper / bundle.MAIN).open("a") as stream:
                stream.write(r"\input{application_numbers.tex}\input{application_tables.tex}")
            (paper / "application_numbers.tex").write_text(r"\providecommand{\AppCount}{10}")
            (paper / "application_tables.tex").write_text(r"\includegraphics{fig_application.pdf}")
            (paper / "fig_application.pdf").write_bytes(b"%PDF-1.4\nfixture\n")
            report = bundle.build(root)
            self.assertEqual(report["manuscript_revision"], "v1.2.0")
            self.assertEqual(report["release_tag"], "v1.2.0")
            self.assertEqual(report["release_url"], bundle.RELEASE_URL)
            self.assertEqual(report["publication_status"], bundle.PUBLICATION_STATUS)
            self.assertTrue({"application_numbers.tex", "application_tables.tex", "fig_application.pdf"}
                            .issubset(report["member_sha256"]))


if __name__ == "__main__":
    unittest.main()
