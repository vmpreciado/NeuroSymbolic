"""Overlapping producers retain exit-code, freshness and cleanup guarantees."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import os
import run_all


class ParallelRunnerTests(unittest.TestCase):
    def execute(self,body,existing=False):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            results=root/"results"
            (results/"stage_logs").mkdir(parents=True)
            (root/"producer.py").write_text(body)
            if existing:
                (results/"output.json").write_text("{}")
                os.utime(results/"output.json",(1,1))
            with patch.object(run_all,"HERE",root),patch.object(run_all,"RESULTS",results),\
                    patch.object(run_all,"PRECOMPUTED",{}),contextlib.redirect_stdout(io.StringIO()):
                try:
                    run_all.start_prefetch("producer.py")
                    run_all.run_script("producer.py",outputs=("results/output.json",))
                finally:
                    run_all.stop_prefetch()

    def test_fresh_success_is_collected_once(self):
        self.execute("from pathlib import Path\nPath('results/output.json').write_text('{}')\n")

    def test_printed_success_cannot_hide_failure(self):
        with self.assertRaisesRegex(run_all.ValidationError,"prefetched process exited 3"):
            self.execute("print('CERTIFIED')\nraise SystemExit(3)\n")

    def test_prefetch_cannot_reuse_stale_output(self):
        with self.assertRaisesRegex(run_all.ValidationError,"stale output"):
            self.execute("print('finished without producing output')\n",existing=True)


if __name__ == "__main__":unittest.main()
