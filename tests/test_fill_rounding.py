"""Printed certificate constants retain their specified bound direction."""
from decimal import Decimal, localcontext
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import fill_tex


def decimal_latex(text):
    if r" \times 10^{" in text:
        mantissa, exponent = text.split(r" \times 10^{")
        return Decimal(mantissa).scaleb(int(exponent[:-1]))
    return Decimal(text)


class DirectedDisplayTests(unittest.TestCase):
    def test_display_bounds_enclose_full_precision_values(self):
        values = [0., .1, -.1, 5.93343477, .002248679, 99999.9, 1000.,
                  1e-200, -1e200, np.nextafter(1., 0.), np.nextafter(1., 2.),
                  np.nextafter(0., 1.), np.finfo(float).max]
        with localcontext() as context:
            context.prec = 800
            for x in values:
                exact = Decimal.from_float(float(x))
                for digits in (2, 3, 6):
                    with self.subTest(value=x, digits=digits):
                        self.assertLessEqual(decimal_latex(fill_tex.fmt_down(x, digits)), exact)
                        self.assertGreaterEqual(decimal_latex(fill_tex.fmt_up(x, digits)), exact)

    def test_old_unrounded_results_cannot_fill_the_rebuilt_paper(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder)/"traces.json").write_text(json.dumps({"config": {}}))
            with patch.object(fill_tex, "RESULTS", folder):
                with self.assertRaisesRegex(ValueError, "outward-rounded"):
                    fill_tex._load("traces.json")


if __name__ == "__main__":
    unittest.main()
