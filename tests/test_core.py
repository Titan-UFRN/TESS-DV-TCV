from __future__ import annotations

import hashlib
import io
import pathlib
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_acceptance_analyses import (  # noqa: E402
    annotate_event_association,
    design,
    phase_error,
    selected_tce_number,
)
from validate_temporal import average_precision, roc_auc  # noqa: E402
from verify_inputs import verify  # noqa: E402


class EventAssociationTests(unittest.TestCase):
    def test_phase_error_wraps_by_period(self) -> None:
        self.assertAlmostEqual(phase_error(110.01, 100.0, 10.0), 0.01)

    def test_selected_tce_number(self) -> None:
        self.assertEqual(selected_tce_number("tess-s0001-s0036-00000001-02_3"), 3)
        self.assertIsNone(selected_tce_number(""))

    def test_strict_event_requires_tce_and_xml_agreement(self) -> None:
        row = {
            "historic_period_days": "10",
            "historic_epoch_btjd": "100",
            "historic_transit_duration_hours": "2.4",
            "dv_data_tperiod": "10.05",
            "dv_data_tepoch": "110.01",
            "dv_candidate_match": "1",
            "dv_selected_tce": "product_2",
            "xml_found": "1",
            "xml_planetcandidate_orbitalperiodindays": "10.05",
            "xml_planetcandidate_epochtjd": "100.01",
            "xml_planetcandidate_planetnumber": "2",
            "toi": "123.01",
            "xml_planetresults_toiid": "123.01",
        }
        strict = annotate_event_association([row])[0]
        self.assertEqual(strict["event_dv_strict"], 1)
        self.assertEqual(strict["event_xml_strict"], 1)
        self.assertEqual(strict["event_ambiguous"], 0)

        row["xml_planetcandidate_planetnumber"] = "3"
        mismatched = annotate_event_association([row])[0]
        self.assertEqual(mismatched["event_dv_strict"], 1)
        self.assertEqual(mismatched["event_xml_strict"], 0)
        self.assertEqual(mismatched["event_ambiguous"], 1)
        self.assertIn("xml_tce", mismatched["event_ambiguity_reason"])


class PreprocessingAndMetricTests(unittest.TestCase):
    def test_design_uses_training_median_for_missing_test_values(self) -> None:
        rows = [{"x": 1.0}, {"x": 3.0}, {"x": ""}]
        _, preprocessing = design(rows, [0, 1], ["x"])
        transformed, _ = design(rows, [2], ["x"], preprocessing)
        self.assertAlmostEqual(float(transformed[0, 0]), 0.0)

    def test_ranking_metrics_on_perfect_order(self) -> None:
        labels = np.array([0.0, 1.0, 0.0, 1.0])
        scores = np.array([0.1, 0.8, 0.2, 0.9])
        self.assertAlmostEqual(average_precision(labels, scores), 1.0)
        self.assertAlmostEqual(roc_auc(labels, scores), 1.0)


class InputVerificationTests(unittest.TestCase):
    def test_checksum_verification_detects_match_and_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            target = root / "input.txt"
            target.write_bytes(b"paper input\n")
            expected = hashlib.sha256(target.read_bytes()).hexdigest()
            checksums = root / "SHA256SUMS"
            checksums.write_text(f"{expected}  input.txt\n", encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                self.assertTrue(verify(checksums, root))
                target.write_bytes(b"changed\n")
                self.assertFalse(verify(checksums, root))


if __name__ == "__main__":
    unittest.main()
