"""Failures must not turn into successful reproductions."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

import reproduce
from verify_results import compare, measurements


class InputTests(unittest.TestCase):
    def test_changed_observation_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "observation.csv").write_bytes(b"changed")
            expected = [{"path": "observation.csv", "sha256": hashlib.sha256(b"original").hexdigest()}]
            with self.assertRaisesRegex(RuntimeError, "Missing or changed input"):
                reproduce.verify_records(root, expected)

    def test_incomplete_archive_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = root / "empty.tar.gz"
            with tarfile.open(archive, "w:gz"):
                pass
            expected = [{"path": "raw/needed.csv", "bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()}]
            with self.assertRaisesRegex(RuntimeError, "missing observations"):
                reproduce.unpack(archive, root / "out", expected)

    def test_unlisted_archive_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = root / "extra.tar.gz"
            with tarfile.open(archive, "w:gz") as bundle:
                info = tarfile.TarInfo("unexpected.txt")
                info.size = 1
                bundle.addfile(info, io.BytesIO(b"x"))
            with self.assertRaisesRegex(RuntimeError, "Unexpected archive member"):
                reproduce.unpack(archive, root / "out", [])


class ResultTests(unittest.TestCase):
    def test_lost_validation_signal_fails(self):
        record = {"80364427.ztf.validation_gain": 33.57}
        checked = compare({"80364427.ztf.validation_gain": 0.0}, record)
        self.assertFalse(checked[0]["passed"])

    def test_missing_or_nonfinite_result_fails(self):
        for value in (None, float("nan"), float("inf")):
            self.assertFalse(compare({"gain": value}, {"gain": 20.0})[0]["passed"])

    def test_counts_must_match_exactly(self):
        self.assertFalse(compare({"n": 358}, {"n": 359})[0]["passed"])

    def test_reference_includes_failed_and_incomplete_checks(self):
        values = measurements(Path(__file__).resolve().parents[1] / "reference/data")
        self.assertEqual(values["80364427.spectra.broad_colour_test"], "insufficient_usable_passband_coverage")
        self.assertEqual(values["65258778.images.planned"] - values["65258778.images.extracted"], 40)


if __name__ == "__main__":
    unittest.main()
