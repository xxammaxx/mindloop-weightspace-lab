import io
import json
import zipfile
from unittest import TestCase

from weightspace_ci import cpu_exploratory as cpu
from weightspace_ci import pilot, publish_cpu_result as pub


def valid_data():
    row = {"correct": 12, "total": 24, "unsafe_write": 0}
    return {
        "schema": 1, "experiment": cpu.EXPERIMENT,
        "context": "github-public-cpu-exploratory",
        "run_id": "123", "source_sha": "a"*40,
        "model": pilot.MODEL, "revision": pilot.REVISION,
        "weights_sha256": pilot.WEIGHT_SHA, "data_sha256": cpu.EXPECTED_DATA_SHA,
        "transformers": "5.12.0", "torch": "2.9.1+cpu", "numpy": "2.2.6",
        "device": "cpu", "seed": pilot.SEED, "dev_count": 24, "holdout_count": 24,
        "metrics": {**{k: dict(row) for k in ("base", "prompt", "rank2", "placebo")},
                    "delta_rank": 2},
        "factorization_error": 1e-9, "duration_seconds": 120.1,
    }


def archive(rows):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, value in rows:
            z.writestr(name, value)
    return buf.getvalue()


class PublisherTests(TestCase):
    def test_exact_cpu_evidence_passes(self):
        data = valid_data()
        self.assertEqual(pub.validate_report(data, "123", "a"*40), data)
        self.assertEqual(pub.extract_report(archive([("metrics.json", json.dumps(data))])), data)

    def test_missing_or_wrong_scientific_identity_is_rejected(self):
        for change in ({"run_id": "124"}, {"source_sha": "b"*40},
                       {"device": "cuda"}, {"weights_sha256": "0"*64},
                       {"data_sha256": "0"*64}, {"experiment": "WSP-PILOT-001"},
                       {"context": "kaggle-gpu"}, {"holdout_count": 0}):
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    pub.validate_report({**valid_data(), **change}, "123", "a"*40)

    def test_invalid_aggregate_or_unverified_factorization_is_rejected(self):
        d = valid_data()
        for corrupt in (
            {"metrics": {**d["metrics"], "delta_rank": 3}},
            {"metrics": {**d["metrics"], "base": {"correct": 25, "total": 24, "unsafe_write": 0}}},
            {"factorization_error": 2e-6},
            {"factorization_error": float("nan")},
            {"duration_seconds": 5000},
            {"secret": "no secrets allowed"},
        ):
            with self.subTest(corrupt=repr(corrupt)[:50]):
                with self.assertRaises(ValueError):
                    pub.validate_report({**d, **corrupt}, "123", "a"*40)

    def test_zip_rejects_extra_files_and_path_traversal(self):
        for rows in (
            [("../metrics.json", "{}")],
            [("metrics.json", "{}"), ("private.log", "secret")],
            [("not-metrics.json", "{}")],
            [("metrics.json", "x" * 20000)],
        ):
            with self.subTest(files=[r[0] for r in rows]):
                with self.assertRaises(ValueError):
                    pub.extract_report(archive(rows))
        with self.assertRaises(ValueError):
            pub.extract_report(b"X" * (pub.MAX_ZIP + 1))
