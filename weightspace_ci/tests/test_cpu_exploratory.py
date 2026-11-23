import os
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from weightspace_ci import cpu_exploratory as cpu
from weightspace_ci import pilot


class CpuExploratoryTests(TestCase):
    def test_four_arms_use_frozen_original_split_and_rank_two(self):
        # Synthetic observations only; no model download in host tests.
        n = len(pilot.DEV)
        h = np.eye(n)
        base = np.zeros((n, 4))
        prompt = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
        measured, error = cpu.score_tables(h, base, h, base, prompt)
        self.assertEqual(set(measured), {"base", "prompt", "rank2", "placebo", "delta_rank"})
        self.assertGreater(measured["delta_rank"], 0)
        self.assertLessEqual(measured["delta_rank"], 2)
        self.assertLessEqual(error, 1e-6)
        for key in ("base", "prompt", "rank2", "placebo"):
            self.assertEqual(measured[key]["total"], 24)
            self.assertTrue(0 <= measured[key]["correct"] <= 24)
            self.assertTrue(0 <= measured[key]["unsafe_write"] <= 18)

    def test_wrong_shape_and_nonfinite_values_are_rejected(self):
        n = len(pilot.DEV)
        h = np.eye(n)
        logits = np.zeros((n, 4))
        with self.assertRaises(ValueError):
            cpu.score_tables(h, logits, h[:23], logits, logits)
        corrupted = logits.copy()
        corrupted[1, 2] = float("nan")
        with self.assertRaises(ValueError):
            cpu.score_tables(h, corrupted, h, logits, logits)

    def test_split_identity_matches_preregistered_pilot(self):
        import hashlib
        import json
        measured = hashlib.sha256(json.dumps(
            [pilot.DEV, pilot.HOLDOUT], ensure_ascii=False).encode()).hexdigest()
        self.assertEqual(measured, cpu.EXPECTED_DATA_SHA)
        self.assertEqual(len(pilot.DEV), 24)
        self.assertEqual(len(pilot.HOLDOUT), 24)

    def test_only_trusted_first_attempt_main_run(self):
        good = {"GITHUB_REPOSITORY": "xxammaxx/mindloop-weightspace-lab",
                "GITHUB_REF": "refs/heads/main", "GITHUB_RUN_ATTEMPT": "1",
                "GITHUB_RUN_ID": "456", "GITHUB_SHA": "a"*40}
        with patch.dict(os.environ, good, clear=True):
            cpu.assert_trusted_context()
        for change in ({"GITHUB_REF": "refs/heads/test"},
                       {"GITHUB_RUN_ATTEMPT": "2"},
                       {"GITHUB_REPOSITORY": "attacker/repo"},
                       {"GITHUB_SHA": "not-a-real-sha"}):
            with patch.dict(os.environ, {**good, **change}, clear=True):
                with self.assertRaises(RuntimeError):
                    cpu.assert_trusted_context()
