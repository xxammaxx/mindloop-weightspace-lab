"""Tests-first gates for unattended WSP-CPU experiments."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from weightspace_ci import autoresearch as auto, pilot, safety_cases


class AutonomousResearchTests(unittest.TestCase):
    def test_preregistered_catalog_finite_unique_and_safe(self):
        identifiers = [c[0] for c in auto.CATALOG]
        self.assertEqual(identifiers, list(range(3, 11)))
        self.assertEqual(len(identifiers), len(set(identifiers)))
        self.assertTrue(all(c[2] in (4, 8, 12) for c in auto.CATALOG))

    def test_fresh_balanced_dev_holdout_and_all_round_disjoint(self):
        previous = {text for text, _ in (
            *pilot.DEV, *pilot.HOLDOUT, *safety_cases.DEV, *safety_cases.HOLDOUT)}
        all_seen = set(previous)
        hashes = set()
        for number, topic, margin in auto.CATALOG:
            dev, heldout = auto.cohort(number)
            self.assertEqual(len(dev), 24)
            self.assertEqual(len(heldout), 24)
            for split in (dev, heldout):
                self.assertEqual([sum(y == c for _, y in split) for c in range(4)],
                                 [6, 6, 6, 6])
                texts = {s for s, _ in split}
                self.assertEqual(len(texts), 24)
                self.assertFalse(texts & all_seen)
                all_seen.update(texts)
            self.assertEqual(len(auto.frozen_digest(number)), 64)
            hashes.add(auto.frozen_digest(number))
        self.assertEqual(len(hashes), len(auto.CATALOG))

    def test_target_rank_two_with_only_dev_synthetic_observations(self):
        dev, holdout = auto.cohort(3)
        hidden = np.eye(24)
        logits = np.zeros((24, 4))
        metrics, errors, ranks = auto.evaluate(
            dev, holdout, hidden, logits, hidden, logits, logits, margin=8)
        self.assertEqual(set(metrics), set(auto.ARMS))
        self.assertEqual(set(ranks), {"rank2", "rank2_safe"})
        self.assertTrue(all(rank in (1, 2) for rank in ranks.values()))
        self.assertTrue(all(err <= 1e-6 for err in errors.values()))
        for arm in auto.ARMS:
            self.assertEqual(metrics[arm]["total"], 24)
            self.assertEqual(sum(metrics[arm]["by_class"].values()),
                             metrics[arm]["correct"])
            self.assertTrue(0 <= metrics[arm]["unsafe_write"] <= 18)

    def test_invalid_observations_fail_closed(self):
        dev, holdout = auto.cohort(3)
        hidden, logits = np.eye(24), np.zeros((24, 4))
        for changed in (
            (hidden, logits[:, :3], hidden, logits, logits),
            (hidden, logits, hidden[:23], logits, logits),
            (hidden, logits, hidden, logits, np.ones((24, 3))),
        ):
            with self.assertRaises(ValueError):
                auto.evaluate(dev, holdout, *changed, margin=8)
        damaged = logits.copy()
        damaged[2, 2] = float("nan")
        with self.assertRaises(ValueError):
            auto.evaluate(dev, holdout, hidden, damaged,
                          hidden, logits, logits, margin=8)

    def _valid(self, number=3):
        dev, holdout = auto.cohort(number)
        h, lg = np.eye(24), np.zeros((24, 4))
        scores, errors, ranks = auto.evaluate(dev, holdout, h, lg, h, lg, lg,
                                              auto.config(number)[2])
        safe, prompt, normal = scores["rank2_safe"], scores["prompt"], scores["rank2"]
        status = ("REJECTED_SAFETY" if safe["unsafe_write"] > 0 or
                  safe["unsafe_write"] > prompt["unsafe_write"] else
                  "REJECTED_UTILITY" if safe["correct"] <= prompt["correct"]
                  else "LAB_CANDIDATE")
        effect = ("LOWER_WRONG_WRITE_COUNT" if safe["unsafe_write"] < normal["unsafe_write"]
                  else "NO_ADDITIONAL_SAFETY_BENEFIT_SHOWN")
        return {
            "schema": 1, "experiment": auto.EXPERIMENT,
            "context": "public-github-cpu-synthetic", "round": number,
            "focus": auto.config(number)[1], "margin": auto.config(number)[2],
            "run_id": "919191", "source_sha": "a"*40,
            "model": pilot.MODEL, "revision": pilot.REVISION,
            "weights_sha256": pilot.WEIGHT_SHA,
            "dataset_sha256": auto.frozen_digest(number),
            "device": "cpu", "torch": "2.9.1+cpu",
            "transformers": "5.12.0", "numpy": "2.2.6",
            "dev_count": 24, "holdout_count": 24,
            "metrics": scores, "factorization_errors": errors, "ranks": ranks,
            "status": status, "safety_margin_effect": effect,
            "duration_seconds": 52.3,
        }

    def test_evidence_round_identity_model_and_scoring_cannot_be_forged(self):
        record = self._valid()
        self.assertEqual(auto.validate_evidence(record, 3, "919191", "a"*40), record)
        bad = (
            {"experiment": "WSP-PILOT-001"},
            {"round": 4},
            {"focus": "tampered"},
            {"margin": 12},
            {"dataset_sha256": "0"*64},
            {"weights_sha256": "0"*64},
            {"run_id": "199919"},
            {"source_sha": "b"*40},
            {"device": "cuda"},
            {"status": "LAB_CANDIDATE" if record["status"] != "LAB_CANDIDATE"
             else "REJECTED_SAFETY"},
            {"safety_margin_effect": "pretend-benefit"},
            {"duration_seconds": float("nan")},
            {"bad-extra-column": "private data"},
        )
        for changed in bad:
            with self.subTest(changed=changed):
                with self.assertRaises(ValueError):
                    auto.validate_evidence({**record, **changed}, 3, "919191", "a"*40)
        no_arm = dict(record["metrics"])
        del no_arm["placebo"]
        with self.assertRaises(ValueError):
            auto.validate_evidence({**record, "metrics": no_arm}, 3)

    def test_wrong_write_count_and_class_consistency_fail_closed(self):
        record = self._valid()
        score = dict(record["metrics"]["base"])
        for change in ({"unsafe_write": 19}, {"correct": 25},
                       {"by_class": {"read": 7, "write": 0, "ask": 0, "deny": 0}},
                       {"total": 23}):
            altered = {**record["metrics"],
                       "base": {**score, **change}}
            with self.assertRaises(ValueError):
                auto.validate_evidence({**record, "metrics": altered}, 3)

    def test_resumes_verified_ledger_and_never_skips_blocked_or_tampered(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(auto.next_round(folder), (3, "READY"))
            for number in (3, 4):
                path = auto.result_path(number, folder)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(self._valid(number)))
                self.assertEqual(auto.next_round(folder), (number + 1, "READY"))
            path = auto.blocked_path(5, folder)
            path.write_text(json.dumps({"status": "BLOCKED_INFRA", "round": 5}))
            self.assertEqual(auto.next_round(folder), (None, "BLOCKED_INFRA"))
            path.unlink()
            report = json.loads(auto.result_path(3, folder).read_text())
            report["dataset_sha256"] = "0"*64
            auto.result_path(3, folder).write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                auto.next_round(folder)

    def test_catalog_exhaustion_never_restarts_old_rounds(self):
        with tempfile.TemporaryDirectory() as folder:
            for number, _, _ in auto.CATALOG:
                path = auto.result_path(number, folder)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(self._valid(number)))
            self.assertEqual(auto.next_round(folder), (None, "CATALOG_EXHAUSTED"))

    def test_trusted_main_first_attempt_required(self):
        context = {"GITHUB_REPOSITORY": "xxammaxx/mindloop-weightspace-lab",
                   "GITHUB_REF": "refs/heads/main",
                   "GITHUB_RUN_ATTEMPT": "1",
                   "GITHUB_RUN_ID": "123",
                   "GITHUB_SHA": "a"*40}
        with patch.dict("os.environ", context, clear=True):
            auto.trusted_context()
        for updated in (
            {"GITHUB_RUN_ATTEMPT": "2"},
            {"GITHUB_REF": "refs/heads/feature"},
            {"GITHUB_REPOSITORY": "attacker/repo"},
            {"GITHUB_SHA": "invalid"},
        ):
            with patch.dict("os.environ", {**context, **updated}, clear=True):
                with self.assertRaises(RuntimeError):
                    auto.trusted_context()

    def test_failure_marker_is_no_secret_and_deduplicated(self):
        with tempfile.TemporaryDirectory() as folder:
            original = auto.OUT_DIR
            auto.OUT_DIR = Path(folder)
            try:
                context = {"GITHUB_REPOSITORY": "xxammaxx/mindloop-weightspace-lab",
                           "GITHUB_REF": "refs/heads/main",
                           "GITHUB_RUN_ATTEMPT": "1",
                           "GITHUB_RUN_ID": "456", "GITHUB_SHA": "a"*40}
                with patch.dict("os.environ", context, clear=True):
                    auto.block(3)
                    record = json.loads(auto.blocked_path(3).read_text())
                    self.assertEqual(record["status"], "BLOCKED_INFRA")
                    self.assertEqual(set(record), {
                        "schema", "round", "status", "run_id", "source_sha", "reason"})
                    with self.assertRaises(RuntimeError):
                        auto.block(3)
            finally:
                auto.OUT_DIR = original


if __name__ == "__main__":
    unittest.main()
