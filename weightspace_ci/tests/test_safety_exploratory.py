import unittest
import numpy as np

from weightspace_ci import pilot, safety_cases, safety_exploratory as study


class WspSafety002Tests(unittest.TestCase):
    def test_new_holdout_disjoint_from_all_previously_seen_cases(self):
        safety_cases.validate_splits()
        prior = {text for text, _ in (*pilot.DEV, *pilot.HOLDOUT)}
        self.assertFalse({text for text, _ in safety_cases.HOLDOUT} & prior)
        for split in (safety_cases.DEV, safety_cases.HOLDOUT):
            self.assertEqual([sum(y == c for _, y in split) for c in range(4)], [6, 6, 6, 6])

    def test_fixed_safety_margin_only_changes_nonwrite_targets(self):
        zeros = np.zeros((24, 4))
        normal, safe = study.targets(zeros, [y for _, y in safety_cases.DEV])
        for i, (_, label) in enumerate(safety_cases.DEV):
            if label == 1:
                np.testing.assert_array_equal(safe[i], normal[i])
            else:
                self.assertGreaterEqual(safe[i, label] - safe[i, 1], 8.0)
                np.testing.assert_array_equal(np.delete(safe[i], 1), np.delete(normal[i], 1))

    def test_five_arm_dev_only_fit_and_rank_two_gate(self):
        n = len(safety_cases.DEV)
        h = np.eye(n)
        logits = np.zeros((n, 4))
        result, errors, ranks, status = study.evaluate(h, logits, h, logits, logits)
        self.assertEqual(set(result),
                         {"base", "prompt", "rank2_unconstrained", "rank2_safe", "placebo"})
        self.assertEqual(set(ranks), {"rank2_unconstrained", "rank2_safe"})
        self.assertTrue(all(0 < rank <= 2 for rank in ranks.values()))
        self.assertTrue(all(value < 1e-6 for value in errors.values()))
        self.assertIn(status, {"LAB_CANDIDATE", "REJECTED_SAFETY", "REJECTED_UTILITY"})
        for arm in result.values():
            self.assertEqual(arm["total"], 24)
            self.assertTrue(0 <= arm["correct"] <= 24)
            self.assertTrue(0 <= arm["unsafe_write"] <= 18)

    def test_reject_nonfinite_inputs_or_wrong_holdout(self):
        h = np.eye(24)
        logits = np.zeros((24, 4))
        bad = logits.copy()
        bad[0, 2] = float("nan")
        with self.assertRaises(ValueError):
            study.evaluate(h, bad, h, logits, logits)
        with self.assertRaises(ValueError):
            study.evaluate(h, logits, h[:23], logits, logits)

    def test_rank_two_safety_target_validation(self):
        with self.assertRaises(ValueError):
            study.targets(np.zeros((23, 4)), [0]*23)
        with self.assertRaises(ValueError):
            study.targets(np.zeros((24, 4)), [4]*24)


if __name__ == "__main__":
    unittest.main()
