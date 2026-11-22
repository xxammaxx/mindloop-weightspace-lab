import unittest

from weightspace_ci.bootstrap import evaluate, select_next


class BootstrapTests(unittest.TestCase):
    def test_select_next_respects_completed_experiments(self):
        self.assertEqual(select_next({"BASELINE": "PASS"}, gpu_ready=False), "PROMPT_CONTROL")

    def test_gpu_requires_explicit_readiness(self):
        done = {"BASELINE": "PASS", "PROMPT_CONTROL": "PASS", "PATCH_REPRODUCTION": "PASS"}
        self.assertIsNone(select_next(done, gpu_ready=False))
        self.assertEqual(select_next(done, gpu_ready=True), "GPU_REPLICATION")

    def test_failed_or_running_experiment_stops_cycle(self):
        for status in ("FAIL", "BLOCKED", "RUNNING"):
            with self.subTest(status=status):
                self.assertIsNone(select_next({"BASELINE": status}, gpu_ready=True))

    def test_correct_tool_order_and_arguments(self):
        expected = [{"name": "read", "arguments": {"path": "note.txt"}},
                    {"name": "write", "arguments": {"path": "note.txt", "text": "new"}}]
        self.assertTrue(evaluate(expected, expected.copy()))
        self.assertFalse(evaluate(expected, list(reversed(expected))))
        self.assertFalse(evaluate(expected, [{"name": "read", "arguments": {"path": "other.txt"}}]))
        self.assertFalse(evaluate(expected, expected + [{"name": "shell", "arguments": {}}]))

    def test_completed_cycle_stops(self):
        self.assertIsNone(select_next({"BASELINE": "PASS", "PROMPT_CONTROL": "PASS", "PATCH_REPRODUCTION": "PASS", "GPU_REPLICATION": "PASS"}, gpu_ready=True))

    def test_unknown_or_inconsistent_states_stop(self):
        self.assertIsNone(select_next({"PROMPT_CONTROL": "PASS"}, gpu_ready=True))
        self.assertIsNone(select_next({"UNKNOWN": "PASS"}, gpu_ready=True))

    def test_start_from_empty_state(self):
        self.assertEqual(select_next({}, gpu_ready=False), "BASELINE")


if __name__ == "__main__":
    unittest.main()
