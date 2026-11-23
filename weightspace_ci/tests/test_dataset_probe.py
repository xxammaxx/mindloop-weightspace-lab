"""Regression tests for safe one-time Kaggle transport diagnosis."""
import json
import os
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from weightspace_ci import diagnose_dataset as probe


class DatasetProbeTests(TestCase):
    def test_known_existing_dataset_does_not_upload(self):
        with patch.object(probe.dispatch, "kaggle_owner", return_value="example"), \
             patch.object(probe, "state", return_value="ready") as state, \
             patch.object(probe.dispatch, "run_cli") as run_cli:
            self.assertEqual(probe.diagnose("123"), "ready")
            state.assert_called_once_with("example/wsp-offline-45051693771")
            run_cli.assert_not_called()

    def test_non_not_found_error_never_uploads(self):
        for failure in ("ACCESS_DENIED", "AUTH_REJECTED", "UNKNOWN"):
            with self.subTest(failure=failure), \
                 patch.object(probe.dispatch, "kaggle_owner", return_value="example"), \
                 patch.object(probe, "state", side_effect=RuntimeError(
                     "Kaggle CLI command failed: datasets/status; diagnosis=" + failure)), \
                 patch.object(probe.dispatch, "run_cli") as run_cli:
                with self.assertRaisesRegex(RuntimeError, "lookup blocked"):
                    probe.diagnose("123")
                run_cli.assert_not_called()

    def test_only_absent_dataset_allows_one_small_private_probe(self):
        reference = "example/wsp-upload-probe-123"
        receipt = ("Your private Dataset is being created. Please check progress at "
                   "https://www.kaggle.com/datasets/" + reference)
        def fake_cli(*args, **kwargs):
            self.assertEqual(args[0:2], ("datasets", "create"))
            self.assertNotIn("--quiet", args)
            self.assertNotIn("--public", args)
            folder = Path(args[args.index("-p") + 1])
            self.assertEqual(json.loads((folder / "dataset-metadata.json").read_text())["id"], reference)
            self.assertEqual(sorted(x.name for x in folder.iterdir()),
                             ["dataset-metadata.json", "probe.txt"])
            self.assertIn("no model or personal data", (folder / "probe.txt").read_text().lower())
            return receipt
        with patch.object(probe.dispatch, "kaggle_owner", return_value="example"), \
             patch.object(probe, "state", side_effect=[
                 RuntimeError("Kaggle CLI command failed: datasets/status; diagnosis=NOT_FOUND"),
                 "ready"]) as status, \
             patch.object(probe.dispatch, "run_cli", side_effect=fake_cli) as cli:
            self.assertEqual(probe.diagnose("123"), "ready")
            self.assertEqual(cli.call_count, 1)
            self.assertEqual(status.call_count, 2)
            status.assert_any_call(reference)

    def test_missing_receipt_does_not_repeat_upload(self):
        with patch.object(probe.dispatch, "kaggle_owner", return_value="example"), \
             patch.object(probe, "state", side_effect=RuntimeError(
                 "Kaggle CLI command failed: datasets/status; diagnosis=NOT_FOUND")) as status, \
             patch.object(probe.dispatch, "run_cli", return_value="") as cli:
            with self.assertRaisesRegex(RuntimeError, "receipt missing"):
                probe.diagnose("123")
            self.assertEqual(cli.call_count, 1)
            self.assertEqual(status.call_count, 1)

    def test_retries_and_untrusted_repos_never_touch_kaggle(self):
        contexts = [
            {"GITHUB_REPOSITORY": "unknown/repo", "GITHUB_REF": "refs/heads/main",
             "GITHUB_RUN_ATTEMPT": "1", "KAGGLE_API_TOKEN": "fake", "GITHUB_RUN_ID": "123"},
            {"GITHUB_REPOSITORY": "xxammaxx/mindloop-weightspace-lab", "GITHUB_REF": "refs/heads/main",
             "GITHUB_RUN_ATTEMPT": "2", "KAGGLE_API_TOKEN": "fake", "GITHUB_RUN_ID": "123"},
        ]
        for env in contexts:
            with self.subTest(env=env["GITHUB_RUN_ATTEMPT"]), \
                 patch.dict(os.environ, env, clear=True), \
                 patch.object(probe, "diagnose") as func:
                with self.assertRaisesRegex(RuntimeError, "Untrusted context"):
                    probe.main()
                func.assert_not_called()
