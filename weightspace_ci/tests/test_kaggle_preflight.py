import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


class KagglePreflightTests(unittest.TestCase):
    def run_preflight(self, error='', token='fake-secret-never-print'):
        workflow = Path('.github/workflows/kaggle-preflight.yml').read_text()
        script = textwrap.dedent(workflow.split('      - name: Check authenticated accelerator quota', 1)[1].split('        run: |\n', 1)[1])
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = Path(tmp) / 'kaggle-cli' / 'bin'
            bin_dir.mkdir(parents=True)
            cli = bin_dir / 'kaggle'
            cli.write_text('#!/bin/bash\n[ "$*" = "quota --format json" ] || exit 3\nprintf "%s\\n" "$FAKE_ERROR $KAGGLE_API_TOKEN" >&2\nexit 1\n')
            cli.chmod(0o700)
            (bin_dir / 'python').symlink_to(sys.executable)
            return subprocess.run(['bash', '-c', script], env={**os.environ, 'RUNNER_TEMP': tmp, 'KAGGLE_API_TOKEN': token, 'FAKE_ERROR': error}, capture_output=True, text=True)

    def test_auth_failure_classified_without_leaking_token(self):
        result = self.run_preflight('HTTP 401 Unauthorized')
        self.assertEqual(result.returncode, 1)
        self.assertIn('AUTH_REJECTED', result.stdout)
        self.assertNotIn('fake-secret-never-print', result.stdout + result.stderr)

    def test_permission_failure_classified(self):
        result = self.run_preflight('HTTP 403 Forbidden')
        self.assertEqual(result.returncode, 1)
        self.assertIn('ACCESS_DENIED', result.stdout)

    def test_unknown_failure_does_not_echo_raw_error(self):
        result = self.run_preflight('private raw diagnostic')
        self.assertEqual(result.returncode, 1)
        self.assertIn('UNCLASSIFIED', result.stdout)
        self.assertNotIn('private raw diagnostic', result.stdout + result.stderr)

    def test_cli_argument_failure_classified(self):
        result = self.run_preflight('error: unrecognized arguments: --format json')
        self.assertEqual(result.returncode, 1)
        self.assertIn('CLI_ARGUMENT_ERROR', result.stdout)

    def test_connection_failure_classified(self):
        result = self.run_preflight('ConnectionError: Max retries exceeded')
        self.assertEqual(result.returncode, 1)
        self.assertIn('CONNECTION_ERROR', result.stdout)

    def test_sdk_duration_failure_classified(self):
        result = self.run_preflight('ValueError: not enough values to unpack (expected 2, got 1)')
        self.assertEqual(result.returncode, 1)
        self.assertIn('SDK_PARSE_ERROR', result.stdout)

    def test_missing_token_stops_before_call(self):
        result = self.run_preflight('HTTP 401 Unauthorized', token='')
        self.assertEqual(result.returncode, 1)
        self.assertIn('missing in the environment', result.stdout)
        self.assertNotIn('AUTH_REJECTED', result.stdout)
