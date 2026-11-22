import unittest

from weightspace_ci import dispatch


class DiagnosticTests(unittest.TestCase):
    def test_identifies_model_download_error_without_raw_log(self):
        raw = 'secret=TOKEN_EXAMPLE\n  File "/kaggle/working/pilot.py", line 94, in run\nRepositoryNotFoundError: 401 Client Error for URL https://private.example/TOKEN_EXAMPLE'
        result = getattr(dispatch, 'summarize_kernel_log', lambda _: {})(raw)
        self.assertEqual(result, {'categories': ['model_access'], 'pilot_lines': [94]})

    def test_classifies_runtime_and_package_failures(self):
        raw = 'ERROR: No matching distribution found for transformers==5.12.0\nsubprocess.CalledProcessError: exit status 1\nModuleNotFoundError: No module named xyz\nRuntimeError: CUDA out of memory'
        result = getattr(dispatch, 'summarize_kernel_log', lambda _: {})(raw)
        self.assertEqual(result, {'categories': ['dependency_install', 'missing_module', 'out_of_memory'], 'pilot_lines': []})

    def test_unknown_output_does_not_publish_raw_text(self):
        result = getattr(dispatch, 'summarize_kernel_log', lambda _: {} )('username=private; api_token=xyz; arbitrary failure')
        self.assertEqual(result, {'categories': ['unknown'], 'pilot_lines': []})
