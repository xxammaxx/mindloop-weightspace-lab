import hashlib
import json
import subprocess
from unittest.mock import patch
from pathlib import Path
import tempfile
import unittest

from weightspace_ci import dispatch, pilot


class OfflineTests(unittest.TestCase):
    def test_verified_bundle_rejects_changed_missing_and_extra_files(self):
        from weightspace_ci.offline import validate_bundle, MANIFEST_NAME
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'config.json').write_bytes(b'{}')
            (root/'example.whl').write_bytes(b'wheel')
            files = {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                     for name in ('config.json','example.whl')}
            manifest = json.dumps({'schema':1,'files':files},sort_keys=True).encode()
            (root/MANIFEST_NAME).write_bytes(manifest)
            digest = hashlib.sha256(manifest).hexdigest()
            self.assertEqual(validate_bundle(root,digest), files)
            (root/'config.json').write_bytes(b'changed')
            with self.assertRaises(ValueError): validate_bundle(root,digest)
            (root/'config.json').write_bytes(b'{}')
            (root/'example.whl').unlink()
            with self.assertRaises(ValueError): validate_bundle(root,digest)
            (root/'example.whl').write_bytes(b'wheel')
            (root/'unlisted.py').touch()
            with self.assertRaises(ValueError): validate_bundle(root,digest)

    def test_offline_pip_arguments_have_no_network_fallback(self):
        from weightspace_ci.offline import install_argv
        args = install_argv('/input/bundle','/tmp/deps')
        for key in ('--isolated','--no-index','--only-binary=:all:','--target','--find-links','transformers==5.12.0','numpy==2.2.6'):
            self.assertIn(key,args)
        self.assertEqual(args[args.index('--find-links')+1],'/input/bundle')

    def test_dataset_create_and_version_fail_closed(self):
        self.assertTrue(dispatch.private_dataset_created('Your private Dataset is being created. Please check progress at https://www.kaggle.com/datasets/user/data','user/data'))
        self.assertFalse(dispatch.private_dataset_created('Dataset creation error: rejected','user/data'))
        self.assertFalse(dispatch.private_dataset_created('Your public Dataset is being created. Please check progress at https://www.kaggle.com/datasets/user/data','user/data'))
        self.assertFalse(dispatch.private_dataset_created('Your private Dataset is being created. Please check progress at https://www.kaggle.com/datasets/other/data','user/data'))

    def test_private_dataset_creation_keeps_cli_success_receipt(self):
        args = dispatch.dataset_create_args('/tmp/offline-bundle')
        self.assertEqual(args, ('datasets', 'create', '-p', '/tmp/offline-bundle', '--keep-tabular'))
        self.assertNotIn('--public', args)
        self.assertNotIn('--quiet', args)
        self.assertNotIn('--silent', args)

    def test_cli_failure_categories_are_allowlisted_and_do_not_leak_secrets(self):
        cases = [
            ('403 Forbidden token=not-for-public', 'ACCESS_DENIED'),
            ('401 Unauthorized key=not-for-public', 'AUTH_REJECTED'),
            ('400 Bad Request: invalid dataset-metadata.json', 'DATASET_METADATA'),
            ('409 Conflict: dataset already exists', 'ALREADY_EXISTS'),
            ('413 Request Entity Too Large: file too large', 'INPUT_TOO_LARGE'),
            ('429 Too Many Requests, quota exceeded', 'QUOTA_LIMIT'),
            ('Temporary failure in name resolution', 'NETWORK_DNS'),
            ('503 Service Unavailable', 'SERVER_ERROR'),
            ('unrecognized private-account-secret', 'UNKNOWN'),
        ]
        for raw, expected in cases:
            with self.subTest(expected=expected):
                with patch('weightspace_ci.dispatch.subprocess.run') as run:
                    run.return_value = subprocess.CompletedProcess(
                        args=['kaggle'], returncode=1, stdout='secret-username=not-for-public',
                        stderr=raw)
                    with self.assertRaises(RuntimeError) as error:
                        dispatch.run_cli('datasets', 'create', '-p', '/tmp/private')
                message = str(error.exception)
                self.assertIn('Kaggle CLI command failed: datasets/create', message)
                self.assertIn(expected, message)
                self.assertNotIn('not-for-public', message)
                self.assertNotIn('secret-username', message)
                self.assertNotIn('unrecognized private-account-secret', message)
                self.assertNotIn('/tmp/private', message)
                self.assertEqual(run.call_args.kwargs['capture_output'], True)
                self.assertEqual(run.call_args.kwargs['text'], True)

    def test_cli_success_result_not_suppressed(self):
        receipt = 'Your private Dataset is being created. Please check progress at https://www.kaggle.com/datasets/example/wsp-offline-42'
        with patch('weightspace_ci.dispatch.subprocess.run') as run:
            run.return_value = subprocess.CompletedProcess(args=['kaggle'], returncode=0,
                                                           stdout=receipt, stderr='')
            value = dispatch.run_cli(*dispatch.dataset_create_args('/tmp/bundle'))
        self.assertEqual(value, receipt)
        self.assertTrue(dispatch.private_dataset_created(value, 'example/wsp-offline-42'))

    def test_qualification_requires_exact_identity(self):
        data = {'schema':1,'experiment':'WSP-PILOT-001','run_id':'42','source_sha':'a'*40,
                'bundle_sha256':'b'*64,'model':pilot.MODEL,'revision':pilot.REVISION,
                'weights_sha256':pilot.WEIGHT_SHA,'transformers':'5.12.0','device':'cpu','torch':'2.9.1',
                'forward_pass':True,'labels':4,'hidden_size':1024}
        self.assertEqual(dispatch.validate_qualification(data,'42','a'*40,'b'*64),data)
        for change in ({'run_id':'41'},{'bundle_sha256':'c'*64},{'forward_pass':False},{'device':'cuda'},{'hidden_size':0}):
            with self.assertRaises(ValueError): dispatch.validate_qualification({**data,**change},'42','a'*40,'b'*64)

    def test_only_exact_previous_dns_failure_can_be_resolved(self):
        runs=[{'id':42,'status':'in_progress','conclusion':None,'run_attempt':1},
              {'id':45041540858,'status':'completed','conclusion':'failure','run_attempt':1}]
        self.assertFalse(dispatch.budget_for_runs(runs,2,'42',False))
        self.assertTrue(dispatch.budget_for_runs(runs,2,'42',True))
        self.assertFalse(dispatch.budget_for_runs(runs+[{'id':43,'status':'completed','conclusion':'failure','run_attempt':1}],3,'42',True))
        self.assertFalse(dispatch.budget_for_runs(runs,3,'42',True))
        self.assertFalse(dispatch.budget_for_runs([{**runs[0],'run_attempt':2},runs[1]],2,'42',True))
