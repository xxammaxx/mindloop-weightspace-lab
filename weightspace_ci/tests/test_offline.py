import hashlib
import json
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
