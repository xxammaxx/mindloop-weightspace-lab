import unittest
import numpy as np
from weightspace_ci import pilot, dispatch


class PilotTests(unittest.TestCase):
    def test_rank_two_edit_reconstructs_known_residual(self):
        h = np.eye(3)
        target = np.array([[2., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 0., 0.]])
        a, b = pilot.factor_edit(h, target, ridge=0., rank=2)
        self.assertEqual(a.shape, (2, 3))
        self.assertEqual(b.shape, (4, 2))
        np.testing.assert_allclose(h @ a.T @ b.T, target, atol=1e-8)

    def test_placebo_matches_edit_norm(self):
        a, b = pilot.factor_edit(np.eye(3), np.eye(3, 4), ridge=0., rank=2)
        random_edit = pilot.placebo(a, b)
        self.assertAlmostEqual(np.linalg.norm(random_edit), np.linalg.norm(b @ a))
        self.assertLessEqual(np.linalg.matrix_rank(random_edit), 2)
        self.assertFalse(np.allclose(random_edit, b @ a))

    def test_holdout_disjoint_balanced(self):
        dev, holdout = pilot.DEV, pilot.HOLDOUT
        self.assertFalse(set(t for t, _ in dev) & set(t for t, _ in holdout))
        for rows in (dev, holdout):
            self.assertEqual([sum(y == c for _, y in rows) for c in range(4)], [6]*4)

    def test_safety_metrics_count_write_on_deny(self):
        m = pilot.metrics([0, 1, 2, 3], [0, 1, 1, 3])
        self.assertEqual(m, {'correct': 3, 'total': 4, 'unsafe_write': 1})

    def test_quota_gate_rejects_unknown_and_insufficient(self):
        for quota in ([], [{'resource':'GPU','remaining':'0.5h'}], [{'resource':'GPU','remaining':'unknown'}], [{'resource':'TPU','remaining':'30.00h'}], [{'resource':'GPU','remaining':'30.00h'}]*2):
            self.assertFalse(dispatch.quota_ok(quota))
        self.assertTrue(dispatch.quota_ok([{'resource':'GPU','remaining':'2.00h'}]))

    def test_status_must_be_for_exact_kernel(self):
        self.assertEqual(dispatch.parse_status('owner/abc has status "KernelWorkerStatus.COMPLETE"', 'owner/abc'), 'COMPLETE')
        self.assertIsNone(dispatch.parse_status('other/abc has status "KernelWorkerStatus.COMPLETE"', 'owner/abc'))

    def test_result_identity_rejects_stale_run(self):
        with self.assertRaises(ValueError):
            dispatch.validate_metrics({'run_id':'41'}, '42', 'a'*64)

    def test_push_requires_exact_success_and_kernel(self):
        self.assertIsNone(dispatch.push_version('Kernel push error: see previous output', 'owner/abc'))
        self.assertIsNone(dispatch.push_version('Kernel version 1 successfully pushed.  Please check progress at https://www.kaggle.com/code/other/abc', 'owner/abc'))
        self.assertEqual(dispatch.push_version('Kernel version 1 successfully pushed.  Please check progress at https://www.kaggle.com/code/owner/abc', 'owner/abc'), 1)

    def test_valid_metrics_and_wrong_data_hash(self):
        row = {'correct': 12, 'total': 24, 'unsafe_write': 0}
        data = {'schema':1,'experiment':'WSP-PILOT-001','run_id':'42','source_sha':'a'*40,
                'model':pilot.MODEL,'revision':pilot.REVISION,'weights_sha256':pilot.WEIGHT_SHA,
                'data_sha256':'e12eec12664a8241cf8208aa2dfd2b102ea9863c83cde3659fc076a2f952d1ac',
                'device':'Tesla T4','torch':'2.9.1+cu128','transformers':'5.12.0','seed':1701,
                'dev_count':24,'holdout_count':24,'metrics':{**{k:row for k in ('base','prompt','rank2','placebo')},'delta_rank':2},
                'factorization_error':1e-9,'duration_seconds':42,'peak_vram_bytes':2048}
        self.assertEqual(dispatch.validate_metrics(data,'42','a'*40),data)
        with self.assertRaises(ValueError):
            dispatch.validate_metrics({**data,'data_sha256':'f'*64},'42','a'*40)
