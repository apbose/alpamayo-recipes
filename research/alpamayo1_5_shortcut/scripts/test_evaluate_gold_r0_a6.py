#!/usr/bin/env python3
"""CPU tests; no model loading, training, or network access."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

import pandas as pd

from evaluate_gold_r0_a6 import (
    JOBS, METRICS, HF_REVISION, T0_US, audit_training, make_rows,
    report_text, validate_result, write_json, missing_steps, merge_completed,
)


class GoldTests(unittest.TestCase):
    def test_count_order_timestamp_and_chunk(self):
        ids = [f"clip-{i}" for i in range(644)]
        index = pd.DataFrame(dict(chunk=[2]*644, clip_is_valid=[True]*644), index=ids)
        rows = make_rows(ids, index)
        self.assertEqual([r['clip_id'] for r in rows], ids)
        self.assertTrue(all(r['t0_relative'] == T0_US and r['chunk'] == 2 for r in rows))
        with self.assertRaisesRegex(ValueError, '644'):
            make_rows(ids[:-1], index)
        with self.assertRaisesRegex(ValueError, '644'):
            make_rows([ids[0]]*644, index)
        with self.assertRaisesRegex(ValueError, 'absent'):
            make_rows(ids, index.iloc[1:])
        index.loc[ids[0], 'clip_is_valid'] = False
        with self.assertRaisesRegex(ValueError, 'invalid'):
            make_rows(ids, index)

    def test_overlap_fails_even_for_unused_manifest_clip(self):
        manifest = [dict(clip_id='a'), dict(clip_id='b')]
        protocol = dict(source_indices=[0, 0])
        with self.assertRaisesRegex(ValueError, 'overlap'):
            audit_training({'b'}, protocol, manifest, [[0, 1]])
        result = audit_training({'c'}, protocol, manifest, [[0, 1]])
        self.assertEqual(result['used_unique_clips'], 1)
        self.assertEqual(result['gold_manifest_overlap'], 0)

    def test_report_pending_and_separate_baselines(self):
        state = dict(status='evaluating')
        text, table = report_text({}, state, 'output')
        self.assertEqual(table, [])
        self.assertEqual(text.count('Pending'), 8)
        def result(a, b):
            return dict(results={str(s): dict(metrics={k:v for k in METRICS},
                        latency_ms=dict(action_expert_diffusion=dict(mean=20),
                                        end_to_end_model=dict(mean=1000))) for s,v in [(10,a),(2,b)]})
        text, table = report_text(dict(R0=result(2,3), A6=result(1,1.2)), state, 'output')
        self.assertIn('3.0000 (+50.00%)', text)
        self.assertIn('1.2000 (+20.00%)', text)
        self.assertEqual(len(table), 4)

    @patch('evaluate_gold_r0_a6.digest', return_value='hash')
    def test_validator_rejects_missing_and_mispaired_clips(self, _digest):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'val.json'
            rows = [dict(clip_id='a', t0_relative=T0_US), dict(clip_id='b', t0_relative=T0_US)]
            write_json(path, rows)
            cfg = dict(validation_samples=2, validation_unique_clips=2,
                       validation_manifest_sha256='hash', num_traj_samples=6,
                       seed_reset_for_each_step_count=42, attention_backend='eager',
                       warmup_samples_per_step_count=1, access_mode='hf_stream', hf_revision=HF_REVISION,
                       recipe_config=JOBS['A6']['recipe'], checkpoint_config_sha256='hash',
                       zero_step_size_adapter=False, step_size_adapter_scale=1.0,
                       inference_steps=[10], checkpoint=str(JOBS['A6']['checkpoint']),
                       evaluation_split='val', shortcut_runtime=dict(updates=249, inference_weights='ema'))
            samples=[dict(sample_index=i, **row, metrics={m:[1.0] for m in METRICS})
                     for i,row in enumerate(rows)]
            result=dict(configuration=cfg, results={'10':dict(per_sample=samples, metrics={m:1.0 for m in METRICS})})
            validate_result(result, 'A6', path, (10,))
            bad=copy.deepcopy(result)
            bad['results']['10']['per_sample'].pop()
            with self.assertRaisesRegex(ValueError,'Incomplete'):
                validate_result(bad, 'A6', path, (10,))
            bad=copy.deepcopy(result)
            bad['results']['10']['per_sample'][0]['clip_id']='b'
            with self.assertRaisesRegex(ValueError,'pairing'):
                validate_result(bad, 'A6', path, (10,))
            bad=copy.deepcopy(result)
            bad['results']['10']['metrics']['min_ade']=2
            with self.assertRaisesRegex(ValueError,'Aggregate'):
                validate_result(bad, 'A6', path, (10,))
            bad=copy.deepcopy(result)
            bad['configuration']['shortcut_runtime']['inference_weights']='student'
            with self.assertRaisesRegex(ValueError,'EMA'):
                validate_result(bad, 'A6', path, (10,))


class BenchmarkConfigTests(unittest.TestCase):
    def test_hf_override_preserves_model_and_default_local(self):
        from benchmark_inference_steps import build_config
        args=SimpleNamespace(eval_split='val',config_name='sft_stage2_trajectory_shortcut_paper_ema',
                             manifest_dir=Path('/tmp/manifests'),checkpoint=Path('/tmp/checkpoint'),
                             attention_backend='eager',dataset=Path('/tmp/dataset'),output_dir=Path('/tmp/out'),
                             shortcut_inference_weights='ema',access_mode='hf_stream',
                             hf_revision=HF_REVISION,hf_cache_dir=Path('/tmp/hf'))
        summary=dict(chunks=dict(train=[],val=[1,2]))
        hf=build_config(args,summary)
        self.assertEqual(hf.data.val_dataset.access_mode,'hf_stream')
        self.assertEqual(hf.data.val_dataset.hf_revision,HF_REVISION)
        self.assertEqual(hf.model.shortcut_inference_weights,'ema')
        args.access_mode='local'
        local=build_config(args,summary)
        self.assertNotIn('access_mode',local.data.val_dataset)
        self.assertEqual(hf.model,local.model)
        args.access_mode='hf_stream'
        args.hf_revision=None
        with self.assertRaisesRegex(ValueError,'requires'):
            build_config(args,summary)


class RecoveryTests(unittest.TestCase):
    def test_missing_counts_preserve_completed(self):
        self.assertEqual(missing_steps({'results': {'10': {}, '5': {}}}), (4, 2))
        self.assertEqual(missing_steps({'results': {str(s): {} for s in [10,5,4,2]}}), ())

    @patch('evaluate_gold_r0_a6.digest', return_value='new-hash')
    def test_merge_does_not_mutate_or_pool_cross_run_timing(self, _digest):
        old = dict(configuration=dict(inference_steps=[10,5,4,2], seed=42, validation_manifest='old'),
                   results={'10': {'metrics': {'min_ade': 1}}, '5': {'metrics': {'min_ade': 2}}},
                   step_sources={s: dict(path='old.json',sha256='old-hash') for s in ['10','5']})
        new = dict(configuration=dict(inference_steps=[4,2], seed=42, validation_manifest='new',
                                      hf_stream_retry={'max_attempts':6}),
                   results={'4': {'metrics': {'min_ade': 3}, 'comparison_to_10_step': {'bad':1}}})
        before = copy.deepcopy(old)
        merged = merge_completed(old,new,Path('new.json'))
        self.assertEqual(old,before)
        self.assertEqual(merged['results']['10'],old['results']['10'])
        self.assertEqual(merged['configuration']['inference_steps'],[10,5,4])
        self.assertIsNone(merged['results']['4']['comparison_to_10_step'])
        self.assertEqual(merged['step_sources']['4']['path'],'new.json')
        self.assertTrue(merged['merged_across_runs'])
        new['configuration']['seed']=43
        with self.assertRaisesRegex(ValueError,'configuration'):
            merge_completed(old,new,Path('new.json'))
        new['configuration']['seed']=42
        new['results']={'10': {}}
        with self.assertRaisesRegex(ValueError,'overwrite'):
            merge_completed(old,new,Path('new.json'))


class StreamingRetryTests(unittest.TestCase):
    def run_request(self, codes, attempts):
        import requests
        from huggingface_hub import hf_file_system
        from huggingface_hub.utils import _http
        from hf_stream_retry import configure_hf_stream_retries
        responses=[]
        for code in codes:
            response=requests.Response()
            response.status_code=code
            response.url='https://example.invalid/fixture.zip'
            response._content=b'data'
            responses.append(response)
        session=Mock()
        session.request.side_effect=responses
        with patch.object(hf_file_system,'http_backoff',_http.http_backoff), \
             patch.object(_http,'get_session',return_value=session), \
             patch.object(_http.time,'sleep') as sleep:
            policy=configure_hf_stream_retries(attempts)
            configure_hf_stream_retries(attempts)  # must not nest the wrappers
            try:
                result=hf_file_system.http_backoff('GET','https://example.invalid/fixture.zip',
                                                  headers={'Range':'bytes=0-3'},timeout=60)
            except requests.HTTPError:
                return None,session,sleep,policy
        return result,session,sleep,policy

    def test_499_recovers_and_keeps_range_and_timeout(self):
        result,session,sleep,policy=self.run_request([499,206],6)
        self.assertEqual(result.status_code,206)
        self.assertEqual(session.request.call_count,2)
        self.assertEqual(session.request.call_args.kwargs['headers'],{'Range':'bytes=0-3'})
        self.assertEqual(session.request.call_args.kwargs['timeout'],60)
        sleep.assert_called_once_with(2)
        self.assertEqual(policy['max_attempts'],6)

    def test_exhaustion_is_bounded_and_raises(self):
        result,session,sleep,_=self.run_request([499]*6,6)
        self.assertIsNone(result)
        self.assertEqual(session.request.call_count,6)
        self.assertEqual([c.args[0] for c in sleep.call_args_list],[2,4,8,16,30])

    def test_auth_failure_not_retried(self):
        result,session,sleep,_=self.run_request([401],6)
        self.assertEqual(result.status_code,401)  # outer hf_raise_for_status raises
        self.assertEqual(session.request.call_count,1)
        sleep.assert_not_called()

    def test_invalid_budget(self):
        from hf_stream_retry import configure_hf_stream_retries
        with self.assertRaises(ValueError):
            configure_hf_stream_retries(0)


if __name__ == '__main__':
    unittest.main()
