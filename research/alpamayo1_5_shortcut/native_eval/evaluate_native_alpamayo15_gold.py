# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
"""Released 10B gold-644 evaluation using unmodified NVIDIA native inference.

Matches recipes/alpamayo1_5_quant/eval.py quality settings: FP16, seed 42
reset per clip, six independently generated reasoning/trajectory candidates,
four labelled cameras, four labelled frames each, t0=5.1s, 64 future poses.
No training, navigation annotation, human reasoning, or future target enters
the model. Native source is pinned by the launcher. Operational additions:
strict weight/token audits, pinned HF streaming, ordered IO prefetch, durable
per-clip results, resume, synchronized timing, and explicit failure reporting.

The old 1.4646 result used a different wrapper, prompt, precision, backend,
and seed policy. The final paired comparison is a protocol correction, NOT
a controlled camera-label-only or reasoning-only causal ablation.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch
import transformers
from alpamayo1_5 import helper
from alpamayo1_5.config import Alpamayo1_5Config
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
from stream_data import HF_REVISION, OrderedPrefetch, official_interface, validate_rows, validate_sample
from native_eval_io import is_retryable_data_error, read_with_retries

MANIFEST_SHA = '5242053734a9659b1e1f81d294665f2a8841998ab7ac8b46f99321cae1edde74'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metric(pred_xy, gt_xy):
    # Identical full-horizon XY minADE formula to NVIDIA quant/eval.py.
    assert pred_xy.shape == (6, 64, 2) and gt_xy.shape == (64, 2)
    assert np.isfinite(pred_xy).all() and np.isfinite(gt_xy).all()
    ade = np.linalg.norm(pred_xy - gt_xy[None, :, :], axis=-1).mean(axis=-1)
    return dict(min_ade_m=float(ade.min()), ade_mean6_m=float(ade.mean()),
                ade_first_m=float(ade[0]), candidate_ade_m=ade.tolist())


def audit_prompt(messages):
    content = messages[1]['content']
    text = [item['text'] for item in content if item['type'] == 'text']
    cams = [helper.CAMERA_DISPLAY_NAMES[c] + ': ' for c in (0, 1, 2, 6)]
    assert sum(c['type'] == 'image' for c in content) == 16
    assert all(text.count(c) == 1 for c in cams)
    assert all(text.count(f'frame {i} ') == 4 for i in range(4))
    assert 'chain-of-thought reasoning' in text[-1]
    assert '<|route_start|>' not in text[-1]
    assert '<|traj_future_start|>' not in ''.join(text)
    assert messages[-1]['content'] == [{'type': 'text', 'text': '<|cot_start|>'}]
    return dict(cameras=cams, image_count=16, frame_labels=16,
                user_text=text[-1], assistant_prefix='<|cot_start|>',
                future_target_in_model_inputs=False)


class Dataset:
    def __init__(self, rows, cache):
        self.rows = rows
        self.avdi = official_interface(cache)

    def __getitem__(self, index):
        row = self.rows[index]
        def fetch():
            return validate_sample(load_physical_aiavdataset(
                row['clip_id'], t0_us=int(row['t0_relative']), avdi=self.avdi,
                maybe_stream=True))

        def refresh():
            from huggingface_hub import HfFileSystem
            previous = self.avdi.fs
            # Each prefetch thread owns its dataset. Discard only that thread's
            # remote filesystem metadata/file handles; keep the exact revision.
            self.avdi.fs = HfFileSystem(token=previous.token,
                endpoint=previous.endpoint, skip_instance_cache=True)

        return read_with_retries(fetch, refresh, label=row['clip_id'])


def main(args, model_loader=None):
    # A separate distilled loader may reuse this exact data/prompt/metric loop.
    steps = getattr(args, 'steps', 10)
    assert steps in (10, 5)
    output = args.output
    rows = read(args.manifest)
    assert sha(args.manifest) == MANIFEST_SHA
    assert len(rows) == len({r['clip_id'] for r in rows}) == 644
    assert all(r['t0_relative'] == 5_100_000 for r in rows)
    output.mkdir(parents=True, exist_ok=True)
    records_dir = output / 'per_clip'
    records_dir.mkdir(exist_ok=True)
    started = time.time()
    write(output/'status.json', dict(status='running', phase='loading_native_model', pid=os.getpid()))
    checkpoint_config = read(args.checkpoint/'config.json')
    assert checkpoint_config['include_camera_ids'] and checkpoint_config['include_frame_nums']
    # Native config builds its processor in __init__, before HF applies overrides.
    # Set the local asset path before construction; checkpoint weights are unchanged.
    if model_loader is None:
        config = Alpamayo1_5Config(**{**checkpoint_config, 'vlm_name_or_path': str(args.cosmos_assets)})
        model, loading = Alpamayo1_5.from_pretrained(
            args.checkpoint, config=config, dtype=torch.float16,
            local_files_only=True, output_loading_info=True)
    else:
        model, loading = model_loader(args, checkpoint_config)
    write(output/'loading_info.json', loading)
    allowed_unused = set(getattr(args, 'allowed_unused_keys', ()))
    assert not any(loading.get(k) for k in ('missing_keys','mismatched_keys','error_msgs')), loading
    assert not (set(loading.get('unexpected_keys', ())) - allowed_unused), loading
    assert not any(p.is_meta for p in model.parameters())
    assert not any(b.is_meta for b in model.buffers())
    assert model.tokenizer.traj_token_ids == checkpoint_config['traj_token_ids']
    assert model.tokenizer.traj_token_start_idx == checkpoint_config['traj_token_start_idx']
    assert len(model.tokenizer) == checkpoint_config['vocab_size']
    assert model.diffusion.num_inference_steps == 10
    model = model.to(device='cuda', dtype=torch.float16).eval().requires_grad_(False)
    if hasattr(model, '_native_eval_post_cast'):
        model._native_eval_post_cast()
    # Same processor repository as helper.get_processor, resolved once to an
    # immutable local snapshot by the launcher (no tokenizer substitutions).
    helper.BASE_PROCESSOR_NAME = str(args.processor_assets)
    processor = helper.get_processor(model.tokenizer)
    audit = dict(loading=loading, native_class=type(model).__name__,
                 dtype=str(next(model.parameters()).dtype), torch=torch.__version__,
                 transformers=transformers.__version__, gpu=torch.cuda.get_device_name(),
                 vlm_attention=model.vlm.config._attn_implementation,
                 vision_attention=model.vlm.config.vision_config._attn_implementation,
                 text_attention=model.vlm.config.text_config._attn_implementation,
                 expert_attention=model.expert.config._attn_implementation,
                 tokenizer_ids=model.tokenizer.traj_token_ids, dataset_revision=HF_REVISION,
                 checkpoint_config_sha256=sha(args.checkpoint/'config.json'))
    write(output/'model_audit.json', audit)
    print('MODEL_AUDIT', json.dumps(audit), flush=True)
    validate_rows(rows, official_interface(args.hf_cache))
    observed = {}
    real_generate = model.vlm.generate
    real_sample = model.diffusion.sample

    def generate(**kwargs):
        observed['input_len'] = kwargs['input_ids'].shape[1]
        result = real_generate(**kwargs)
        # Keep the original tensor (native replacement creates a new one).
        # Copy to CPU only after the model-call timer finishes.
        observed['sequences'] = result.sequences
        return result

    def sample(*a, **kw):
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        result = real_sample(*a, **kw)
        stop.record()
        observed['expert_events'] = (start, stop)
        return result

    def count_expert(module, inputs, outputs):
        observed['expert_calls'] = observed.get('expert_calls', 0) + 1

    model.vlm.generate = generate
    model.diffusion.sample = sample
    model.expert.register_forward_hook(count_expert)
    future_id = model.tokenizer.convert_tokens_to_ids('<|traj_future_start|>')
    cot_id = model.tokenizer.convert_tokens_to_ids('<|cot_start|>')

    def evaluate(data, index):
        messages = helper.create_message(data['image_frames'].flatten(0, 1),
                                         camera_indices=data['camera_indices'])
        prompt_audit = audit_prompt(messages)
        inputs = processor.apply_chat_template(messages, tokenize=True,
            add_generation_prompt=False, continue_final_message=True,
            return_dict=True, return_tensors='pt')
        assert not (inputs['input_ids'] == future_id).any()
        assert inputs['input_ids'][0, -1].item() == cot_id
        if not (output/'prompt_audit.json').exists():
            prompt_audit['tokenized_prompt'] = model.tokenizer.decode(inputs['input_ids'][0])
            write(output/'prompt_audit.json', prompt_audit)
        model_inputs = helper.to_device(dict(tokenized_data=inputs,
            ego_history_xyz=data['ego_history_xyz'], ego_history_rot=data['ego_history_rot']), 'cuda')
        gt_xy = data['ego_future_xyz'].cpu().numpy()[0, 0, :, :2]
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        observed.clear()
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            xyz, rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs, top_p=.98, temperature=.6, num_traj_samples=6,
                max_generation_length=getattr(args, 'generation_arg', 256),
                diffusion_kwargs={'inference_step':steps}, return_extra=True)
        torch.cuda.synchronize()
        model_ms = 1000*(time.perf_counter()-start)
        assert observed['expert_calls'] == steps
        assert tuple(xyz.shape) == (1, 1, 6, 64, 3)
        assert torch.isfinite(xyz).all() and torch.isfinite(rot).all()
        sequences = observed.pop('sequences').cpu()
        generated = sequences[:, observed['input_len']:]
        has_marker = (generated == future_id).any(dim=1)
        marker_offsets = [int((seq == future_id).nonzero()[0].item()) if found else None
                          for seq, found in zip(generated, has_marker.tolist())]
        record = dict(sample_index=index, **rows[index], solver_steps=steps, seed=42,
                      num_candidates=6, end_to_end_model_ms=model_ms,
                      action_expert_diffusion_ms=observed['expert_events'][0].elapsed_time(observed['expert_events'][1]),
                      expert_calls=observed['expert_calls'], generated_future_marker_offsets=marker_offsets,
                      generated_text={k:v.tolist() for k,v in extra.items()},
                      **metric(xyz.detach().cpu().numpy()[0,0,:,:,:2], gt_xy))
        if index == 0:
            # Native fallback is allowed for rare truncated candidates later,
            # but the launch smoke must prove real reasoning before the marker.
            assert all(p is not None and p > 0 for p in marker_offsets), marker_offsets
            assert all(str(s).strip() for s in extra['cot'].flatten())
        return record, xyz.detach().cpu().numpy(), gt_xy

    pending = []
    records = []
    for i, row in enumerate(rows):
        path = records_dir/f'{i:04d}.json'
        if path.exists():
            record = read(path)
            assert record['sample_index'] == i and record['clip_id'] == row['clip_id']
            assert record['t0_relative'] == row['t0_relative'] and record['solver_steps'] == steps
            records.append(record)
        else:
            pending.append(i)
    if pending:
        write(output/'status.json', dict(status='running', phase='native_smoke_and_warmup', completed=len(records), total=644))
        warm, _, _ = evaluate(Dataset(rows, args.hf_cache)[0], 0)
        write(output/'smoke_passed.json', dict(passed=True, excluded_from_metrics=True,
              labels_present=True, native_reasoning_prefix=True, model_load_strict=True,
              **warm))
        print('NATIVE_SMOKE_PASSED', json.dumps(warm), flush=True)
        if getattr(args, 'smoke_only', False):
            write(output/'status.json', dict(status='complete', phase='smoke_only', evaluated_clips=0))
            return
        prefetch = OrderedPrefetch(pending, lambda: Dataset(rows, args.hf_cache), workers=4, capacity=8)
        try:
            for i in pending:
                record, predictions, gt = evaluate(prefetch.get(i), i)
                # Predictions precede the atomic JSON completion marker.
                np.savez_compressed(records_dir/f'{i:04d}.npz', pred_xyz=predictions, gt_xy=gt)
                write(records_dir/f'{i:04d}.json', record)
                records.append(record)
                elapsed = time.time()-started
                write(output/'status.json', dict(status='running', phase='gold644', pid=os.getpid(),
                      completed=len(records), total=644, last_clip=record['clip_id'],
                      partial_min_ade_m=float(np.mean([r['min_ade_m'] for r in records])),
                      process_elapsed_seconds=elapsed))
                print(f"PROGRESS {len(records)}/644 minADE={record['min_ade_m']:.6f} model_ms={record['end_to_end_model_ms']:.1f}", flush=True)
                gc.collect()
        finally:
            prefetch.close()
    assert len(records) == 644
    records.sort(key=lambda r:r['sample_index'])
    current = np.array([r['min_ade_m'] for r in records])
    comparison = {}
    if getattr(args, 'legacy_records', None) is not None:
        old = [json.loads(line) for line in args.legacy_records.read_text().splitlines()]
        old_by_id = {r['clip_id']:r for r in old}
        assert set(old_by_id) == {r['clip_id'] for r in records} and len(old) == 644
        previous = np.array([old_by_id[r['clip_id']]['metrics']['min_ade'][0] for r in records])
        delta = current-previous
        rng = np.random.default_rng(42)
        interval = np.percentile(delta[rng.integers(0,644,size=(10000,644))].mean(axis=1), [2.5,97.5])
        comparison = dict(old_protocol_min_ade_m=float(previous.mean()), paired_change_m=float(delta.mean()),
            paired_change_percent=float(100*(current.mean()/previous.mean()-1)),
            paired_clip_bootstrap95_delta_m=interval.tolist())
    summary = dict(status='complete', evaluated_clips=644, skipped_clips=0, solver_steps=steps,
        candidates_per_clip=6, min_ade_m=float(current.mean()),
        ade_mean6_m=float(np.mean([r['ade_mean6_m'] for r in records])),
        ade_first_m=float(np.mean([r['ade_first_m'] for r in records])),
        mean_model_ms=float(np.mean([r['end_to_end_model_ms'] for r in records])),
        mean_action_expert_ms=float(np.mean([r['action_expert_diffusion_ms'] for r in records])),
        generated_candidates_missing_future_marker=sum(p is None for r in records for p in r['generated_future_marker_offsets']),
        **comparison,
        caveat='Native inference protocol; checkpoint/cohort comparisons are reported separately by the suite. Not a safety evaluation.')
    write(output/'summary.json', summary)
    write(output/'status.json', summary)
    print('COMPLETE', json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('checkpoint','cosmos-assets','processor-assets','manifest','hf-cache','output'):
        parser.add_argument('--'+flag, required=True, type=Path)
    parser.add_argument('--legacy-records', type=Path)
    parser.add_argument('--steps', type=int, choices=(10,5), default=10)
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        from requests.exceptions import RequestException
        network = is_retryable_data_error(error)
        write(args.output/'failure.json', dict(error=repr(error), traceback=traceback.format_exc(), retryable_network=network))
        write(args.output/'status.json', dict(status='failed', error=repr(error), retryable_network=network))
        traceback.print_exc()
        raise SystemExit(75 if network else 1)
