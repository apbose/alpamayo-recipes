#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Portable, resumable native gold-644 evaluation. Never trains or skips clips.

Provide machine-local assets in a JSON plan, not in checked-in source. Runs
trained checkpoints before optional released references. Independent stages
continue after a bounded failure; partial data never becomes a complete score.
Only the public released 10B model and its A6/A8 shortcut checkpoints are supported.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback

NATIVE_COMMIT = '7a8f1c781a826f09be53e1e211f26e947ec18019'
MANIFEST_SHA = '5242053734a9659b1e1f81d294665f2a8841998ab7ac8b46f99321cae1edde74'
KINDS = {'R0', 'A6', 'A8'}
TRAINED = {'A6', 'A8'}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_jobs(entries):
    ordered = sorted(entries, key=lambda entry: entry['kind'] not in TRAINED)
    return [(entry, steps) for entry in ordered for steps in (10, 5)]


def execute_jobs(jobs, run, after_each):
    outcomes = {}
    for entry, steps in jobs:
        outcomes[f"{entry['name']}_{steps}step"] = bool(run(entry, steps))
        after_each()
    return outcomes


def load_plan(args):
    plan = read(args.plan)
    def path(value):
        p = Path(value).expanduser()
        return str((args.plan.parent / p).resolve() if not p.is_absolute() else p.resolve())
    for key in ('manifest', 'processor_assets', 'hf_cache'):
        plan[key] = path(plan[key])
    assert sha(plan['manifest']) == MANIFEST_SHA, 'Wrong gold manifest/order'
    rows = read(plan['manifest'])
    assert len(rows) == len({r['clip_id'] for r in rows}) == 644
    assert all(r['t0_relative'] == 5_100_000 for r in rows)
    assert plan['models'] and len({e['name'] for e in plan['models']}) == len(plan['models'])
    for entry in plan['models']:
        assert re.fullmatch(r'[A-Za-z0-9_-]+', entry['name']), 'Invalid stage name'
        assert entry['kind'] in KINDS
        for key in ('checkpoint', 'cosmos_assets'):
            entry[key] = path(entry[key])
            assert (Path(entry[key]) / 'config.json').is_file(), entry[key]
        if entry['kind'] in TRAINED:
            entry['action_checkpoint'] = path(entry['action_checkpoint'])
            assert (Path(entry['action_checkpoint']) / 'COMPLETE.json').is_file()
    if plan.get('cohort_audit'):
        plan['cohort_audit'] = path(plan['cohort_audit'])
        audit = read(plan['cohort_audit'])
        assert set(audit['shared_strict_heldout_ids']) <= {r['clip_id'] for r in rows}
        for entry in plan['models']:
            if entry['kind'] not in TRAINED:
                continue
            checkpoint = Path(entry['action_checkpoint'])
            metadata = read(checkpoint.parent / 'protocol.json')
            assert metadata['train_manifest_sha256'] == audit['models'][entry['name']]['train_manifest_sha256'], 'Cohort audit belongs to another training population'
    assert subprocess.check_output(['git', '-C', str(args.native_source), 'rev-parse', 'HEAD'], text=True).strip() == NATIVE_COMMIT
    assert not subprocess.check_output(['git', '-C', str(args.native_source), 'status', '--porcelain'], text=True).strip(), 'Native source must be clean'
    return plan


def prepare(args, plan):
    source = args.output / 'source_snapshot'; source.mkdir(exist_ok=True)
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    pairs = [(p, p.name) for p in here.glob('*.py')]
    pairs += [(repo / 'recipes/alpamayo1_5_sft/models' / name, 'legacy_' + name)
              for name in ('shortcut_modules.py', 'paper_shortcut_alpamayo.py')]
    for original, name in pairs:
        destination = source / name
        if destination.exists():
            assert sha(destination) == sha(original), 'Source changed on resume; use the existing frozen source or a new output'
        else:
            shutil.copy2(original, destination)
    provenance = dict(plan=plan, native_commit=NATIVE_COMMIT,
        native_source=str(args.native_source),
        source_sha256={p.name:sha(p) for p in source.glob('*.py')},
        protocol='FP16, VLM FA2/expert SDPA, native labelled reasoning prompt, seed 42 per clip, 6 candidates',
        step_sizes=[0.1, 0.2], off_dyadic_training_grid=True,
        training=False, safety_evaluated=False, timeout=None)
    if (args.output / 'plan.json').exists():
        assert read(args.output / 'plan.json') == provenance, 'Plan/source changed on resume'
    else:
        write(args.output / 'plan.json', provenance)
    env = dict(os.environ)
    env.pop('HF_HUB_OFFLINE', None); env.pop('TRANSFORMERS_OFFLINE', None)
    paths = [args.native_source / 'src', source]
    paths += [p for p in env.get('PYTHONPATH', '').split(os.pathsep) if p]
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONUNBUFFERED='1', OMP_NUM_THREADS='8',
        TOKENIZERS_PARALLELISM='false', PYTHONPATH=os.pathsep.join(map(str, paths)),
        HF_HUB_DOWNLOAD_TIMEOUT='120', HF_HUB_ETAG_TIMEOUT='30')
    # Preserve the user's HF auth/cache location. Only computation caches move.
    for key, subdir in [('TORCH_HOME','torch'), ('CUDA_CACHE_PATH','cuda'), ('TRITON_CACHE_DIR','triton')]:
        env.setdefault(key, str(args.output / 'runtime_cache' / subdir))
    with (args.output / 'unit_tests.log').open('wb') as log:
        subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(source),
            '-p', 'test_native*.py', '-v'], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    return source, env


def wait_for_idle(args, phase):
    consecutive = 0
    while consecutive < 2:
        line = subprocess.check_output(['nvidia-smi', f'--id={args.gpu}',
            '--query-gpu=uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
        uuid, memory, utilization = [v.strip() for v in line.split(',')]
        applications = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
            '--format=csv,noheader,nounits'], text=True)
        busy = any(row.split(',')[0].strip() == uuid for row in applications.splitlines())
        idle = not busy and int(memory) < 512 and int(utilization) == 0
        consecutive = consecutive + 1 if idle else 0
        write(args.output / 'status.json', dict(status='waiting_for_gpu', phase=phase,
            gpu=args.gpu, supervisor_pid=os.getpid(), idle_checks=consecutive))
        if consecutive < 2:
            time.sleep(30)


def run_stage(args, plan, source, env, entry, steps):
    label = f"{entry['name']}_{steps}step"
    stage = args.output / label; stage.mkdir(exist_ok=True)
    if (stage / ('smoke_passed.json' if args.smoke_only else 'summary.json')).exists():
        return True
    command = [sys.executable, '-u', str(source / 'evaluate_native_shortcut_gold.py'),
        '--kind', entry['kind'], '--steps', str(steps), '--output', str(stage),
        '--checkpoint', entry['checkpoint'], '--cosmos-assets', entry['cosmos_assets'],
        '--processor-assets', plan['processor_assets'], '--manifest', plan['manifest'],
        '--hf-cache', plan['hf_cache'], '--audit-cache', str(args.output / 'load_audits' / entry['name'])]
    if entry['kind'] in TRAINED:
        command += ['--action-checkpoint', entry['action_checkpoint']]
    if args.smoke_only:
        command += ['--smoke-only']
    for attempt in range(1, 4):
        wait_for_idle(args, label)
        with (stage / f'attempt{attempt}.log').open('ab', buffering=0) as log:
            child = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT)
        write(args.output / 'status.json', dict(status='running', phase=label,
            worker_pid=child.pid, supervisor_pid=os.getpid(), attempt=attempt))
        code = child.wait()
        if code == 0:
            return True
        if code != 75 or attempt == 3:
            write(stage / 'stage_failed.json', dict(exit_code=code, attempt=attempt,
                saved_clips=len(list((stage / 'per_clip').glob('*.json'))),
                decision='keep saved clips and continue other stages'))
            return False
        time.sleep(30)


def report(args, plan):
    expected = {r['clip_id'] for r in read(plan['manifest'])}
    audit = read(plan['cohort_audit']) if plan.get('cohort_audit') else None
    cohorts = {'full644':expected}
    if audit:
        cohorts[f"shared_heldout{len(audit['shared_strict_heldout_ids'])}"] = set(audit['shared_strict_heldout_ids'])
    scores, unavailable = {}, []
    for entry, steps in build_jobs(plan['models']):
        label = f"{entry['name']}_{steps}step"; stage = args.output / label
        if not (stage / 'summary.json').exists():
            unavailable.append(label); continue
        rows = [read(p) for p in sorted((stage / 'per_clip').glob('*.json'))]
        assert len(rows) == 644 and {r['clip_id'] for r in rows} == expected
        scores[label] = {}
        for name, ids in cohorts.items():
            selected = [r for r in rows if r['clip_id'] in ids]
            assert selected and len(selected) == len(ids)
            scores[label][name] = dict(clips=len(selected),
                min_ade_m=sum(r['min_ade_m'] for r in selected)/len(selected),
                model_ms=sum(r['end_to_end_model_ms'] for r in selected)/len(selected),
                expert_ms=sum(r['action_expert_diffusion_ms'] for r in selected)/len(selected))
    for entry in plan['models']:
        ten, five = (f"{entry['name']}_{steps}step" for steps in (10,5))
        if ten in scores and five in scores:
            for cohort in cohorts:
                scores[five][cohort]['change_vs_own10_percent'] = 100*(scores[five][cohort]['min_ade_m']/scores[ten][cohort]['min_ade_m']-1)
    write(args.output / 'summary.json', dict(status='complete' if not unavailable else 'incomplete',
        metrics=scores, unavailable=unavailable, safety_evaluated=False,
        cohort_caveat='Full-644 A8 is diagnostic: 147 pool-overlap/112 actually trained gold clips; use the verified shared heldout cohort for these saved experiments.' if audit else 'Training overlap not checked; no heldout claim.'))


def main(args):
    args.plan = args.plan.resolve(); args.output = args.output.resolve()
    args.native_source = args.native_source.resolve()
    plan = load_plan(args)
    if args.validate_only:
        print(json.dumps(dict(validated=True, jobs=[(e['name'],s) for e,s in build_jobs(plan['models'])]), indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source, env = prepare(args, plan)
        outcomes = execute_jobs(build_jobs(plan['models']),
            lambda entry,steps:run_stage(args, plan, source, env, entry, steps),
            lambda:None if args.smoke_only else report(args, plan))
        status = 'smoke_complete' if args.smoke_only else 'complete'
        write(args.output / 'status.json', dict(status=status if all(outcomes.values()) else 'incomplete', outcomes=outcomes))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('plan', 'output', 'native-source'):
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        write(args.output / 'status.json', dict(status='failed', error=repr(error), traceback=traceback.format_exc()))
        raise
