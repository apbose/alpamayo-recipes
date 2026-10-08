# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
"""Native-input golden evaluation of existing A6/A8 EMA action weights.

No retraining and no adapter reset. A6/A8 use the native 10B inference shell
with their exact EMA action state; all frozen VLM tensors are audited against
the released shell at inference precision.
Saved analytical action buffers remain FP32; matrix compute/weights are FP16.
10/5 steps set d=0.1/0.2, both off the paper-style dyadic training grid.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import traceback

import torch
from torch import nn
from safetensors import safe_open

import evaluate_native_alpamayo15_gold as common
from native_eval_io import is_retryable_data_error
from alpamayo1_5.config import Alpamayo1_5Config
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5


def recipe_source(name):
    """Use immutable helper snapshots when launched by the portable runner."""
    source = Path(__file__).resolve().parent
    frozen = source / ("legacy_" + name)
    if frozen.exists():
        return frozen
    return source.parents[2] / "recipes/alpamayo1_5_sft/models" / name


def import_file(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def original_paper_encoder(path):
    # Reuse exactly the original class definition without importing its full
    # training wrapper/dependencies or replacing the native inference model.
    tree=ast.parse(path.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PaperLogStepEncoder')
    namespace=dict(torch=torch,nn=nn,math=math)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    return namespace['PaperLogStepEncoder']


def read_prefix(checkpoint, prefix):
    index=common.read(checkpoint/'model.safetensors.index.json')['weight_map']
    keys={k:v for k,v in index.items() if k.startswith(prefix)}
    assert keys, prefix
    result={}
    for shard in sorted(set(keys.values())):
        with safe_open(checkpoint/shard,framework='pt',device='cpu') as handle:
            for key in sorted(k for k,v in keys.items() if v==shard):
                result[key[len(prefix):]]=handle.get_tensor(key)
    return result


def frozen_vlm_audit(model, args):
    index=common.read(args.action_checkpoint/'model.safetensors.index.json')['weight_map']
    source_files=[args.action_checkpoint/p for p in sorted(set(index.values()))]
    base_index=common.read(args.checkpoint/'model.safetensors.index.json')['weight_map']
    source_files += [args.checkpoint/p for p in sorted(set(base_index.values()))]
    signature={str(p):dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in source_files}
    signature['base_config_sha256']=common.sha(args.checkpoint/'config.json')
    signature['trained_index_sha256']=common.sha(args.action_checkpoint/'model.safetensors.index.json')
    cached=args.audit_cache/'frozen_vlm.json'
    if cached.exists():
        report=common.read(cached)
        assert report['signature']==signature and report['passed']
        return dict(report,reused=True)
    target=model.vlm.state_dict()
    keys={k:v for k,v in index.items() if k.startswith('vlm.')}
    assert {k[4:] for k in keys}==set(target)
    checked=0
    for shard in sorted(set(keys.values())):
        with safe_open(args.action_checkpoint/shard,framework='pt',device='cpu') as handle:
            for key in sorted(k for k,v in keys.items() if v==shard):
                value=handle.get_tensor(key)
                expected=target[key[4:]].detach().cpu()
                assert value.shape==expected.shape and torch.equal(value.to(expected.dtype),expected), key
                checked+=1
        print('VLM_AUDIT',checked,'/',len(keys),flush=True)
    report=dict(passed=True,tensors=checked,equality='all checkpoint VLM tensors equal at FP16 inference dtype',
                signature=signature,reused=False)
    common.write(cached,report)
    return report


def preserve_buffers_after_cast(model, modules):
    buffers={name:value.detach().float().clone() for name,value in modules.named_buffers()
             if value.is_floating_point()}
    def restore():
        for name,value in buffers.items():
            parent,_,leaf=name.rpartition('.')
            module=modules.get_submodule(parent)
            device=getattr(module,leaf).device
            setattr(module,leaf,value.to(device=device,dtype=torch.float32))
    model._native_eval_post_cast=restore
    return list(buffers)


def load_adapted(args, base_config):
    trained=common.read(args.action_checkpoint/'config.json')
    complete=common.read(args.action_checkpoint/'COMPLETE.json')
    assert args.kind in ('A6', 'A8'), 'Only public 10B shortcut checkpoints are supported'
    assert complete.get('global_step') == 249 and complete.get('completed_utc'), complete
    # A6 predates serialization of this option; its original class defaults
    # to ema_bootstrap and its saved protocol records 16 bootstrap pairs.
    assert trained.get('paper_supervision','ema_bootstrap')=='ema_bootstrap'
    assert common.read(args.action_checkpoint.parent/'protocol.json')['bootstrap_pairs']==16
    assert trained['shortcut_inference_weights']=='ema'
    assert trained['paper_step_encoding']=='negative_log2_sinusoidal_256_maxperiod10000'
    config=Alpamayo1_5Config(**{**base_config,'vlm_name_or_path':str(args.cosmos_assets)})
    model,loading=Alpamayo1_5.from_pretrained(args.checkpoint,config=config,dtype=torch.float16,
        local_files_only=True,output_loading_info=True)
    vlm_audit=frozen_vlm_audit(model,args)
    legacy=import_file(recipe_source('shortcut_modules.py'),'native_eval_legacy_shortcut_modules')
    Encoder=original_paper_encoder(recipe_source('paper_shortcut_alpamayo.py'))
    options={k:v for k,v in trained['action_in_proj_cfg'].items() if k!='_target_'}
    projection=legacy.StepSizeConditionedActionInProjV2(
        in_dims=model.action_space.get_action_space_dims(),out_dim=model.expert.config.hidden_size,**options)
    projection.step_size_fourier_encoder=Encoder(256)
    model.action_in_proj=projection
    action=nn.ModuleDict({name:getattr(model,name).float() for name in ('action_in_proj','expert','action_out_proj')})
    loaded={}
    for name in action:
        state=read_prefix(args.action_checkpoint,'ema_'+name+'.')
        action[name].load_state_dict(state,strict=True)
        assert all(torch.equal(action[name].state_dict()[k],v.to(action[name].state_dict()[k].dtype)) for k,v in state.items())
        loaded[name]=len(state)
    projection.set_default_step_size(1.0/args.steps)
    expected=torch.exp(-math.log(10000)*torch.arange(128).float()/128)[None]
    assert torch.allclose(projection.step_size_fourier_encoder.freqs,expected,rtol=1e-5,atol=1e-7)
    assert all(torch.isfinite(v).all() for v in action.state_dict().values() if v.is_floating_point())
    buffers=preserve_buffers_after_cast(model,action)
    model.eval().requires_grad_(False)
    common.write(args.output/'action_load_audit.json',dict(passed=True,kind=args.kind,
        checkpoint=str(args.action_checkpoint),selected_weights='EMA',loaded_tensors=loaded,
        adapter_reinitialized=False,strict_action_state=True,all_finite=True,
        step_size=1.0/args.steps,off_dyadic_grid=True,vlm_audit=vlm_audit,
        preserved_fp32_buffers=buffers,action_config_sha256=common.sha(args.action_checkpoint/'config.json')))
    return model,loading


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ('checkpoint','cosmos-assets','processor-assets','manifest','hf-cache','output','audit-cache'):
        parser.add_argument('--'+flag,type=Path,required=True)
    parser.add_argument('--legacy-records',type=Path)
    parser.add_argument('--action-checkpoint',type=Path)
    parser.add_argument('--kind',choices=('A6','A8','R0'),required=True)
    parser.add_argument('--steps',type=int,choices=(10,5),required=True)
    parser.add_argument('--smoke-only',action='store_true')
    args=parser.parse_args()
    try:
        common.main(args,model_loader=(None if args.kind=='R0' else load_adapted))
    except Exception as error:
        from requests.exceptions import RequestException
        network=is_retryable_data_error(error)
        common.write(args.output/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),retryable_network=network))
        common.write(args.output/'status.json',dict(status='failed',error=repr(error),retryable_network=network))
        traceback.print_exc()
        raise SystemExit(75 if network else 1)
