# SPDX-License-Identifier: Apache-2.0
"""Small CPU parity checks using the original training adapter source."""
import copy
from pathlib import Path
import unittest

import torch
from alpamayo1_5.models.action_in_proj import PerWaypointActionInProjV2
from evaluate_native_shortcut_gold import import_file, original_paper_encoder, preserve_buffers_after_cast, recipe_source

SOURCE=Path(__file__).parent


class AdapterParityTest(unittest.TestCase):
    def test_original_projection_zero_branch_matches_native(self):
        legacy=import_file(recipe_source('shortcut_modules.py'),'test_legacy_shortcut_modules')
        options=dict(in_dims=[8,2],out_dim=16,num_enc_layers=2,hidden_size=32,num_fourier_feats=20)
        base=PerWaypointActionInProjV2(**options)
        projection=legacy.StepSizeConditionedActionInProjV2(**options,step_size_fourier_feats=256,step_size_hidden_size=32)
        projection.load_state_dict(base.state_dict(),strict=False)
        projection.step_size_fourier_encoder=original_paper_encoder(recipe_source('paper_shortcut_alpamayo.py'))(256)
        x,t=torch.randn(3,8,2),torch.rand(3,1,1)
        for steps in (10,5):
            projection.set_default_step_size(1/steps)
            self.assertTrue(torch.equal(projection(x,t),base(x,t)))
        # Once nonzero, the saved original branch responds to requested d.
        torch.nn.init.normal_(projection.step_size_adapter[-1].weight,std=.01)
        projection.set_default_step_size(.1); first=projection(x,t)
        projection.set_default_step_size(.2); second=projection(x,t)
        self.assertFalse(torch.equal(first,second))
        clone=copy.deepcopy(projection)
        clone.load_state_dict(projection.state_dict(),strict=True)
        self.assertTrue(torch.equal(clone(x,t),second))

    def test_fp32_analytical_buffer_survives_global_half_cast(self):
        model=torch.nn.Module()
        model.action=torch.nn.Linear(2,2)
        model.action.register_buffer('frequency',torch.tensor([.123456789],dtype=torch.float32))
        action=torch.nn.ModuleDict({'action':model.action})
        expected=model.action.frequency.clone()
        preserve_buffers_after_cast(model,action)
        model.half(); model._native_eval_post_cast()
        self.assertEqual(model.action.weight.dtype,torch.float16)
        self.assertEqual(model.action.frequency.dtype,torch.float32)
        self.assertTrue(torch.equal(model.action.frequency,expected))


if __name__=='__main__':
    unittest.main()
