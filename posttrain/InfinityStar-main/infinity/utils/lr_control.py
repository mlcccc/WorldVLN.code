# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT
import math
from pprint import pformat
from typing import Tuple, List, Dict, Union

import torch.nn
import infinity.utils.dist as dist

def filter_params(model, ndim_dict, nowd_keys=(), lr_scale=0.0) -> Tuple[
    List[str], List[torch.nn.Parameter], List[Dict[str, Union[torch.nn.Parameter, float]]]
]:
    with_lr_scale = hasattr(model, 'get_layer_id_and_scale_exp') and 0 < lr_scale <= 1
    print(f'[get_param_groups][lr decay] with_lr_scale={with_lr_scale}, lr_scale={lr_scale}')
    para_groups, para_groups_dbg = {}, {}
    names, paras = [], []
    names_no_grad = []
    frozen_count, frozen_numel = 0, 0
    count, numel = 0, 0
    for name, para in model.named_parameters():
        name = name.replace('_fsdp_wrapped_module.', '')
        if not para.requires_grad:
            names_no_grad.append(name)
            frozen_count += 1
            frozen_numel += para.numel()
            continue  # frozen weights
        count += 1
        numel += para.numel()
        names.append(name)
        paras.append(para)
        
        if ndim_dict.get(name, 2) == 1 or name.endswith('bias') or any(k in name for k in nowd_keys):
            cur_wd_sc, group_name = 0., 'ND'
        # elif any(k in name for k in small_wd_keys):
        #     cur_wd_sc, group_name = small_wd, 'small_decay'
        else:
            cur_wd_sc, group_name = 1., 'D'
        
        if with_lr_scale:
            layer_id, scale_exp = model.get_layer_id_and_scale_exp(name)
            group_name = f'layer{layer_id}_' + group_name
            cur_lr_sc = lr_scale ** scale_exp
            dbg = f'[layer {layer_id}][sc = {lr_scale} ** {scale_exp}]'
        else:
            cur_lr_sc = 1.
            dbg = f'[no scale]'
        
        if group_name not in para_groups:
            para_groups[group_name] = {'params': [], 'wd_sc': cur_wd_sc, 'lr_sc': cur_lr_sc}
            para_groups_dbg[group_name] = {'params': [], 'wd_sc': cur_wd_sc, 'lr_sc': dbg}
        para_groups[group_name]['params'].append(para)
        para_groups_dbg[group_name]['params'].append(name)
    
    for g in para_groups_dbg.values():
        g['params'] = pformat(', '.join(g['params']), width=200)
    
    print(f'[get_param_groups] param_groups = \n{pformat(para_groups_dbg, indent=2, width=240)}\n')
    
    for rk in range(dist.get_world_size()):
        dist.barrier()
        if dist.get_rank() == rk:
            print(f'[get_param_groups][rank{dist.get_rank()}] {type(model).__name__=} {count=}, {numel=}', flush=True, force=True)
    print('')
    
    if len(names_no_grad) > 0:
        print(
            '[get_param_groups][frozen] '
            f'frozen_count={frozen_count}, frozen_numel={frozen_numel}, '
            f'samples={names_no_grad[:16]}',
            flush=True,
        )
    del ndim_dict
    return names, paras, list(para_groups.values())