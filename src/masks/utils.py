# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
from logging import getLogger

logger = getLogger()

def apply_masks(x, masks):
    """
    Apply masks to the input tensor x using index-based masks.
    
    Args:
        x: Input tensor of shape [B, N, D] (batch_size, sequence_length, embed_dim)
        masks: List of index tensors, each of shape [num_masks, B, num_points]
    
    Returns:
        Masked tensor of shape [B, num_points, D]
    """
    if not isinstance(masks, list):
        masks = [masks]
    
    all_x = []
    for m in masks:
        
        # logger.info(f"apply_masks: mask shape={m.shape}") # Debug: Kiểm tra shape của mask
        
        # m: [num_masks, B, num_points] -> lấy mask đầu tiên nếu num_masks=1
        m = m[0]  # [B, num_points]
        B, num_points = m.shape
        
        
        #logger.info(f"apply_masks: input x shape={x.shape}") # Kiểm tra shape của x
        
        # Chọn các điểm từ x theo chỉ số trong m
        batch_indices = torch.arange(B, device=x.device).view(-1, 1).expand(B, num_points)
        x_masked = x[batch_indices, m]  # [B, num_points, D]
        
        all_x.append(x_masked)
    
    # Nếu có nhiều mask, ghép kết quả
    if len(all_x) > 1:
        x = torch.cat(all_x, dim=0)
    else:
        x = all_x[0]
    
    #logger.info(f"apply_masks: output x shape={x.shape}")
    return x

def repeat_interleave_batch(x, B, repeat):
    """
    Repeat and interleave a batch tensor.
    
    Args:
        x: Input tensor of shape [B, ...]
        B: Batch size
        repeat: Number of times to repeat
    
    Returns:
        Repeated tensor of shape [B * repeat, ...]
    """
    N = x.size(0) // B
    x = torch.cat([x[i::N] for i in range(N)], dim=0)
    x = x.view(B, N, *x.shape[1:]).repeat_interleave(repeat, dim=0)
    return x