
import torch
from logging import getLogger

logger = getLogger()

def apply_masks(x, masks):
    if not isinstance(masks, list):
        masks = [masks]
    
    all_x = []
    for m in masks:
        m = m[0]  # [B, num_points]
        B, num_points = m.shape
                
        # Chọn các điểm từ x theo chỉ số trong m
        batch_indices = torch.arange(B, device=x.device).view(-1, 1).expand(B, num_points)
        x_masked = x[batch_indices, m]  # [B, num_points, D]
        
        all_x.append(x_masked)
    
    if len(all_x) > 1:
        x = torch.cat(all_x, dim=0)
    else:
        x = all_x[0]
    return x

def repeat_interleave_batch(x, B, repeat):
   
    N = x.size(0) // B
    x = torch.cat([x[i::N] for i in range(N)], dim=0)
    x = x.view(B, N, *x.shape[1:]).repeat_interleave(repeat, dim=0)
    return x