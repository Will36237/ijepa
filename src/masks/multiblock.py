# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math
from multiprocessing import Value
from logging import getLogger
import torch

_GLOBAL_SEED = 0
logger = getLogger()

class Multiblock_Mask(object):
    def __init__(
        self,
        window_size=20,
        segment_size=5,
        enc_mask_scale=(0.2, 0.5),
        pred_mask_scale=(0.2, 0.5),
        nenc=1,
        npred=2,
        min_keep=4,
        allow_overlap=True  # Chuỗi thời gian đơn giản hơn, cho phép overlap
    ):
        super(Multiblock_Mask, self).__init__()
        self.window_size = window_size
        self.segment_size = segment_size
        self.num_segments = window_size // segment_size
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.nenc = nenc
        self.npred = npred
        self.min_keep = min_keep
        self.allow_overlap = allow_overlap
        self._itr_counter = Value('i', -1)

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_block_size(self, generator, scale):
        _rand = torch.rand(1, generator=generator).item()
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        num_segments_to_mask = max(1, int(self.num_segments * mask_scale))
        return num_segments_to_mask

    def _sample_block_mask(self, num_segments_to_mask, acceptable_regions=None):
        tries = 0
        timeout = og_timeout = 20
        valid_mask = False
        while not valid_mask:
            # Chọn ngẫu nhiên các đoạn nhỏ
            indices = torch.randint(0, self.num_segments, (num_segments_to_mask,))
            mask = torch.zeros(self.window_size, dtype=torch.int32)
            for idx in indices:
                start = idx * self.segment_size
                end = start + self.segment_size
                mask[start:end] = 1
            
            # Nếu không cho phép overlap, kiểm tra với acceptable_regions
            if acceptable_regions is not None and not self.allow_overlap:
                for region in acceptable_regions:
                    if region.shape != mask.shape:
                        logger.warning(f"Shape mismatch: region {region.shape}, mask {mask.shape}")
                        region = torch.ones(self.window_size, dtype=torch.int32)  # Fallback
                    mask *= region
            
            mask_indices = torch.nonzero(mask).squeeze()
            valid_mask = len(mask_indices) >= self.min_keep
            if not valid_mask:
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = og_timeout
                    logger.warning(f'Mask generator says: "Valid mask not found, decreasing acceptable-regions [{tries}]"')
        
        mask_complement = torch.ones(self.window_size, dtype=torch.int32)
        mask_complement[mask_indices] = 0
        complement_indices = torch.nonzero(mask_complement).squeeze()
        return mask_indices, mask_complement  # Trả về mask_complement thay vì complement_indices

    def __call__(self, batch):
        B = len(batch)
        collated_batch = torch.utils.data.default_collate(batch)
        
        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        
        num_segments_pred = self._sample_block_size(generator=g, scale=self.pred_mask_scale)
        num_segments_enc = self._sample_block_size(generator=g, scale=self.enc_mask_scale)
        
        collated_masks_pred, collated_masks_enc = [], []
        min_keep_pred = self.window_size
        min_keep_enc = self.window_size
        
        for _ in range(B):
            masks_p, masks_C = [], []
            for _ in range(self.npred):
                mask, mask_C = self._sample_block_mask(num_segments_pred)
                masks_p.append(mask)
                masks_C.append(mask_C)
                min_keep_pred = min(min_keep_pred, len(mask))
            collated_masks_pred.append(masks_p)
            
            acceptable_regions = masks_C if not self.allow_overlap else None
            masks_e = []
            for _ in range(self.nenc):
                mask, _ = self._sample_block_mask(num_segments_enc, acceptable_regions=acceptable_regions)
                masks_e.append(mask)
                min_keep_enc = min(min_keep_enc, len(mask))
            collated_masks_enc.append(masks_e)
        
        collated_masks_pred = [[cm[:min_keep_pred] for cm in cm_list] for cm_list in collated_masks_pred]
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)
        collated_masks_enc = [[cm[:min_keep_enc] for cm in cm_list] for cm_list in collated_masks_enc]
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)
        
        return collated_batch, collated_masks_enc, collated_masks_pred