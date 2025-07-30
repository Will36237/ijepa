from multiprocessing import Value
from logging import getLogger
import torch
import random

logger = getLogger()

class Random_Mask(object):
    def __init__(self, ratio=(0.4, 0.6), window_size=20, segment_size=5, future_steps=0):
        super().__init__()
        self.window_size = window_size
        self.segment_size = segment_size
        self.num_segments = window_size // segment_size
        self.ratio = ratio
        self.future_steps = future_steps
        self._itr_counter = Value('i', -1)

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def __call__(self, batch, log_freq=100):
        B = len(batch)
        windows, labels, group_ids = zip(*batch) 
        collated_batch = torch.stack(windows)  
        collated_labels = torch.stack(labels)  
        collated_group_ids = list(group_ids)

        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)

        min_keep = int(self.num_segments * (1. - self.ratio[1]))
        max_keep = int(self.num_segments * (1. - self.ratio[0]))
        num_keep_segments = random.randint(min_keep, max_keep)

        collated_masks_pred, collated_masks_enc = [], []
        for i in range(B):
            indices = torch.randperm(self.num_segments)
            keep_indices = indices[:num_keep_segments]
            mask_indices = indices[num_keep_segments:]

            enc_mask = torch.zeros(self.window_size, dtype=torch.int32)
            pred_mask = torch.zeros(self.window_size, dtype=torch.int32)
            for idx in keep_indices:
                start = idx * self.segment_size
                enc_mask[start:start + self.segment_size] = 1
            for idx in mask_indices:
                start = idx * self.segment_size
                pred_mask[start:start + self.segment_size] = 1

            # Force mask future_steps
            future_start = self.window_size - self.future_steps
            pred_mask[future_start:] = 1
            enc_mask[future_start:] = 0

            enc_indices = torch.nonzero(enc_mask).squeeze()
            pred_indices = torch.nonzero(pred_mask).squeeze()

            if enc_indices.dim() == 0: enc_indices = torch.tensor([], dtype=torch.long)
            if pred_indices.dim() == 0: pred_indices = torch.tensor([], dtype=torch.long)

            collated_masks_enc.append(enc_indices)
            collated_masks_pred.append(pred_indices)

        collated_masks_enc = torch.stack(collated_masks_enc, dim=0).unsqueeze(0)
        collated_masks_pred = torch.stack(collated_masks_pred, dim=0).unsqueeze(0)
        return collated_batch, collated_masks_enc, collated_masks_pred, collated_labels, collated_group_ids
