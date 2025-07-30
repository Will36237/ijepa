
from multiprocessing import Value
from logging import getLogger
import torch
import random

logger = getLogger()

class Random_Mask(object):
    def __init__(
        self,
        ratio=(0.4, 0.6),
        window_size=20,
        segment_size=5
    ):
        super(Random_Mask, self).__init__()
        self.window_size = window_size
        self.segment_size = segment_size
        self.num_segments = window_size // segment_size
        self.ratio = ratio
        self._itr_counter = Value('i', -1)

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def __call__(self, batch, log_freq=100):
        B = len(batch)
        collated_batch = torch.utils.data.default_collate(batch)

        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)

        # Tính số segment cần giữ
        min_keep = int(self.num_segments * (1. - self.ratio[1]))  # 2
        max_keep = int(self.num_segments * (1. - self.ratio[0]))  # 3
        num_keep_segments = random.randint(min_keep, max_keep)  # 2 hoặc 3

        collated_masks_pred, collated_masks_enc = [], []
        for i in range(B):
            indices = torch.randperm(self.num_segments)
            keep_indices = indices[:num_keep_segments]
            mask_indices = indices[num_keep_segments:]

            # Log cho batch đầu hoặc mỗi log_freq iterations
            # if i == 0 and (self._itr_counter.value % log_freq == 0):
            #     logger.info(f"Iter {self._itr_counter.value}, Batch {i}: num_keep_segments={num_keep_segments}, "
            #             f"keep_indices={keep_indices.tolist()}, mask_indices={mask_indices.tolist()}")

            # Tạo mask
            enc_mask = torch.zeros(self.window_size, dtype=torch.int32)
            pred_mask = torch.zeros(self.window_size, dtype=torch.int32)
            for idx in keep_indices:
                start = idx * self.segment_size
                enc_mask[start:start + self.segment_size] = 1
            for idx in mask_indices:
                start = idx * self.segment_size
                pred_mask[start:start + self.segment_size] = 1

            # Log mask sum cho batch đầu
            # if i == 0 and (self._itr_counter.value % log_freq == 0):
            #     logger.info(f"Iter {self._itr_counter.value}, Batch {i}: enc_mask sum={enc_mask.sum().item()}, "
            #             f"pred_mask sum={pred_mask.sum().item()}")

            # Lấy chỉ số
            enc_indices = torch.nonzero(enc_mask).squeeze()
            pred_indices = torch.nonzero(pred_mask).squeeze()

            # Xử lý tensor rỗng hoặc scalar
            if enc_indices.dim() == 0:
                enc_indices = torch.tensor([], dtype=torch.long)
            elif enc_indices.dim() > 1:
                enc_indices = enc_indices.squeeze()
            if pred_indices.dim() == 0:
                pred_indices = torch.tensor([], dtype=torch.long)
            elif pred_indices.dim() > 1:
                pred_indices = pred_indices.squeeze()

            collated_masks_enc.append(enc_indices)
            collated_masks_pred.append(pred_indices)

        # Ghép tensor
        try:
            collated_masks_enc = torch.stack(collated_masks_enc, dim=0).unsqueeze(0)  # [1, B, num_points]
            collated_masks_pred = torch.stack(collated_masks_pred, dim=0).unsqueeze(0)  # [1, B, num_points]
        except Exception as e:
            logger.error(f"Error stacking masks: {e}")
            logger.info(f"collated_masks_enc content: {collated_masks_enc}")
            raise e

        # # Log shape cho batch đầu hoặc mỗi log_freq iterations
        # if self._itr_counter.value % log_freq == 0:
        #     logger.info(f"Iter {self._itr_counter.value}: collated_masks_enc shape={collated_masks_enc.shape}, "
        #             f"values={collated_masks_enc[0, 0, :].tolist()}")
        #     logger.info(f"Iter {self._itr_counter.value}: collated_masks_pred shape={collated_masks_pred.shape}, "
        #             f"values={collated_masks_pred[0, 0, :].tolist()}")

        return collated_batch, collated_masks_enc, collated_masks_pred, num_keep_segments