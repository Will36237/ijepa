from src.models.TimeSeries_transformer import time_series_base, time_series_transformer_predictor
from src.datasets.TimeSeries_dataloader import make_time_series
from src.masks.random import Random_Mask
import torch
import torch.nn as nn
import numpy as np
from logging import getLogger

logger = getLogger()

# ✅ Khởi tạo dataloader
collator = Random_Mask(window_size=20, segment_size=5)
dataset, data_loader, dist_sampler = make_time_series(
    root_path="./data",
    data_file="trading/XAUUSD_M15.csv",
    window_size=20,
    segment_size=5,
    batch_size=32,
    training=True,
    copy_data=False,
    collator=collator
)

# ✅ Khởi tạo mô hình
context_encoder = time_series_base(window_size=20, num_features=6)
predictor = time_series_transformer_predictor(num_points=20)

# ✅ Hàm debug shape
def debug_tensor(name, tensor):
    print(f"{name}: type={type(tensor)}, shape={tensor.shape if hasattr(tensor, 'shape') else 'n/a'}, dtype={getattr(tensor, 'dtype', 'n/a')}")

# ✅ Hàm kiểm tra giá trị NaN
def check_nan(name, tensor):
    if torch.isnan(tensor).any():
        print(f"⚠️ NaN detected in {name}!")
    else:
        print(f"{name} OK (no NaNs)")

# ✅ Duyệt batch
for batch in data_loader:
    window, masks_enc, masks_pred = batch

    # ✅ Debug input
    debug_tensor("window", window)
    debug_tensor("masks_enc", masks_enc)
    debug_tensor("masks_pred", masks_pred)

    # ✅ Context encoder
    context = context_encoder(window, masks=masks_enc)
    debug_tensor("context (output of encoder)", context)

    # ✅ Predictor
    pred = predictor(context, masks_x=masks_enc, masks=masks_pred)
    debug_tensor("pred (output of predictor)", pred)

    # ✅ Target
    target = context_encoder(window, masks=masks_pred)
    debug_tensor("target", target)

    # ✅ Kiểm tra NaN
    check_nan("pred", pred)
    check_nan("target", target)

    # ✅ Loss
    try:
        loss_fn = nn.MSELoss()
        loss = loss_fn(pred, target)
        print(f"✅ Loss: {loss.item()}")
    except Exception as e:
        print(f"❌ Lỗi khi tính loss: {e}")
        logger.error(f"pred shape: {pred.shape}, target shape: {target.shape}")
        logger.error(f"pred min: {pred.min().item()}, max: {pred.max().item()}")
        logger.error(f"target min: {target.min().item()}, max: {target.max().item()}")

    break  # chỉ debug 1 batch





# file train backup (cũ)
# # Copyright (c) Meta Platforms, Inc. and affiliates.
# # All rights reserved.
# #
# # This source code is licensed under the license found in the
# # LICENSE file in the root directory of this source tree.
# #

# import os
# import copy
# import logging
# import sys
# import yaml
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from src.models.TimeSeries_transformer import time_series_base, time_series_transformer_predictor
# from src.datasets.TimeSeries_dataloader import make_time_series
# from src.masks.random import Random_Mask
# from src.masks.utils import apply_masks
# from src.utils.logging import CSVLogger, gpu_timer, grad_logger, AverageMeter
# from src.utils.tensors import repeat_interleave_batch
# from src.helper import load_checkpoint

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logger = logging.getLogger()

# def main(args, resume_preempt=False):
#     # ----------------------------------------------------------------------- #
#     #  PASSED IN PARAMS FROM CONFIG
#     # ----------------------------------------------------------------------- #

#     # -- META
#     use_bfloat16 = args['meta']['use_bfloat16']
#     load_checkpoint = args['meta']['load_checkpoint'] or resume_preempt
#     checkpoint_path = args['meta']['checkpoint_path']
#     copy_data = args['meta']['copy_data']

#     # -- DATA
#     root_path = args['data']['root_path']
#     data_file = args['data']['data_file']
#     batch_size = args['data']['batch_size']
#     pin_mem = args['data']['pin_mem']
#     num_workers = args['data']['num_workers']
#     window_size = args['data']['window_size']
#     segment_size = args['data']['segment_size']
#     num_features = args['data']['num_features']

#     # -- MASK
#     ratio = args['mask']['ratio']
#     window_size_mask = args['mask']['window_size']
#     segment_size_mask = args['mask']['segment_size']

#     # -- OPTIMIZATION
#     ema = args['optimization']['ema']
#     wd = float(args['optimization']['weight_decay'])
#     final_wd = float(args['optimization']['final_weight_decay'])
#     num_epochs = args['optimization']['epochs']
#     warmup = args['optimization']['warmup']
#     start_lr = args['optimization']['start_lr']
#     lr = args['optimization']['lr']
#     final_lr = args['optimization']['final_lr']

#     # -- LOGGING
#     folder = args['logging']['folder']
#     tag = args['logging']['write_tag']
#     log_freq = args['logging']['log_freq']
#     checkpoint_freq = args['logging']['checkpoint_freq']

#     # Thiết bị
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     logger.info(f"Using device: {device}")

#     # Lưu config
#     os.makedirs(folder, exist_ok=True)
#     dump = os.path.join(folder, 'params-ijepa.yaml')
#     with open(dump, 'w') as f:
#         yaml.dump(args, f)

#     # -- log/checkpointing paths
#     log_file = os.path.join(folder, f'{tag}_train.csv')
#     save_path = os.path.join(folder, f'{tag}-ep{{epoch}}.pth.tar')
#     latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
#     load_path = checkpoint_path if load_checkpoint else None

#     # -- make csv_logger
#     csv_logger = CSVLogger(
#         log_file,
#         ('%d', 'epoch'),
#         ('%d', 'itr'),
#         ('%.5f', 'loss'),
#         ('%d', 'time (ms)')
#     )

#     # -- init dataloader
#     collator = Random_Mask(
#         ratio=ratio,
#         window_size=window_size_mask,
#         segment_size=segment_size_mask
#     )
#     dataset, data_loader, dist_sampler = make_time_series(
#         root_path=root_path,
#         data_file=data_file,
#         window_size=window_size,
#         segment_size=segment_size,
#         batch_size=batch_size,
#         training=True,
#         collator=collator,
#         pin_mem=pin_mem,
#         num_workers=num_workers
#     )
#     ipe = len(data_loader)
#     logger.info(f"Iterations per epoch: {ipe}")

#     # -- init model
#     encoder = time_series_base(
#         window_size=window_size,
#         num_features=num_features
#     ).to(device)
#     predictor = time_series_transformer_predictor(
#         num_points=window_size
#     ).to(device)
#     target_encoder = copy.deepcopy(encoder).to(device)
#     for p in target_encoder.parameters():
#         p.requires_grad = False

#     # -- init optimizer
#     optimizer = optim.Adam(
#         list(encoder.parameters()) + list(predictor.parameters()),
#         lr=lr,
#         weight_decay=wd
#     )
#     scaler = torch.cuda.amp.GradScaler() if use_bfloat16 else None

#     # -- scheduler
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(
#         optimizer,
#         T_max=num_epochs * ipe,
#         eta_min=final_lr
#     )
#     wd_scheduler = optim.lr_scheduler.CosineAnnealingLR(
#         optimizer,
#         T_max=num_epochs * ipe,
#         eta_min=final_wd
#     )
#     momentum_scheduler = (
#         ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs)
#         for i in range(int(ipe*num_epochs)+1)
#     )

#     start_epoch = 0
#     # -- load training checkpoint
#     if load_checkpoint and load_path and os.path.exists(load_path):
#         encoder, predictor, target_encoder, optimizer, scaler, start_epoch = load_checkpoint(
#             device=device,
#             r_path=load_path,
#             encoder=encoder,
#             predictor=predictor,
#             target_encoder=target_encoder,
#             opt=optimizer,
#             scaler=scaler
#         )
#         for _ in range(start_epoch * ipe):
#             scheduler.step()
#             wd_scheduler.step()
#             next(momentum_scheduler)

#     def save_checkpoint(epoch):
#         save_dict = {
#             'encoder': encoder.state_dict(),
#             'predictor': predictor.state_dict(),
#             'target_encoder': target_encoder.state_dict(),
#             'opt': optimizer.state_dict(),
#             'scaler': scaler.state_dict() if scaler else None,
#             'epoch': epoch,
#             'loss': loss_meter.avg,
#             'batch_size': batch_size,
#             'lr': lr
#         }
#         torch.save(save_dict, latest_path)
#         if (epoch + 1) % checkpoint_freq == 0:
#             torch.save(save_dict, save_path.format(epoch=f'{epoch + 1}'))
#         logger.info(f"Saved checkpoint at epoch {epoch + 1}")

#     # -- TRAINING LOOP
#     for epoch in range(start_epoch, num_epochs):
#         logger.info(f'Epoch {epoch + 1}/{num_epochs}')
#         loss_meter = AverageMeter()
#         time_meter = AverageMeter()

#         for itr, (window, masks_enc, masks_pred) in enumerate(data_loader):
#             window = window.to(device, non_blocking=True)
#             masks_enc = masks_enc.to(device, non_blocking=True)
#             masks_pred = masks_pred.to(device, non_blocking=True)

#             def train_step():
#                 scheduler.step()
#                 wd_scheduler.step()
#                 optimizer.zero_grad()

#                 def forward_target():
#                     with torch.no_grad():
#                         logger.info(f"Window shape: {window.shape}")
#                         h = target_encoder(window, masks=masks_pred)
#                         logger.info(f"Target encoder output shape: {h.shape}")
#                         return h

#                 def forward_context():
#                     z = encoder(window, masks=masks_enc)
#                     logger.info(f"Encoder output shape: {z.shape}")
#                     z = predictor(z, masks_x=masks_enc, masks=masks_pred)
#                     logger.info(f"Predictor output shape: {z.shape}")
#                     return z

#                 def loss_fn(z, h):
#                     loss = nn.MSELoss()(z, h)
#                     return loss

#                 # Forward
#                 with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
#                     h = forward_target()
#                     z = forward_context()
#                     loss = loss_fn(z, h)

#                 # Backward
#                 if use_bfloat16:
#                     scaler.scale(loss).backward()
#                     scaler.step(optimizer)
#                     scaler.update()
#                 else:
#                     loss.backward()
#                     optimizer.step()

#                 # Momentum update
#                 with torch.no_grad():
#                     m = next(momentum_scheduler)
#                     for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
#                         param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

#                 return float(loss)

#             (loss, etime) = gpu_timer(train_step)
#             loss_meter.update(loss)
#             time_meter.update(etime)

#             # Logging
#             if itr % log_freq == 0 or np.isnan(loss) or np.isinf(loss):
#                 csv_logger.log(epoch + 1, itr, loss, etime)
#                 logger.info(f'[{epoch + 1}, {itr:5d}] loss: {loss_meter.avg:.3f} '
#                             f'[mem: {torch.cuda.max_memory_allocated()/1024.**2:.2e} MB] '
#                             f'({time_meter.avg:.1f} ms)')

#             assert not np.isnan(loss), 'Loss is NaN'

#         logger.info(f'Epoch {epoch + 1} avg. loss: {loss_meter.avg:.3f}')
#         save_checkpoint(epoch + 1)


