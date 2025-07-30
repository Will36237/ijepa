# import os
# import copy
# import logging
# import sys
# import yaml
# import tqdm as tqdm 
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from src.models.TimeSeries_transformer import time_series_base, time_series_transformer_predictor
# from src.datasets.TimeSeries_dataloader import make_time_series
# from src.masks.random import Random_Mask
# from src.utils.logging import CSVLogger, gpu_timer, grad_logger, AverageMeter


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
#         num_points=window_size,
#         embed_dim=768,  # Xác nhận từ Ijepa_Params.yaml
#         predictor_embed_dim=384  # Đúng với thiết kế I-JEPA
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
#     criterion = nn.MSELoss()
#     for epoch in range(start_epoch, num_epochs):
#         logger.info(f'Epoch {epoch + 1}/{num_epochs}')
#         loss_meter = AverageMeter()
#         time_meter = AverageMeter()

#         for itr, (window, masks_enc, masks_pred) in enumerate(data_loader):
#             window = window.to(device, non_blocking=True)
#             masks_enc = masks_enc.to(device, non_blocking=True)
#             masks_pred = masks_pred.to(device, non_blocking=True)

#             def train_step():
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

#                 # Forward
#                 with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
#                     h = forward_target()
#                     z = forward_context()
#                     loss = criterion(z, h)

#                 # Backward
#                 if use_bfloat16:
#                     scaler.scale(loss).backward()
#                     scaler.step(optimizer)
#                     scaler.update()
#                 else:
#                     loss.backward()
#                     optimizer.step()

#                 # Momentum update for target encoder
#                 with torch.no_grad():
#                     m = next(momentum_scheduler)
#                     for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
#                         param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

#                 # Scheduler step
#                 scheduler.step()
#                 wd_scheduler.step()

#                 return float(loss)

#             (loss, etime) = gpu_timer(train_step)
#             loss_meter.update(loss)
#             time_meter.update(etime)

#             # Logging
#             if itr % log_freq == 0 or np.isnan(loss) or np.isinf(loss):
#                 csv_logger.log(epoch + 1, itr, loss, etime)
#                 logger.info(f'[{epoch + 1}, {itr:5d}] loss: {loss:.3f} '
#                             f'[mem: {torch.cuda.max_memory_allocated()/1024.**2:.2e} MB] '
#                             f'({time_meter.avg:.1f} ms)')

#             assert not np.isnan(loss), 'Loss is NaN'

#         logger.info(f'Epoch {epoch + 1} avg. loss: {loss_meter.avg:.3f}')
#         save_checkpoint(epoch)

#=================================File train cũ=================================================



import os
import copy
import logging
import sys
import yaml
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.models.TimeSeries_transformer import time_series_base, time_series_transformer_predictor
from src.datasets.TimeSeries_dataloader import make_time_series
from src.masks.random import Random_Mask
from src.utils.logging import CSVLogger, gpu_timer, grad_logger, AverageMeter

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

def main(args, resume_preempt=False):
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG
    # ----------------------------------------------------------------------- #

    # -- META
    use_bfloat16 = args['meta']['use_bfloat16']
    load_checkpoint = args['meta']['load_checkpoint'] or resume_preempt
    checkpoint_path = args['meta']['checkpoint_path']
    copy_data = args['meta']['copy_data']

    # -- DATA
    root_path = args['data']['root_path']
    data_file = args['data']['data_file']
    batch_size = args['data']['batch_size']
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    window_size = args['data']['window_size']
    segment_size = args['data']['segment_size']
    num_features = args['data']['num_features']

    # -- MASK
    ratio = args['mask']['ratio']
    window_size_mask = args['mask']['window_size']
    segment_size_mask = args['mask']['segment_size']

    # -- OPTIMIZATION
    ema = args['optimization']['ema']
    wd = float(args['optimization']['weight_decay'])
    final_wd = float(args['optimization']['final_weight_decay'])
    num_epochs = args['optimization']['epochs']
    warmup = args['optimization']['warmup']
    start_lr = args['optimization']['start_lr']
    lr = args['optimization']['lr']
    final_lr = args['optimization']['final_lr']

    # -- LOGGING
    folder = args['logging']['folder']
    tag = args['logging']['write_tag']
    log_freq = args['logging']['log_freq']
    checkpoint_freq = args['logging']['checkpoint_freq']

    # Thiết bị
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Lưu config
    os.makedirs(folder, exist_ok=True)
    dump = os.path.join(folder, 'params-ijepa.yaml')
    with open(dump, 'w') as f:
        yaml.dump(args, f)

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f'{tag}_train.csv')
    save_path = os.path.join(folder, f'{tag}-ep{{epoch}}.pth.tar')
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    load_path = checkpoint_path if load_checkpoint else None

    # -- make csv_logger
    csv_logger = CSVLogger(
        log_file,
        ('%d', 'epoch'),
        ('%d', 'itr'),
        ('%.5f', 'loss'),
        ('%d', 'time (ms)')
    )

    # -- init dataloader
    collator = Random_Mask(
        ratio=ratio,
        window_size=window_size_mask,
        segment_size=segment_size_mask
    )
    dataset, data_loader, dist_sampler = make_time_series(
        root_path=root_path,
        data_file=data_file,
        window_size=window_size,
        segment_size=segment_size,
        batch_size=batch_size,
        training=True,
        collator=collator,
        pin_mem=pin_mem,
        num_workers=num_workers
    )
    ipe = len(data_loader)
    logger.info(f"Iterations per epoch: {ipe}")

    # -- init model
    encoder = time_series_base(
        window_size=window_size,
        num_features=num_features
    ).to(device)
    predictor = time_series_transformer_predictor(
        num_points=window_size,
        embed_dim=768,
        predictor_embed_dim=384
    ).to(device)
    target_encoder = copy.deepcopy(encoder).to(device)
    for p in target_encoder.parameters():
        p.requires_grad = False

    # -- init optimizer
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=lr,
        weight_decay=wd
    )
    scaler = torch.cuda.amp.GradScaler() if use_bfloat16 else None

    # -- scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs * ipe,
        eta_min=final_lr
    )
    wd_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs * ipe,
        eta_min=final_wd
    )
    momentum_scheduler = (
        ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs)
        for i in range(int(ipe*num_epochs)+1)
    )

    start_epoch = 0
    if load_checkpoint and load_path and os.path.exists(load_path):
        encoder, predictor, target_encoder, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=load_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler
        )
        for _ in range(start_epoch * ipe):
            scheduler.step()
            wd_scheduler.step()
            next(momentum_scheduler)

    def save_checkpoint(epoch):
        save_dict = {
            'encoder': encoder.state_dict(),
            'predictor': predictor.state_dict(),
            'target_encoder': target_encoder.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': scaler.state_dict() if scaler else None,
            'epoch': epoch,
            'loss': loss_meter.avg,
            'batch_size': batch_size,
            'lr': lr
        }
        torch.save(save_dict, latest_path)
        if (epoch + 1) % checkpoint_freq == 0:
            torch.save(save_dict, save_path.format(epoch=f'{epoch + 1}'))
        logger.info(f"Saved checkpoint at epoch {epoch + 1}")

    # -- TRAINING LOOP
    criterion = nn.MSELoss()
    for epoch in tqdm(range(start_epoch, num_epochs), desc="Training", leave=True):
        logger.info(f'Epoch {epoch + 1}/{num_epochs}')
        loss_meter = AverageMeter()
        time_meter = AverageMeter()
        keep_segments_counter = {2: 0, 3: 0}  # Đếm num_keep_segments

        progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for itr, (window, masks_enc, masks_pred, num_keep_segments) in progress_bar:
            window = window.to(device, non_blocking=True)
            masks_enc = masks_enc.to(device, non_blocking=True)
            masks_pred = masks_pred.to(device, non_blocking=True)

            def train_step():
                optimizer.zero_grad()

                def forward_target():
                    with torch.no_grad():
                        # if itr % 100 == 0 or np.isnan(loss_meter.avg) or np.isinf(loss_meter.avg):
                        #     logger.info(f"Window shape: {window.shape}")
                        #     logger.info(f"masks_pred shape: {masks_pred.shape}, values: {masks_pred[0, 0, :]}")
                        h = target_encoder(window, masks=masks_pred)
                        # if itr % 100 == 0 or np.isnan(loss_meter.avg) or np.isinf(loss_meter.avg):
                        #     logger.info(f"Target encoder output shape: {h.shape}")
                        return h

                def forward_context():
                    # if itr % 100 == 0 or np.isnan(loss_meter.avg) or np.isinf(loss_meter.avg):
                    #     logger.info(f"masks_enc shape: {masks_enc.shape}, values: {masks_enc[0, 0, :]}")
                    z = encoder(window, masks=masks_enc)
                    # if itr % 100 == 0 or np.isnan(loss_meter.avg) or np.isinf(loss_meter.avg):
                    #     logger.info(f"Encoder output shape: {z.shape}")
                    z = predictor(z, masks_x=masks_enc, masks=masks_pred)
                    # if itr % 100 == 0 or np.isnan(loss_meter.avg) or np.isinf(loss_meter.avg):
                    #     logger.info(f"Predictor output shape: {z.shape}")
                    return z

                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                    h = forward_target()
                    z = forward_context()
                    loss = criterion(z, h)

                if use_bfloat16:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                with torch.no_grad():
                    m = next(momentum_scheduler)
                    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                        param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

                scheduler.step()
                wd_scheduler.step()

                return float(loss)

            (loss, etime) = gpu_timer(train_step)
            loss_meter.update(loss)
            time_meter.update(etime)

            # Cập nhật progress bar
            lr = scheduler.get_last_lr()[0]
            mem = torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0
            progress_bar.set_postfix({
                "loss": f"{loss:.4f}",
                "lr": f"{lr:.6f}",
                "mem": f"{mem:.1f}MB"
            })

            # Log to CSV
            if itr % log_freq == 0 or np.isnan(loss) or np.isinf(loss):
                csv_logger.log(epoch + 1, itr, loss, etime)

            # Log lỗi nếu có
            if np.isnan(loss) or np.isinf(loss):
                logger.error(f"Loss is {'NaN' if np.isnan(loss) else 'Inf'} at epoch {epoch+1}, batch {itr}")
                raise ValueError(f"Loss is {'NaN' if np.isnan(loss) else 'Inf'}")

            # Đếm num_keep_segments (dựa trên masks_enc shape)
            keep_segments_counter[num_keep_segments] = keep_segments_counter.get(num_keep_segments, 0) + 1

        # Log epoch summary
        avg_loss = loss_meter.avg
        logger.info(f'Epoch {epoch + 1} avg. loss: {avg_loss:.4f}, time: {time_meter.avg:.1f} ms')
        logger.info(f'num_keep_segments: 1={keep_segments_counter[1]}, num_keep_segments: 2={keep_segments_counter[2]}, 3={keep_segments_counter[3]}')
        progress_bar.set_postfix({"avg_loss": f"{avg_loss:.4f}"})
        save_checkpoint(epoch)

    return encoder, target_encoder, predictor

if __name__ == "__main__":
    with open("Ijepa_Params.yaml", 'r') as stream:
        args = yaml.safe_load(stream)
    main(args)