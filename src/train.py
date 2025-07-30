import os
import copy
import logging
import sys
import yaml
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from src.models.TimeSeries_transformer import time_series_base, time_series_transformer_predictor
from src.datasets.TimeSeries_dataloader import make_time_series
from src.masks.random import Random_Mask
from src.utils.logging import CSVLogger, gpu_timer, AverageMeter
from src.utils.distributed import init_distributed, AllReduceSum
from src.helper import load_checkpoint, init_opt  # Import từ helper.py
import pickle

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

def save_checkpoint(epoch, encoder, predictor, target_encoder, class_head, optimizer, scaler, loss_meter, batch_size, lr, latest_path, save_path, checkpoint_freq):
    save_dict = {
        'encoder': encoder.state_dict(),
        'predictor': predictor.state_dict(),
        'target_encoder': target_encoder.state_dict(),
        'class_head': class_head.state_dict(),
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

def main(args, resume_preempt=False):
    # Load all from YAML
    use_bfloat16 = args['meta']['use_bfloat16']
    load_checkpoint_flag = args['meta']['load_checkpoint'] or resume_preempt
    checkpoint_path = args['meta']['checkpoint_path']
    # Data params
    dataset_name = args['data']['dataset_name']
    batch_size = args['data']['batch_size']
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    window_size = args['data']['window_size']
    segment_size = args['data']['segment_size']
    num_features = args['data']['num_features']
    future_steps = args['data']['future_steps']
    # Mask params
    ratio = args['mask']['ratio']
    window_size_mask = args['mask']['window_size']
    segment_size_mask = args['mask']['segment_size']
    # Opt params
    ema = args['optimization']['ema']
    wd = float(args['optimization']['weight_decay'])
    final_wd = float(args['optimization']['final_weight_decay'])
    num_epochs = args['optimization']['epochs']
    warmup = args['optimization']['warmup']
    start_lr = args['optimization']['start_lr']
    lr = args['optimization']['lr']
    final_lr = args['optimization']['final_lr']
    # Logging
    folder = args['logging']['folder']
    tag = args['logging']['write_tag']
    log_freq = args['logging']['log_freq']
    checkpoint_freq = args['logging']['checkpoint_freq']
    output_path = args['logging']['output_path']
    # Model
    num_classes = args['model']['num_classes']

    # Distributed for 2 GPUs
    world_size, rank = init_distributed(port=args['distributed']['port'])
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(rank)
    logger.info(f"Using device: {device}, rank: {rank}, world_size: {world_size}")

    os.makedirs(folder, exist_ok=True)
    dump = os.path.join(folder, 'params-ijepa.yaml')
    with open(dump, 'w') as f:
        yaml.dump(args, f)

    log_file = os.path.join(folder, f'{tag}_train.csv')
    save_path = os.path.join(folder, f'{tag}-ep{{epoch}}.pth.tar')
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    load_path = checkpoint_path if load_checkpoint_flag else None

    csv_logger = CSVLogger(log_file, ('%d', 'epoch'), ('%d', 'itr'), ('%.5f', 'loss'), ('%d', 'time (ms)'))

    # Init dataloader
    collator = Random_Mask(ratio=ratio, window_size=window_size_mask, segment_size=segment_size_mask, future_steps=future_steps)
    dataset, data_loader, dist_sampler = make_time_series(
        dataset_name=dataset_name,
        window_size=window_size,
        segment_size=segment_size,
        batch_size=batch_size // world_size,
        training=True,
        collator=collator,
        pin_mem=pin_mem,
        num_workers=num_workers,
        world_size=world_size,
        rank=rank,
        split_ratios=args['data']['split_ratios'],
        features_list=args['data']['features_list'],
        future_steps=future_steps,
        num_features=num_features
    )
    ipe = len(data_loader)

    # Init model (ở đây thay vì init_model từ helper)
    encoder = time_series_base(window_size=window_size, num_features=num_features).to(device)
    predictor = time_series_transformer_predictor(num_points=window_size, embed_dim=768, predictor_embed_dim=384).to(device)
    target_encoder = copy.deepcopy(encoder).to(device)
    for p in target_encoder.parameters():
        p.requires_grad = False
    class_head = nn.Linear(768, num_classes).to(device)  # Classification head

    # Init opt từ helper.py
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        class_head=class_head,
        iterations_per_epoch=ipe,
        start_lr=start_lr,
        ref_lr=lr,
        warmup=warmup,
        num_epochs=num_epochs,
        wd=wd,
        final_wd=final_wd,
        final_lr=final_lr,
        use_bfloat16=use_bfloat16
    )

    # Momentum scheduler (giữ ở train.py vì không phải helper chung)
    momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs) for i in range(int(ipe*num_epochs)+1))

    start_epoch = 0
    if load_checkpoint_flag and load_path and os.path.exists(load_path):
        encoder, predictor, target_encoder, optimizer, scaler, start_epoch, class_head = load_checkpoint(
            device, load_path, encoder, predictor, target_encoder, optimizer, scaler, class_head
        )
        for _ in range(start_epoch * ipe):
            scheduler.step()
            wd_scheduler.step()
            next(momentum_scheduler)

    criterion_mse = nn.MSELoss()
    criterion_ce = nn.CrossEntropyLoss()

    output_weights = {}  # Collect probs per symbol_timeframe

    for epoch in tqdm(range(start_epoch, num_epochs), desc="Training", leave=True):
        loss_meter = AverageMeter()
        time_meter = AverageMeter()

        progress_bar = tqdm(enumerate(data_loader), total=ipe, desc=f"Epoch {epoch+1}", leave=False)
        for itr, (window, masks_enc, masks_pred, labels, group_ids) in progress_bar:
            window = window.to(device, non_blocking=True)
            masks_enc = masks_enc.to(device, non_blocking=True)
            masks_pred = masks_pred.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)  # Assume [batch, future_steps] class indices (0,1,2)

            def train_step():
                optimizer.zero_grad()

                def forward_target():
                    with torch.no_grad():
                        h = target_encoder(window, masks=masks_pred)
                        return h

                def forward_context():
                    z = encoder(window, masks=masks_enc)
                    z = predictor(z, masks_x=masks_enc, masks=masks_pred)
                    return z

                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                    h = forward_target()
                    z = forward_context()
                    loss_mse = criterion_mse(z, h)
                    # Predict per future step
                    preds = class_head(z[:, -future_steps:, :].mean(dim=-1))  # [batch, future_steps, num_classes]
                    loss_ce = criterion_ce(preds.view(-1, num_classes), labels.view(-1))  # Flatten
                    loss = loss_mse + loss_ce

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

                return float(loss), preds

            ((loss, preds), etime) = gpu_timer(train_step)
            loss = AllReduceSum.apply(loss) / world_size  # Sync loss across GPUs
            loss_meter.update(loss)
            time_meter.update(etime)

            # Collect outputs (only on rank 0)
            if rank == 0:
                probs = nn.Softmax(dim=-1)(preds)
                for gid, prob in zip(group_ids, probs):
                    output_weights[gid] = prob.detach().cpu().numpy()

            if itr % log_freq == 0:
                csv_logger.log(epoch + 1, itr, loss, etime)

        if rank == 0:
            logger.info(f'Epoch {epoch + 1} avg. loss: {loss_meter.avg:.4f}, time: {time_meter.avg:.1f} ms')
            save_checkpoint(epoch, encoder, predictor, target_encoder, class_head, optimizer, scaler, loss_meter, batch_size, lr, latest_path, save_path, checkpoint_freq)
            with open(output_path, 'wb') as f:
                pickle.dump(output_weights, f)

    return encoder, target_encoder, predictor

if __name__ == "__main__":
    with open("Ijepa_Params.yaml", 'r') as stream:
        args = yaml.safe_load(stream)
    main(args)
