import os
import time

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchmetrics import functional as fm
from tqdm import tqdm

from config import CONFIG
from datasets.dataset_loader import build_trainloader
from models import build_models
from utils.losses import build_losses
from utils.utils import AverageMeter


SAVE_DIR = f"work_space/ckpts/{CONFIG.MODEL.NAME}"


if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)
    config = CONFIG()

    # Dataset & dataloader
    trainloader, train, sampler = build_trainloader()

    # Model
    model, id_classifier = build_models(config, num_ids=train.num_pids, train=True)
    model = model.cuda()
    id_classifier = id_classifier.cuda()

    # Losses
    criterion_cla, criterion_pair = build_losses(CONFIG)

    # Optimizer
    optimizer = optim.Adam(
        params=list(model.parameters()) + list(id_classifier.parameters()),
        lr=CONFIG.TRAIN.OPTIMIZER.LR,
        weight_decay=CONFIG.TRAIN.OPTIMIZER.WEIGHT_DECAY
    )

    scheduler = lr_scheduler.StepLR(
        optimizer=optimizer,
        step_size=CONFIG.TRAIN.LR_SCHEDULER.STEPSIZE,
        gamma=CONFIG.TRAIN.LR_SCHEDULER.DECAY_RATE
    )

    # Training
    model.train()
    id_classifier.train()

    prev_best_acc = 0.
    prev_best_epoch = -1
    for epoch in range(0, CONFIG.TRAIN.MAX_EPOCH):
        print(f"Epoch: {epoch}")
        sampler.set_epoch(epoch)

        loss_meter = AverageMeter()
        cla_loss_meter = AverageMeter()
        pair_loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        start_time = time.time()
        for clip, pids, _, _ in tqdm(trainloader):
            clip = clip.cuda()
            pids = pids.cuda()

            # pids = pids.repeat_interleave(CONFIG.AUG.SEQ_LEN, dim=0)

            # clip: (b, c, t, h, w)
            # b, c, t, h, w = clip.shape
            # clip = rearrange(clip, 'b c t h w -> (b t) c h w')

            # Forward
            feature = model(clip)
            # feature = rearrange(feature, '(b t) c -> b t c', b=b, t=t)
            # feature = feature.mean(dim=1)
            # feature = model.bn(feature)

            # Classification
            logits = id_classifier(feature)

            # Loss
            loss_cla = criterion_cla(logits, pids)
            loss_pair = criterion_pair(feature, pids)
            loss = loss_cla * CONFIG.LOSS.CLA_LOSS_WEIGHT + loss_pair * config.LOSS.PAIR_LOSS_WEIGHT

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics
            loss_meter.update(loss.item(), pids.size(0))
            cla_loss_meter.update(loss_cla.item(), pids.size(0))
            pair_loss_meter.update(loss_pair.item(), pids.size(0))
            acc = fm.accuracy(logits, pids, 'multiclass', average='macro', num_classes=train.num_pids)
            acc_meter.update(acc.item(), pids.size(0))

        # Print stats
        end_time = time.time()
        print(f"Epoch: {epoch} - loss: {loss_meter.avg:.5f}, cla_loss: {cla_loss_meter.avg:.5f}, pair_loss: {pair_loss_meter.avg:.5f}, acc: {acc_meter.avg:.5f}, time: {end_time - start_time:.2f}s. best_acc: {prev_best_acc:.5f} @ epoch {prev_best_epoch}")
        scheduler.step()

        # Save model
        data = {
            "epoch": epoch,
            "model": model.state_dict(),
            "id_classifier": id_classifier.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict()
        }
        torch.save(data, f"{SAVE_DIR}/latest.pth")

        # Save best model
        if acc_meter.avg > prev_best_acc:
            prev_best_acc = acc_meter.avg
            prev_best_epoch = epoch
            torch.save(data, f"{SAVE_DIR}/best.pth")
            print(f"New best model saved. Acc: {prev_best_acc} @ epoch {prev_best_epoch}")

    print("Training finished.")
