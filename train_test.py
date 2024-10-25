import logging
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torchmetrics import functional as fm
from tqdm import tqdm

from config import CONFIG
from datasets.dataset_loader import build_trainloader, build_testloader
from models import build_models
from datasets.samplers import RandomIdentitySampler
from utils.eval_metrics import evaluate, evaluate_with_clothes
from utils.evaluate import extract_vid_feature
from utils.losses import build_losses
from utils.utils import AverageMeter


config = CONFIG()
SAVE_DIR = f"work_space/ckpts/{config.DATA.DATASET}/resnet50_attn_stride_1_bn_cal"
os.makedirs(SAVE_DIR, exist_ok=True)
logger = logging.getLogger()
fh = logging.FileHandler(os.path.join(SAVE_DIR, 'train.log'))
logger.addHandler(fh)
logger.setLevel(logging.INFO)
fh.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def train_epoch(epoch: int,
                model: nn.Module,
                id_classifier: nn.Module,
                clothes_classifier: nn.Module,
                sampler: RandomIdentitySampler,
                trainloader: DataLoader,
                criterion_cla: nn.Module,
                criterion_pair: nn.Module,
                criterion_clothes: nn.Module,
                criterion_adv: nn.Module,
                optimizer: optim.Adam,
                optimizer_cc: optim.Adam,
                scheduler: lr_scheduler.StepLR,
                pid2clothes: torch.Tensor):
    logger.info("Training")
    sampler.set_epoch(epoch)

    model.train()
    id_classifier.train()
    clothes_classifier.train()

    # For logging
    loss_meter = AverageMeter()
    cla_loss_meter = AverageMeter()
    pair_loss_meter = AverageMeter()
    adv_loss_meter = AverageMeter()
    id_acc_meter = AverageMeter()
    clothes_acc_meter = AverageMeter()

    start_time = time.time()
    for clip, pids, camids, clothes_ids in tqdm(trainloader):
        clip = clip.cuda()
        pids = pids.cuda()
        clothes_ids = clothes_ids.cuda()

        # Get all positive clothes classes (belonging to the same identity) for each sample
        clothes_pos_mask = pid2clothes[pids].float().cuda()

        # Forward
        feature = model(clip)

        # Classification
        id_logits = id_classifier(feature)  # ID classification
        # Clothes classification.
        clothes_logits = clothes_classifier(feature.detach())  # Detach to update only clothes classifier

        # Update clothes classifier
        clothes_loss = criterion_clothes(clothes_logits, clothes_ids)
        if epoch >= 25:
            optimizer_cc.zero_grad()
            clothes_loss.backward()
            optimizer_cc.step()

        # Update the rest
        clothes_logits = clothes_classifier(feature)

        # Loss
        loss_cla = criterion_cla(id_logits, pids)
        loss_pair = criterion_pair(feature, pids)
        if epoch >= 25:
            loss_adv = criterion_adv(clothes_logits, clothes_ids, clothes_pos_mask)
        else:
            loss_adv = torch.tensor(0.)

        if epoch >= 25:
            loss = loss_cla * CONFIG.LOSS.CLA_LOSS_WEIGHT + loss_pair * CONFIG.LOSS.PAIR_LOSS_WEIGHT + loss_adv
        else:
            loss = loss_cla * CONFIG.LOSS.CLA_LOSS_WEIGHT + loss_pair * CONFIG.LOSS.PAIR_LOSS_WEIGHT

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        loss_meter.update(loss.item(), pids.size(0))
        cla_loss_meter.update(loss_cla.item(), pids.size(0))
        pair_loss_meter.update(loss_pair.item(), pids.size(0))
        adv_loss_meter.update(loss_adv.item(), pids.size(0))
        id_acc = fm.accuracy(id_logits, pids, 'multiclass', average='macro', num_classes=train.num_pids)
        clothes_acc = fm.accuracy(clothes_logits, clothes_ids, 'multiclass', average='macro', num_classes=train.num_clothes)
        id_acc_meter.update(id_acc.item(), pids.size(0))
        clothes_acc_meter.update(clothes_acc.item(), pids.size(0))

    # Print stats
    end_time = time.time()
    logger.info(f"Epoch {epoch} - lr {scheduler.get_last_lr()}, "
                f"loss {loss_meter.avg:.5f}, "
                f"cla_loss {cla_loss_meter.avg:.5f}, "
                f"pair_loss {pair_loss_meter.avg:.5f}, "
                f"adv_loss {adv_loss_meter.avg:.5f}, "
                f"acc {id_acc_meter.avg:.5f}, "
                f"epoch time {end_time - start_time:.2f}s.")
    scheduler.step()

    return id_acc_meter.avg


def test_epoch(model, queryloader, galleryloader, query, gallery):
    logger.info("Testing")
    model.eval()

    query_features, query_pids, query_camids, query_clothes_ids = extract_vid_feature(
        model=model,
        dataloader=queryloader,
        vid2clip_index=query.vid2clip_index,
        data_length=len(query.dataset),
    )
    gallery_features, gallery_pids, gallery_camids, gallery_clothes_ids = extract_vid_feature(
        model,
        galleryloader,
        vid2clip_index=gallery.vid2clip_index,
        data_length=len(gallery.dataset),
    )

    torch.cuda.empty_cache()

    m, n = query_features.size(0), gallery_features.size(0)
    distance_matrix = torch.zeros((m, n))
    query_features, gallery_features = query_features.cuda(), gallery_features.cuda()
    # Cosine similarity
    for i in range(m):
        distance_matrix[i] = (-torch.mm(query_features[i:i + 1], gallery_features.t())).cpu()
        # distance_matrix[i] = (-torch.mm(query_features[i:i + 1], gallery_features.t()) / (torch.norm(query_features[i])*torch.norm(gallery_features, dim=1))).cpu()
    distance_matrix = distance_matrix.numpy()
    query_pids, query_camids, query_clothes_ids = query_pids.numpy(),\
          query_camids.numpy(), query_clothes_ids.numpy()
    gallery_pids, gallery_camids, gallery_clothes_ids = gallery_pids.numpy(),\
          gallery_camids.numpy(), gallery_clothes_ids.numpy()

    standard_cmc, standard_mAP = evaluate(distance_matrix, query_pids, gallery_pids,
                        query_camids, gallery_camids)
    sc_cmc, sc_mAP = evaluate_with_clothes(distance_matrix,
                                    query_pids,
                                    gallery_pids,
                                    query_camids,
                                    gallery_camids,
                                    query_clothes_ids,
                                    gallery_clothes_ids,
                                    mode='SC')

    cc_cmc, cc_mAP = evaluate_with_clothes(distance_matrix,
                                    query_pids,
                                    gallery_pids,
                                    query_camids,
                                    gallery_camids,
                                    query_clothes_ids,
                                    gallery_clothes_ids,
                                    mode='CC')

    return standard_cmc*100, standard_mAP*100, sc_cmc*100, sc_mAP*100, cc_cmc*100, cc_mAP*100


if __name__ == "__main__":
    # Dataset & dataloader
    trainloader, train, sampler = build_trainloader()
    queryloader, galleryloader, query, gallery = build_testloader()
    pid2clothes = torch.from_numpy(train.pid2clothes).cuda()

    # Model
    model, id_classifier, clothes_classifier = build_models(config, num_ids=train.num_pids, num_clothes=train.num_clothes, train=True)
    model = model.cuda()
    id_classifier = id_classifier.cuda()
    clothes_classifier = clothes_classifier.cuda()

    # Losses
    criterion_cla, criterion_pair, criterion_clothes, criterion_cal = build_losses(CONFIG, train.num_clothes)

    # Optimizer
    optimizer = optim.Adam(
        params=list(model.parameters()) + list(id_classifier.parameters()),
        lr=CONFIG.TRAIN.OPTIMIZER.LR,
        weight_decay=CONFIG.TRAIN.OPTIMIZER.WEIGHT_DECAY
    )
    optimizer_cc = optim.Adam(
        params=list(clothes_classifier.parameters()),
        lr=CONFIG.TRAIN.OPTIMIZER.LR,
        weight_decay=CONFIG.TRAIN.OPTIMIZER.WEIGHT_DECAY
    )

    # LR Scheduler
    scheduler = lr_scheduler.StepLR(
        optimizer=optimizer,
        step_size=CONFIG.TRAIN.LR_SCHEDULER.STEPSIZE,
        gamma=CONFIG.TRAIN.LR_SCHEDULER.DECAY_RATE
    )

    prev_best_acc = 0.
    prev_best_cc_r1 = 0.
    prev_best_cc_map = 0.

    for epoch in range(0, CONFIG.TRAIN.MAX_EPOCH):
        logger.info(f"Epoch: {epoch}")

        train_acc = train_epoch(epoch,
                                model,
                                id_classifier,
                                clothes_classifier,
                                sampler,
                                trainloader,
                                criterion_cla,
                                criterion_pair,
                                criterion_clothes,
                                criterion_cal,
                                optimizer,
                                optimizer_cc,
                                scheduler,
                                pid2clothes)
        torch.cuda.empty_cache()

        standard_cmc, standard_mAP, sc_cmc, sc_mAP, cc_cmc, cc_mAP = test_epoch(model, queryloader, galleryloader, query, gallery)
        logger.info("==============================")
        sc_results = f"Same Clothes | R-1: {sc_cmc[0]:.1f} | R-5: {sc_cmc[4]:.1f} | R-10: {sc_cmc[9]:.1f} | mAP: {sc_mAP:.1f}"
        logger.info(sc_results)
        standard_results = f"Standard | R-1: {standard_cmc[0]:.1f} | R-5: {standard_cmc[4]:.1f} | R-10: {standard_cmc[9]:.1f} | mAP: {standard_mAP:.1f}"
        logger.info(standard_results)
        cc_results = f"Cloth-changing | R-1: {cc_cmc[0]:.1f} | R-5: {cc_cmc[4]:.1f} | R-10: {cc_cmc[9]:.1f} | mAP: {cc_mAP:.1f}"
        logger.info(cc_results)
        logger.info("==============================")

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
        if train_acc > prev_best_acc:
            prev_best_acc = train_acc
            torch.save(data, f"{SAVE_DIR}/best.pth")
            logger.info(f"New best model saved. Acc: {prev_best_acc} @ epoch {epoch}")

        # Save best model for cloth-changing setting
        if cc_cmc[0] > prev_best_cc_r1:
            prev_best_cc_r1 = cc_cmc[0]
            torch.save(data, f"{SAVE_DIR}/best_cc_r1.pth")
            logger.info(f"New best model saved for cloth-changing setting. R-1: {prev_best_cc_r1} @ epoch {epoch}")

        if cc_mAP > prev_best_cc_map:
            prev_best_cc_map = cc_mAP
            torch.save(data, f"{SAVE_DIR}/best_cc_map.pth")
            logger.info(f"New best model saved for cloth-changing setting. mAP: {prev_best_cc_map} @ epoch {epoch}")

    logger.info("Training finished.")
