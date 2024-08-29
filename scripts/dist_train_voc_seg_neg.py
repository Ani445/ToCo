import argparse
import datetime
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from datasets import voc as voc
from model.losses import get_masked_ptc_loss, get_seg_loss, CTCLoss_neg, DenseEnergyLoss, get_energy_loss
from model.model_seg_neg import network
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from model.PAR import PAR
from utils import evaluate, imutils, optimizer
from utils.camutils import cam_to_label, cam_to_roi_mask2, multi_scale_cam2, label_to_aff_mask, refine_cams_with_bkg_v2, crop_from_roi_neg
from utils.pyutils import AverageMeter, cal_eta, format_tabs, setup_logger

torch.hub.set_dir("./pretrained")
parser = argparse.ArgumentParser()

# (Your argument definitions...)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def validate(model=None, data_loader=None, args=None, device=None):
    preds, gts, cams, cams_aux = [], [], [], []
    model.eval()
    avg_meter = AverageMeter()
    with torch.no_grad():
        for _, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >="):
            name, inputs, labels, cls_label = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            cls_label = cls_label.to(device)

            inputs = F.interpolate(inputs, size=[args.crop_size, args.crop_size], mode='bilinear', align_corners=False)
            cls, segs, _, _ = model(inputs)

            cls_pred = (cls > 0).type(torch.int16)
            _f1 = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])
            avg_meter.add({"cls_score": _f1})

            _cams, _cams_aux = multi_scale_cam2(model, inputs, args.cam_scales)
            resized_cam = F.interpolate(_cams, size=labels.shape[1:], mode='bilinear', align_corners=False)
            cam_label = cam_to_label(resized_cam, cls_label, bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index)

            resized_cam_aux = F.interpolate(_cams_aux, size=labels.shape[1:], mode='bilinear', align_corners=False)
            cam_label_aux = cam_to_label(resized_cam_aux, cls_label, bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index)

            preds += list(torch.argmax(resized_segs, dim=1).cpu().numpy().astype(np.int16))
            cams += list(cam_label.cpu().numpy().astype(np.int16))
            gts += list(labels.cpu().numpy().astype(np.int16))
            cams_aux += list(cam_label_aux.cpu().numpy().astype(np.int16))

    cls_score = avg_meter.pop('cls_score')
    seg_score = evaluate.scores(gts, preds)
    cam_score = evaluate.scores(gts, cams)
    cam_aux_score = evaluate.scores(gts, cams_aux)
    model.train()

    tab_results = format_tabs([cam_score, cam_aux_score, seg_score], name_list=["CAM", "aux_CAM", "Seg_Pred"], cat_list=voc.class_list)
    return cls_score, tab_results

def train(args=None):
    # Determine if a GPU is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        dist.init_process_group(backend=args.backend)
        logging.info("Total GPUs: %d, samples per GPU: %d..." % (dist.get_world_size(), args.spg))
    else:
        logging.info("Using CPU...")

    time0 = datetime.datetime.now().replace(microsecond=0)

    # (Dataset loading and preprocessing code...)

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if torch.cuda.is_available() else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.spg,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        sampler=train_sampler,
        prefetch_factor=4 if torch.cuda.is_available() else 2)

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=torch.cuda.is_available(),
                            drop_last=False)

    model = network(
        backbone=args.backbone,
        num_classes=args.num_classes,
        pretrained=args.pretrained,
        init_momentum=args.momentum,
        aux_layer=args.aux_layer
    )

    model.to(device)

    if torch.cuda.is_available():
        model = DistributedDataParallel(model, device_ids=[int(os.environ['LOCAL_RANK'])], find_unused_parameters=True)

    # (Optimizer setup code...)

    train_loader_iter = iter(train_loader)
    avg_meter = AverageMeter()

    # (Loss layers and training loop...)

        inputs = inputs.to(device, non_blocking=True)
        cls_label = cls_label.to(device, non_blocking=True)

        # (Main training logic...)

        optim.zero_grad()
        loss.backward()
        optim.step()

        # (Logging and checkpoint saving...)

if __name__ == "__main__":
    args = parser.parse_args()
    setup_seed(args.seed)
    train(args)
