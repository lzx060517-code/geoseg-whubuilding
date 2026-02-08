import torch
from torch.utils.data import DataLoader

from geoseg.losses import *
from geoseg.datasets.whu_building_dataset import WHUBuildingDataset
from geoseg.models.UNetFormer import UNetFormer
from tools.utils import Lookahead
from tools.utils import process_model_params

# =========================
# Training hparam
# =========================
max_epoch = 40
ignore_index = 255

train_batch_size = 4
val_batch_size = 4

# ---- optimizer hparam (按你要求) ----
lr = 6e-4
weight_decay = 0.01

backbone_lr = 6e-5
backbone_weight_decay = 0.01

# =========================
# Classes
# =========================
# 二分类：背景/建筑
classes = ["background", "building"]
num_classes = 2

# =========================
# Logging / ckpt
# =========================
weights_name = "unetformer-r18-whu-1024-e40"
weights_path = "model_weights/whu_building/{}".format(weights_name)
test_weights_name = "last"

log_name = "whu_building/{}".format(weights_name)
monitor = "val_mIoU"
monitor_mode = "max"
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1

pretrained_ckpt_path = None  # 如需预训练权重再填
resume_ckpt_path = None

gpus = "auto"

# =========================
# Network / Loss
# =========================
net = UNetFormer(num_classes=num_classes)
loss = UnetFormerLoss(ignore_index=ignore_index)

use_aux_loss = True

# =========================
# Dataset / Dataloader
# =========================
# 你的 WHU 目录应为：data/whu_building/train/images, train/masks, val/images, val/masks
data_root = r"D:\airs\data\whubuilding\1"

train_dataset = WHUBuildingDataset(root=data_root, split="train")
val_dataset   = WHUBuildingDataset(root=data_root, split="val")

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=train_batch_size,
    num_workers=4,
    pin_memory=True,
    shuffle=True,
    drop_last=True
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=val_batch_size,
    num_workers=4,
    shuffle=False,
    pin_memory=True,
    drop_last=False
)
# =========================
# Test Dataset / Dataloader
# =========================
test_dataset = WHUBuildingDataset(root=data_root, split="test")

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=val_batch_size,   # 测试用 val_batch_size 就行
    num_workers=4,
    shuffle=False,
    pin_memory=True,
    drop_last=False
)

# =========================
# Optimizer / Scheduler
# =========================
# 如果你的 UNetFormer(backbone) 参数名里包含 "backbone"
# 这行会对 backbone 单独设置 lr 与 weight_decay
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = process_model_params(net, layerwise_params=layerwise_params)

base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch)
