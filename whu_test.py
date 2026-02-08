# whu_test.py
import os
import time
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import multiprocessing as mp
import multiprocessing.pool as mpp

from train_supervision import *  # Supervision_Train, Evaluator, py2cfg 等


# =========================
# WHU building: binary label -> rgb
# 0: background -> white
# 1: building   -> red
# =========================
def label2rgb(mask: np.ndarray):
    h, w = mask.shape[:2]
    mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    mask_rgb[mask == 0] = [255, 255, 255]
    mask_rgb[mask == 1] = [255, 0, 0]
    return mask_rgb


def img_writer(inp):
    (mask, mask_path_noext, rgb, binary255) = inp
    if rgb:
        out = label2rgb(mask)
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        cv2.imwrite(mask_path_noext + ".png", out)
    else:
        out = mask.astype(np.uint8)
        if binary255:
            out = (out * 255).astype(np.uint8)
        cv2.imwrite(mask_path_noext + ".png", out)


def parse_scales(s: str):
    s = s.strip()
    if not s:
        return [1.0]
    parts = [p.strip() for p in s.split(",")]
    scales = []
    for p in parts:
        if p:
            scales.append(float(p))
    return scales if scales else [1.0]


def pad_to_multiple(x: torch.Tensor, multiple: int = 256, mode: str = "reflect"):
    """
    Pad BCHW tensor at right/bottom so that H,W are multiples of `multiple`.
    Return padded tensor and (orig_h, orig_w).
    """
    b, c, h, w = x.shape
    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, (h, w)
    # pad format: (left, right, top, bottom)
    x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
    return x, (h, w)


def crop_back(x: torch.Tensor, orig_hw):
    oh, ow = orig_hw
    return x[..., :oh, :ow]


def tta_transforms(mode: str):
    """
    Yield tuples describing transforms on BCHW:
    ('none', None) or ('hflip', None) etc.
    We'll use a simple set:
      lr: hflip + vflip
      d4: (none, hflip, vflip, rot90, rot90+hflip, rot90+vflip)  (够用且和你给的脚本一致风格)
    """
    if mode is None or mode == "none":
        return [("none", None)]

    if mode == "lr":
        return [("none", None), ("hflip", None), ("vflip", None)]

    if mode == "d4":
        return [
            ("none", None),
            ("hflip", None),
            ("vflip", None),
            ("rot90", 1),
            ("rot90_hflip", 1),
            ("rot90_vflip", 1),
        ]

    raise ValueError(f"Unknown TTA mode: {mode}")


def apply_tf(x: torch.Tensor, tf):
    name, k = tf
    if name == "none":
        return x
    if name == "hflip":
        return torch.flip(x, dims=[-1])
    if name == "vflip":
        return torch.flip(x, dims=[-2])
    if name == "rot90":
        return torch.rot90(x, k=k, dims=[-2, -1])
    if name == "rot90_hflip":
        x = torch.rot90(x, k=k, dims=[-2, -1])
        return torch.flip(x, dims=[-1])
    if name == "rot90_vflip":
        x = torch.rot90(x, k=k, dims=[-2, -1])
        return torch.flip(x, dims=[-2])
    raise ValueError(f"Unknown tf name: {name}")


def invert_tf(x: torch.Tensor, tf):
    name, k = tf
    if name == "none":
        return x
    if name == "hflip":
        return torch.flip(x, dims=[-1])
    if name == "vflip":
        return torch.flip(x, dims=[-2])
    if name == "rot90":
        return torch.rot90(x, k=4 - k, dims=[-2, -1])
    if name == "rot90_hflip":
        x = torch.flip(x, dims=[-1])
        return torch.rot90(x, k=4 - k, dims=[-2, -1])
    if name == "rot90_vflip":
        x = torch.flip(x, dims=[-2])
        return torch.rot90(x, k=4 - k, dims=[-2, -1])
    raise ValueError(f"Unknown tf name: {name}")


def _safe_get(batch: dict, keys):
    for k in keys:
        if k in batch:
            return batch[k]
    return None


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, required=True, help="Path to config py file, e.g. config/whu_config.py")
    arg("-o", "--output_path", type=Path, required=True, help="Folder to save predicted masks.")
    arg("-t", "--tta", default="none", choices=["none", "lr", "d4"], help="Test time augmentation.")
    arg("--scales", default="", help="Multi-scale list for TTA d4, e.g. '0.75,1.0,1.25,1.5'. Empty -> 1.0")
    arg("--pad_multiple", type=int, default=256, help="Pad H/W to multiple of this before forward (recommended 256).")
    arg("--rgb", action="store_true", help="Save rgb masks for visualization (white/red).")
    arg("--binary255", action="store_true", help="If not --rgb, save binary mask as {0,255} instead of {0,1}.")
    arg("--eval", action="store_true", help="Compute metrics (requires GT in dataset).")
    return parser.parse_args()


def main():
    args = get_args()
    config = py2cfg(args.config_path)
    args.output_path.mkdir(exist_ok=True, parents=True)

    # -------------------------
    # Load checkpoint
    # -------------------------
    ckpt_path = os.path.join(config.weights_path, config.test_weights_name + ".ckpt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Check config.weights_path/test_weights_name, or rename your ckpt."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Supervision_Train.load_from_checkpoint(ckpt_path, config=config).to(device)
    model.eval()

    # -------------------------
    # Dataset: prefer test_dataset
    # -------------------------
    test_dataset = getattr(config, "test_dataset", None)
    if test_dataset is None:
        # 如果你的 config 里还没写 test_dataset，这里就只能退回 val_dataset
        # 你要真跑 test，请在 config 里补一个 test_dataset（下面我也给你示例）
        test_dataset = config.val_dataset

    evaluator = None
    if args.eval:
        evaluator = Evaluator(num_class=config.num_classes)
        evaluator.reset()

    # -------------------------
    # DataLoader
    # -------------------------
    test_loader = DataLoader(
        test_dataset,
        batch_size=2,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    # -------------------------
    # TTA settings
    # -------------------------
    tfs = tta_transforms(args.tta)
    scales = parse_scales(args.scales) if args.tta == "d4" else [1.0]
    softmax = nn.Softmax(dim=1)

    results = []

    total_processed = 0  # <--- 新增全局计数器

    with torch.no_grad():
        for batch in tqdm(test_loader):
            imgs = _safe_get(batch, ["img", "image", "imgs"])
            if imgs is None:
                raise KeyError("Cannot find image tensor in batch. Expected keys like 'img'.")

            imgs = imgs.to(device, non_blocking=True)
            b, c, h0, w0 = imgs.shape

            # accumulate prob
            prob_sum = None
            count = 0

            for tf in tfs:
                x_tf = apply_tf(imgs, tf)

                for s in scales:
                    if abs(s - 1.0) > 1e-6:
                        hs = max(1, int(round(h0 * s)))
                        ws = max(1, int(round(w0 * s)))
                        x_s = F.interpolate(x_tf, size=(hs, ws), mode="bilinear", align_corners=False)
                    else:
                        x_s = x_tf
                        hs, ws = x_s.shape[-2], x_s.shape[-1]

                    # KEY: pad to multiple (default 256) to avoid UNetFormer internal reflect-pad crash
                    x_pad, orig_hw = pad_to_multiple(x_s, multiple=args.pad_multiple, mode="reflect")

                    logits = model(x_pad)
                    if isinstance(logits, (tuple, list)):
                        logits = logits[0]

                    logits = crop_back(logits, orig_hw)  # crop back to (hs,ws)

                    probs = softmax(logits)

                    # resize back to original (h0,w0)
                    if (hs, ws) != (h0, w0):
                        probs = F.interpolate(probs, size=(h0, w0), mode="bilinear", align_corners=False)

                    # invert transform back
                    probs = invert_tf(probs, tf)

                    if prob_sum is None:
                        prob_sum = probs
                    else:
                        prob_sum = prob_sum + probs
                    count += 1

            prob_avg = prob_sum / max(count, 1)
            preds = prob_avg.argmax(dim=1)  # NxHxW

            image_ids = _safe_get(batch, ["img_id", "image_id", "name", "id"])
            if image_ids is None:
                current_batch_size = preds.shape[0]
                # 生成如: img_0, img_1 ... img_2415
                image_ids = [f"img_{total_processed + i}" for i in range(current_batch_size)]
                total_processed += current_batch_size

            masks_true = None
            if args.eval:
                masks_true = _safe_get(batch, ["gt_semantic_seg", "mask", "masks", "label", "gt"])
                if masks_true is None:
                    raise KeyError("You used --eval but cannot find GT mask in batch (e.g. 'gt_semantic_seg').")

            for i in range(preds.shape[0]):
                mask = preds[i].detach().cpu().numpy().astype(np.uint8)

                mid = str(image_ids[i])
                mid = os.path.splitext(mid)[0]  # avoid xxx.tif.png

                out_path_noext = str(args.output_path / mid)

                if args.eval and evaluator is not None:
                    gt = masks_true[i].detach().cpu().numpy()
                    evaluator.add_batch(pre_image=mask, gt_image=gt)

                results.append((mask, out_path_noext, args.rgb, args.binary255))

    # -------------------------
    # Print metrics
    # -------------------------
    if args.eval and evaluator is not None:
        iou_per_class = evaluator.Intersection_over_Union()
        f1_per_class = evaluator.F1()
        OA = evaluator.OA()
        for class_name, class_iou, class_f1 in zip(config.classes, iou_per_class, f1_per_class):
            print(f"F1_{class_name}: {class_f1}, IOU_{class_name}: {class_iou}")
        print(f"F1(mean): {np.nanmean(f1_per_class)}, mIoU: {np.nanmean(iou_per_class)}, OA: {OA}")

    # -------------------------
    # Save masks (multiprocess)
    # -------------------------
    t0 = time.time()
    num_writers = min(8, mp.cpu_count())
    print(f"Starting image writers with {num_writers} processes...")
    with mpp.Pool(processes=num_writers) as pool:
        pool.map(img_writer, results)
    t1 = time.time()
    print(f"images writing spends: {t1 - t0:.3f} s")


if __name__ == "__main__":
    main()
