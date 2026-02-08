# geoseg/datasets/whu_building_dataset.py
from pathlib import Path
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

def _list_images(img_dir: Path):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp")
    files = []
    for e in exts:
        files += list(img_dir.glob(e))
    return sorted(files)

class WHUBuildingDataset(Dataset):
    """
    Binary building segmentation.
    Directory structure:
      root/train/image, root/train/label
      root/val/image,   root/val/label
      root/test/image,  root/test/label
    """
    def __init__(self, root, split="train", transform=None):
        self.root = Path(root)
        self.split = split
        self.transform = transform  # 关键：必须定义

        self.img_dir = self.root / split / "image"
        self.msk_dir = self.root / split / "label"

        assert self.img_dir.exists(), f"Missing: {self.img_dir}"
        assert self.msk_dir.exists(), f"Missing: {self.msk_dir}"

        self.images = _list_images(self.img_dir)
        assert len(self.images) > 0, f"No images found in {self.img_dir}"

        # build mask paths by same stem
        self.masks = []
        for p in self.images:
            mp = (self.msk_dir / (p.stem + ".png"))
            if not mp.exists():
                mp2 = self.msk_dir / p.name
                if mp2.exists():
                    mp = mp2
                else:
                    raise FileNotFoundError(f"Mask not found for {p.name}: tried {mp} and {mp2}")
            self.masks.append(mp)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 关键：这里要用 self.images / self.masks
        img_path = self.images[idx]
        msk_path = self.masks[idx]

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(msk_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Cannot read mask: {msk_path}")

        # WHU 常见 mask 是 0/255，这里统一成 0/1
        mask = (mask > 0).astype(np.uint8)

        # Albumentations：输入/输出 dict
        if self.transform is not None:
            out = self.transform(image=img, mask=mask)
            img, mask = out["image"], out["mask"]

        # 兜底：转 Tensor
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img).permute(2, 0, 1).contiguous().float() / 255.0
        else:
            # 有些 ToTensorV2 已经是 float tensor，这里不强制 /255
            img = img.contiguous()

        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).long()
        else:
            mask = mask.long()

        return {
            "img": img,
            "gt_semantic_seg": mask
        }
