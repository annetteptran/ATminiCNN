import pickle
from pathlib import Path
from typing import Tuple, List, Any, Union
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from PIL import Image
import numpy as np

IM_MEAN = (0.4802, 0.4481, 0.3975)
IM_STD  = (0.2302, 0.2265, 0.2262)


def _to_pil(img: Union[np.ndarray, torch.Tensor, Image.Image]) -> Image.Image:
    """Convert a numpy array or tensor (C,H,W or H,W,C) to PIL.Image."""
    if isinstance(img, Image.Image):
        return img

    if isinstance(img, torch.Tensor):
        x = img
        if x.ndim == 3 and x.shape[0] in (1, 3):
            x = x.detach().cpu().float()
            if x.max() <= 1.0:
                x = (x * 255.0).clamp(0, 255)
            x = x.byte().permute(1, 2, 0).numpy()
            return Image.fromarray(x)
        elif x.ndim == 3 and x.shape[-1] in (1, 3):
            x = x.detach().cpu().float()
            if x.max() <= 1.0:
                x = (x * 255.0).clamp(0, 255)
            x = x.byte().numpy()
            return Image.fromarray(x)
        else:
            raise ValueError(f"Unsupported tensor shape {tuple(x.shape)}")

    if isinstance(img, np.ndarray):
        # handle CHW or HWC
        if img.ndim == 3 and img.shape[0] in (1, 3):  # CHW
            x = img
            if x.dtype != np.uint8:
                x = np.clip(x, 0, 1) * 255.0 if x.max() <= 1.0 else np.clip(x, 0, 255)
                x = x.astype(np.uint8)
            x = np.transpose(x, (1, 2, 0))  # HWC
            return Image.fromarray(x)
        elif img.ndim == 3 and img.shape[-1] in (1, 3):  # HWC
            x = img
            if x.dtype != np.uint8:
                x = np.clip(x, 0, 1) * 255.0 if x.max() <= 1.0 else np.clip(x, 0, 255)
                x = x.astype(np.uint8)
            return Image.fromarray(x)
        else:
            raise ValueError(f"Unsupported ndarray shape {img.shape}")

    raise TypeError(f"Unsupported image type: {type(img)}")


class PKLDataset(Dataset):
    """
    Supports two common pickle formats:
      - dict with keys like 'images'/'labels' (or 'x'/'y')
      - list/tuple of (image, label) pairs
    """
    def __init__(self, pkl_path: str, transform=None):
        self.path = Path(pkl_path)
        self.transform = transform

        with open(self.path, "rb") as f:
            obj = pickle.load(f)

        self.samples: List[Tuple[Any, int]] = []

        if isinstance(obj, dict):
            img_key = "images" if "images" in obj else ("x" if "x" in obj else None)
            lab_key = "labels" if "labels" in obj else ("y" if "y" in obj else None)
            if img_key is None or lab_key is None:
                raise KeyError(f"Could not find images/labels keys in {list(obj.keys())}")

            images = obj[img_key]
            labels = obj[lab_key]
            assert len(images) == len(labels), "images and labels must be same length"

            for im, lb in zip(images, labels):
                pil = _to_pil(im)
                self.samples.append((pil, int(lb)))

        elif isinstance(obj, (list, tuple)):
            for item in obj:
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    raise ValueError("List format must be [(image, label), ...]")
                im, lb = item[0], item[1]
                pil = _to_pil(im)
                self.samples.append((pil, int(lb)))
        else:
            raise TypeError(f"Unsupported PKL root type: {type(obj)}")

        self.num_classes = int(max(lb for _, lb in self.samples)) + 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, label = self.samples[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def make_loaders(
    data_dir="data",
    dataset="cifar10",   # "cifar10", "cifar100", or "pkl"
    train_pkl="train-70_.pkl",
    val_pkl="validation-10_.pkl",
    img_size=64,
    batch_size=128,
    num_workers=4,
):
    """
    Build train/val dataloaders.

    - dataset="cifar10"  → CIFAR-10 *tiny subset* (2k train / 1k val)
    - dataset="cifar100" → full CIFAR-100
    - dataset="pkl"      → PKL files in data_dir
    """
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(IM_MEAN, IM_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.12)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IM_MEAN, IM_STD),
    ])

    ds = dataset.lower()

    if ds == "cifar10":
        # Full CIFAR-10 datasets
        full_train = CIFAR10(root=data_dir, download=True, train=True,  transform=train_tf)
        full_val   = CIFAR10(root=data_dir, download=True, train=False, transform=val_tf)

        # Tiny subset for faster training
        train_size = min(2000, len(full_train))   # 2k train samples
        val_size   = min(1000, len(full_val))     # 1k val samples

        train_ds = Subset(full_train, range(train_size))
        val_ds   = Subset(full_val,   range(val_size))

        classes = full_train.classes

    elif ds == "cifar100":
        train_ds = CIFAR100(root=data_dir, download=True, train=True,  transform=train_tf)
        val_ds   = CIFAR100(root=data_dir, download=True, train=False, transform=val_tf)
        classes = train_ds.classes

    else: 
        train_ds = PKLDataset(Path(data_dir) / train_pkl, transform=train_tf)
        val_ds   = PKLDataset(Path(data_dir) / val_pkl,   transform=val_tf)
        classes = [
            str(i)
            for i in range(max(train_ds.num_classes, getattr(val_ds, "num_classes", 0)))
        ]

    train_ld = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_ld = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_ld, val_ld, classes
