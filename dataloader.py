from pathlib import Path
import random

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

try:
    from .mask_generation import load_candidate_masks
except ImportError:
    from mask_generation import load_candidate_masks


class OpticalSARDataset(Dataset):
    """Load ``opt/``, ``sar/``, ``lbl/`` and offline region ``.npy`` files.

    Each split is expected to contain matching basenames. Images are resized
    only when ``image_size`` is explicitly different from their stored size.
    """

    def __init__(self, root, region_dir="sam_all_masks", train=False,
                 image_size=128, max_instances=10,
                 opt_mean=(0.5, 0.5, 0.5), opt_std=(0.5, 0.5, 0.5),
                 sar_mean=(0.5,), sar_std=(0.5,)):
        self.root = Path(root)
        self.train = train
        self.image_size = image_size
        self.max_instances = max_instances
        self.opt_mean, self.opt_std = opt_mean, opt_std
        self.sar_mean, self.sar_std = sar_mean, sar_std
        self.opt_dir = self.root / "opt"
        self.sar_dir = self.root / "sar"
        self.label_dir = self.root / "lbl"
        self.region_dir = self.root / region_dir
        self.ids = sorted(path.stem for path in self.sar_dir.glob("*.jpg"))
        if not self.ids:
            raise FileNotFoundError(f"No SAR .jpg files found in {self.sar_dir}")

    def __len__(self):
        return len(self.ids)

    def _image(self, path, mode):
        image = Image.open(path).convert(mode)
        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        tensor = TF.to_tensor(image)
        return TF.normalize(tensor,
                            self.opt_mean if mode == "RGB" else self.sar_mean,
                            self.opt_std if mode == "RGB" else self.sar_std)

    def __getitem__(self, index):
        sample_id = self.ids[index]
        opt = Image.open(self.opt_dir / f"{sample_id}.jpg").convert("RGB")
        sar = Image.open(self.sar_dir / f"{sample_id}.jpg").convert("L")
        label = Image.open(self.label_dir / f"{sample_id}.png").convert("L")
        if opt.size != (self.image_size, self.image_size):
            opt = opt.resize((self.image_size, self.image_size), Image.BILINEAR)
            sar = sar.resize((self.image_size, self.image_size), Image.BILINEAR)
            label = label.resize((self.image_size, self.image_size), Image.NEAREST)

        masks, valid = load_candidate_masks(
            self.region_dir / f"{sample_id}.npy",
            max_instances=self.max_instances,
            image_size=self.image_size,
        )

        if self.train and random.random() < 0.5:
            opt, sar, label = opt.transpose(Image.FLIP_LEFT_RIGHT), sar.transpose(Image.FLIP_LEFT_RIGHT), label.transpose(Image.FLIP_LEFT_RIGHT)
            masks = torch.flip(masks, dims=(-1,))
        if self.train and random.random() < 0.5:
            opt = ImageEnhance.Brightness(opt).enhance(1.2)

        opt_tensor = TF.normalize(TF.to_tensor(opt), self.opt_mean, self.opt_std)
        sar_tensor = TF.normalize(TF.to_tensor(sar), self.sar_mean, self.sar_std)
        label_tensor = torch.from_numpy(np.asarray(label, dtype=np.int64))
        return {
            "id": sample_id,
            "img_opt": opt_tensor,
            "img_sar": sar_tensor,
            "label": label_tensor,
            "sam_masks": masks.float(),
            "valid_mask": valid.float(),
        }


def build_dataloaders(data_root, region_dir="sam_all_masks", batch_size=8,
                      workers=0, image_size=128, max_instances=10):
    from torch.utils.data import DataLoader

    train = OpticalSARDataset(Path(data_root) / "train", region_dir, True, image_size, max_instances)
    val = OpticalSARDataset(Path(data_root) / "val", region_dir, False, image_size, max_instances)
    test = OpticalSARDataset(Path(data_root) / "test", region_dir, False, image_size, max_instances)
    kwargs = dict(batch_size=batch_size, num_workers=workers,
                  pin_memory=torch.cuda.is_available())
    return (DataLoader(train, shuffle=True, drop_last=True, **kwargs),
            DataLoader(val, shuffle=False, **kwargs),
            DataLoader(test, shuffle=False, **kwargs))
