"""Evaluate a standalone DRM-Seg checkpoint."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from .dataloader import OpticalSARDataset
    from .model import DualConvNeXtV2Segmentation
except ImportError:
    from dataloader import OpticalSARDataset
    from model import DualConvNeXtV2Segmentation


def perturb_regions(regions, valid, mode, seed):
    if mode == "full":
        return regions, valid
    if mode == "no-regions":
        return torch.zeros_like(regions), torch.zeros_like(valid)
    generator = torch.Generator(device=regions.device).manual_seed(seed)
    if mode == "random-shift":
        return torch.roll(regions, shifts=(regions.shape[-2] // 8, regions.shape[-1] // 8), dims=(-2, -1)), valid
    if mode == "random-area":
        output = torch.zeros_like(regions)
        h, w = regions.shape[-2:]
        for b in range(regions.shape[0]):
            for n in range(regions.shape[1]):
                if valid[b, n] > 0:
                    area = regions[b, n].sum().sqrt().clamp(2, min(h, w) / 2).long()
                    y = torch.randint(0, max(1, h - area.item() + 1), (1,), generator=generator, device=regions.device).item()
                    x = torch.randint(0, max(1, w - area.item() + 1), (1,), generator=generator, device=regions.device).item()
                    output[b, n, y:y + area, x:x + area] = 1
        return output, valid
    raise ValueError(f"Unknown region mode: {mode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--region-dir", default="sam_all_masks")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--region-mode", choices=("full", "no-regions", "random-shift", "random-area"), default="full")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    device = torch.device(args.device)
    dataset = OpticalSARDataset(args.data_root, args.region_dir, False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = DualConvNeXtV2Segmentation(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
    model.cloud_threshold = args.threshold if args.threshold is not None else checkpoint.get("cloud_threshold", 0.6)
    model.top_k = args.top_k
    model.eval()
    confusion = torch.zeros(6, 6, dtype=torch.long)
    with torch.no_grad():
        for batch_index, batch in enumerate(tqdm(loader, desc="test")):
            regions, valid = perturb_regions(batch["sam_masks"].to(device), batch["valid_mask"].to(device), args.region_mode, args.seed + batch_index)
            logits, _, _ = model.test(batch["img_sar"].to(device), batch["img_opt"].to(device), regions, valid)
            target = batch["label"].to(device)
            indices = target.flatten() * 6 + logits.argmax(1).flatten()
            confusion += torch.bincount(indices, minlength=36).reshape(6, 6).cpu()
    intersection = confusion.diag().double()
    union = confusion.sum(0).double() + confusion.sum(1).double() - intersection
    iou = (intersection / union.clamp_min(1)).numpy()
    f1 = (2 * intersection / (confusion.sum(0).double() + confusion.sum(1).double()).clamp_min(1)).numpy()
    print(f"Region mode: {args.region_mode}")
    print(f"OA: {intersection.sum().item() / confusion.sum().item():.5f}")
    print(f"IoU: {iou.tolist()}")
    print(f"mIoU: {iou.mean():.5f}")
    print(f"F1: {f1.tolist()}")
    print(f"mF1: {f1.mean():.5f}")


if __name__ == "__main__":
    main()
