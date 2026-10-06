"""Train the standalone optical--SAR DRM-Seg model."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from .dataloader import OpticalSARDataset
    from .model import DualConvNeXtV2Segmentation
except ImportError:
    from dataloader import OpticalSARDataset
    from model import DualConvNeXtV2Segmentation


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def update_confusion(confusion, prediction, target, classes=6):
    valid = (target >= 0) & (target < classes)
    indices = target[valid] * classes + prediction[valid]
    confusion += torch.bincount(indices, minlength=classes * classes).reshape(classes, classes).cpu()


def metrics(confusion):
    matrix = confusion.double()
    intersection = matrix.diag()
    union = matrix.sum(0) + matrix.sum(1) - intersection
    iou = intersection / union.clamp_min(1)
    return (intersection.sum() / matrix.sum().clamp_min(1)).item(), iou.mean().item(), iou.tolist()


def run_epoch(model, loader, optimizer, scaler, device, amp):
    model.train()
    confusion = torch.zeros(6, 6, dtype=torch.long)
    total_loss, steps = 0.0, 0
    for batch in tqdm(loader, desc="train", leave=False):
        sar = batch["img_sar"].to(device)
        opt = batch["img_opt"].to(device)
        label = batch["label"].to(device)
        regions = batch["sam_masks"].to(device)
        valid = batch["valid_mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp):
            logits, loss = model.trainSeg2(sar, opt, label, regions, valid)
        if not torch.isfinite(loss):
            continue
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        update_confusion(confusion, logits.detach().argmax(1), label)
        total_loss += loss.item()
        steps += 1
    oa, miou, _ = metrics(confusion)
    return total_loss / max(steps, 1), oa, miou


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    confusion = torch.zeros(6, 6, dtype=torch.long)
    for batch in tqdm(loader, desc="val", leave=False):
        logits, _, _ = model.test(
            batch["img_sar"].to(device), batch["img_opt"].to(device),
            batch["sam_masks"].to(device), batch["valid_mask"].to(device)
        )
        update_confusion(confusion, logits.argmax(1), batch["label"])
    return metrics(confusion)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--region-dir", default="sam_all_masks")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    train = OpticalSARDataset(Path(args.data_root) / "train", args.region_dir, True)
    val = OpticalSARDataset(Path(args.data_root) / "val", args.region_dir, False)
    common = dict(batch_size=args.batch_size, num_workers=args.workers,
                  pin_memory=device.type == "cuda")
    train_loader = DataLoader(train, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val, shuffle=False, **common)
    model = DualConvNeXtV2Segmentation(pretrained=args.pretrained).to(device)
    model.cloud_threshold = args.threshold
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    amp = device.type == "cuda" and not args.no_amp
    scaler = GradScaler(enabled=amp)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        loss, train_oa, train_miou = run_epoch(model, train_loader, optimizer, scaler, device, amp)
        val_oa, val_miou, val_iou = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:03d}: loss={loss:.4f} train_OA={train_oa:.4f} "
              f"train_mIoU={train_miou:.4f} val_OA={val_oa:.4f} val_mIoU={val_miou:.4f}")
        if val_miou >= best:
            best = val_miou
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_oa": val_oa, "val_miou": val_miou,
                        "val_iou": val_iou, "cloud_threshold": args.threshold,
                        "backbone": "convnextv2_tiny"}, output)
            print(f"Saved {output} (mIoU={best:.4f})")


if __name__ == "__main__":
    main()
