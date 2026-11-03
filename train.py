import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import time
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import numpy as np
import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from dataloader import FullPartitionDataset, collate_partitions
from model import DRMSeg
from engine import evaluate, save_atomic, schedule, seed_everything, run_epoch


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2 ** 32
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default='dataset/xian523')
    parser.add_argument('--region-dir', default='felzenszwalb_full_partitions')
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', '--workers', type=int, default=4)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--threshold', type=float, default=0.6)
    parser.add_argument('--temperature', type=float, default=0.15)
    parser.add_argument('--gamma', type=float, default=0.15)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--calibration-warmup', type=int, default=10)
    parser.add_argument('--utility-every', type=int, default=4)
    parser.add_argument('--utility-temperature', type=float, default=0.01)
    parser.add_argument('--lambda-utility', type=float, default=1.0)
    parser.add_argument('--lambda-prior', type=float, default=1.0)
    parser.set_defaults(fusion_mode='adaptive_consistency', prototype_weight=1.0)
    parser.add_argument('--opt-ce-weight', type=float, default=0.5)
    parser.add_argument('--sar-ce-weight', type=float, default=0.5)
    parser.add_argument('--seg1-ce-weight', type=float, default=0.5)
    args = parser.parse_args()
    if args.num_workers < 0 or args.batch_size < 1 or args.epochs < 1:
        raise ValueError('Invalid loader or epoch configuration.')
    if not 0 <= args.prototype_weight <= 1:
        raise ValueError('Prototype weight must be in [0,1].')
    output = Path(args.run_dir)
    if output.exists():
        raise FileExistsError(f'Existing run is not overwritten: {output}')
    seed_everything(args.seed)
    torch.set_num_threads(8)
    torch.cuda.set_device(args.device)
    train_set = FullPartitionDataset(Path(args.data_root) / 'train', args.region_dir, train=True)
    val_set = FullPartitionDataset(Path(args.data_root) / 'val', args.region_dir)
    generator = torch.Generator().manual_seed(args.seed)
    val_generator = torch.Generator().manual_seed(args.seed + 1)
    common = dict(batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_partitions, pin_memory=True, worker_init_fn=seed_worker)
    if args.num_workers:
        common.update(persistent_workers=True, prefetch_factor=2)
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, generator=generator, **common)
    val_loader = DataLoader(val_set, shuffle=False, generator=val_generator, **common)
    model = DRMSeg(gamma=args.gamma, alpha=args.alpha, threshold=args.threshold, temperature=args.temperature, opt_ce_weight=args.opt_ce_weight, sar_ce_weight=args.sar_ce_weight, seg1_ce_weight=args.seg1_ce_weight, pretrained=True).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = schedule(optimizer, args.epochs * len(train_loader), 3 * len(train_loader))
    scaler = GradScaler(enabled=True, init_scale=1024.0)
    protected = [Path(__file__).resolve().parent / name for name in ('model.py', 'engine.py', 'dataloader.py', 'train.py')]
    digests = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
    config = dict(
        vars(args), model_type='joint_region_single_modal',
        architecture='final_proto100_region_mlp_above_mean',
        train_samples=len(train_set), val_samples=len(val_set),
        initialization='ConvNeXtV2 local cached pretraining; new modules initialized directly',
        optimizer=f'AdamW, shared lr={args.lr:g}, weight_decay=0.05',
        segmentation_loss_weights=dict(opt=args.opt_ce_weight, sar=args.sar_ce_weight,
                                       seg1=args.seg1_ce_weight, seg2=1.),
        mask='Haar cues + bounded 6->16->1 regional calibration; M1 repair / M0 fusion eligibility',
        rfusion='Strictly above eligible-region mean consistency',
        optical_repair='100% class prototype inside mask; no SAR concatenation',
        sar_repair='OPT-gated SAR residual; no concatenation',
        final_fusion='Four-representation concatenation',
        degradation_supervision=False, test_data_used_during_training=False,
        parameters=sum(p.numel() for p in model.parameters()),
        calibrator_parameters=sum(p.numel() for p in model.mask_refiner.parameters()),
        source_sha256=digests)
    output.mkdir(parents=True)
    (output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(json.dumps(config, indent=2), flush=True)
    (best, start) = (-1.0, time.perf_counter())
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        print(f'Starting epoch {epoch:03d}/{args.epochs}', flush=True)
        training = run_epoch(model, train_loader, optimizer, scaler, scheduler, args.device, args, epoch)
        validation = evaluate(model, val_loader, args.device)
        improved = validation['seg2']['miou'] > best
        best = max(best, validation['seg2']['miou'])
        record = dict(epoch=epoch, seconds=time.perf_counter() - epoch_start, train=training, val=validation, best_val_miou=best)
        with (output / 'history.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record) + '\n')
        row = dict(epoch=epoch, seconds=record['seconds'], loss=training['loss'], utility=training['utility'], prior=training['prior'], head_grad=training['head_grad'], utility_targets=training['utility_targets'], skipped=training['skipped'], val_oa=validation['seg2']['oa'], val_miou=validation['seg2']['miou'], **{f'val_{key}': value for (key, value) in validation['masks'].items()})
        path = output / 'history.csv'
        exists = path.exists()
        with path.open('a', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        print(f"Epoch {epoch:03d}/{args.epochs} loss={training['loss']:.4f} val_OA={validation['seg2']['oa']:.5f} val_mIoU={validation['seg2']['miou']:.5f} M0={validation['masks']['initial_area']:.3f} M1={validation['masks']['refined_area']:.3f} head_grad={training['head_grad']:.6f} seconds={record['seconds']:.1f}", flush=True)
        payload = dict(model_type='joint_region_single_modal', model=model.state_dict(), args=vars(args), epoch=epoch, best_miou=best, val=validation, calibration_enabled=model.calibration_enabled)
        if improved:
            save_atomic(payload, output / 'best.pth')
        if epoch <= 3 or epoch % 10 == 0 or epoch == args.epochs:
            save_atomic(dict(payload, optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(), loader_rng=generator.get_state()), output / 'last.pth')
    checks = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() == digests[str(path)] for path in protected}
    (output / 'source_checks.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    if not all(checks.values()):
        raise RuntimeError('A source file changed during training.')
    print(f'Training complete in {time.perf_counter() - start:.1f}s; best val mIoU={best:.6f}', flush=True)

if __name__ == '__main__':
    main()
