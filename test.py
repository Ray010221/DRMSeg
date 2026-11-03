import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataloader import FullPartitionDataset, collate_partitions
from engine import evaluate
from model import DRMSeg


def load_model(path, device):
    # Only load trusted checkpoints: older PyTorch checkpoint files use pickle.
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint.get('model_type') != 'joint_region_single_modal':
        raise ValueError('Expected a no-concatenation repair checkpoint.')
    args = checkpoint['args']
    if args.get('prototype_weight', .2) != 1.0:
        raise ValueError('This release expects a natively trained 100% replacement checkpoint.')
    if args.get('fusion_mode') != 'adaptive_consistency':
        raise ValueError('This release only implements above-mean consistency fusion.')
    model = DRMSeg(
        gamma=args['gamma'], alpha=args['alpha'], threshold=args['threshold'],
        temperature=args['temperature'], opt_ce_weight=args.get('opt_ce_weight', .5),
        sar_ce_weight=args.get('sar_ce_weight', .5),
        seg1_ce_weight=args.get('seg1_ce_weight', .5), pretrained=False)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.calibration_enabled = checkpoint.get('calibration_enabled', True)
    return model.to(device).eval(), checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', required=True, help='A split directory containing opt/, sar/, lbl/.')
    parser.add_argument('--region-dir', default='felzenszwalb_full_partitions')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output', default='results/test.json')
    args = parser.parse_args()
    torch.set_num_threads(8)
    model, checkpoint = load_model(args.checkpoint, args.device)
    dataset = FullPartitionDataset(args.data_root, args.region_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                        collate_fn=collate_partitions, shuffle=False)
    memory = {key: value.detach().clone() for key, value in model.prototype_buffer.state_dict().items()}
    result = evaluate(model, loader, args.device)
    if any(not torch.equal(value, model.prototype_buffer.state_dict()[key]) for key, value in memory.items()):
        raise RuntimeError('Prototype memory changed during evaluation.')
    result.update(checkpoint_epoch=checkpoint.get('epoch'), configuration=checkpoint['args'],
                  evaluated_samples=len(dataset), prototype_memory_frozen=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    for stage in ('seg1', 'seg2'):
        values = result[stage]
        print(f"{stage}: OA={100*values['oa']:.4f}% mIoU={100*values['miou']:.4f}% "
              f"mF1={100*values['mf1']:.4f}%")
        print('Per-class IoU (%):', [round(100*x, 4) for x in values['iou']])
    print(f'Saved: {output}')


if __name__ == '__main__':
    main()
