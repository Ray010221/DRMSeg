from pathlib import Path
import random
import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


class FullPartitionDataset(Dataset):

    def __init__(self, root, region_dir='felzenszwalb_full_partitions', train=False):
        self.root = Path(root)
        self.region_dir = self.root / region_dir
        self.train = train
        self.ids = sorted((path.stem for path in (self.root / 'sar').glob('*.jpg')))
        if not self.ids:
            raise FileNotFoundError(self.root / 'sar')
        for sample_id in self.ids:
            for path in (self.root / 'opt' / f'{sample_id}.jpg', self.root / 'lbl' / f'{sample_id}.png', self.region_dir / f'{sample_id}.npy'):
                if not path.is_file():
                    raise FileNotFoundError(path)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        sample_id = self.ids[index]
        with Image.open(self.root / 'opt' / f'{sample_id}.jpg') as image:
            opt = image.convert('RGB')
        with Image.open(self.root / 'sar' / f'{sample_id}.jpg') as image:
            sar = image.convert('L')
        with Image.open(self.root / 'lbl' / f'{sample_id}.png') as image:
            label = image.convert('L')
        partition = np.load(self.region_dir / f'{sample_id}.npy')
        if partition.ndim != 2 or partition.shape != (opt.height, opt.width):
            raise ValueError(f'Expected full HxW partition for {sample_id}')
        if sar.size != opt.size or label.size != opt.size:
            raise ValueError(f'Image/label dimensions differ for {sample_id}')
        partition_image = Image.fromarray(partition.astype(np.int32), mode='I')
        if self.train:
            angle = random.uniform(-180, 180)
            (opt, sar, label) = [image.rotate(angle) for image in (opt, sar, label)]
            partition_image = partition_image.rotate(angle, fillcolor=-1)
            for enhancer in (ImageEnhance.Brightness, ImageEnhance.Color, ImageEnhance.Contrast):
                if random.random() > 0.5:
                    opt = enhancer(opt).enhance(1.5)
        partition = np.array(partition_image, dtype=np.int32)
        (labels, remapped) = np.unique(partition, return_inverse=True)
        regions = torch.from_numpy(remapped.reshape(partition.shape).astype(np.int64))
        masks = torch.nn.functional.one_hot(regions, num_classes=len(labels)).permute(2, 0, 1)
        return {'id': sample_id, 'img_opt': TF.normalize(TF.to_tensor(opt), (0.5,) * 3, (0.5,) * 3), 'img_sar': TF.normalize(TF.to_tensor(sar), (0.5,), (0.5,)), 'label': torch.from_numpy(np.array(label, dtype=np.int64)), 'sam_masks': masks.float()}


def collate_partitions(samples):
    max_regions = max((sample['sam_masks'].shape[0] for sample in samples))
    (height, width) = samples[0]['label'].shape
    masks = torch.zeros(len(samples), max_regions, height, width)
    valid = torch.zeros(len(samples), max_regions)
    for (index, sample) in enumerate(samples):
        count = sample['sam_masks'].shape[0]
        masks[index, :count] = sample['sam_masks']
        valid[index, :count] = 1
    if not torch.all(masks.sum(1) == 1):
        raise ValueError('Each pixel must belong to exactly one graph region')
    return {'id': [sample['id'] for sample in samples], **{key: torch.stack([sample[key] for sample in samples]) for key in ('img_opt', 'img_sar', 'label')}, 'sam_masks': masks, 'valid_mask': valid}
