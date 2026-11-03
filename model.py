import math
from contextlib import contextmanager
import torch
from torch import nn
from torch.nn import functional as F
import timm
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder


class ConvNeXtV2PyramidEncoder(nn.Module):
    """Expose ConvNeXtV2-Tiny features in the six-level U-Net convention.

    The first two levels preserve the input and add a shallow stem. The four
    remaining levels are the feature maps returned by timm's ConvNeXtV2.
    """

    def __init__(self, in_channels, image_size=128, pretrained=False, model_name='convnextv2_tiny', stem_channels=64):
        super().__init__()
        del image_size
        self.backbone = timm.create_model(model_name, pretrained=pretrained, features_only=True, in_chans=in_channels, out_indices=(0, 1, 2, 3))
        feature_channels = tuple(self.backbone.feature_info.channels())
        self.out_channels = (in_channels, stem_channels, *feature_channels)
        self.stem = nn.Sequential(nn.Conv2d(in_channels, stem_channels, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(stem_channels), nn.GELU())

    @staticmethod
    def _channels_first(feature, channels):
        if feature.ndim == 4 and feature.shape[-1] == channels and (feature.shape[1] != channels):
            return feature.permute(0, 3, 1, 2).contiguous()
        return feature

    def forward(self, x):
        features = self.backbone(x)
        features = [self._channels_first(feature, channels) for (feature, channels) in zip(features, self.out_channels[2:])]
        return [x, self.stem(x), *features]

class ConvNeXtV2UNetBranch(nn.Module):
    """One modality-specific ConvNeXtV2 encoder and U-Net decoder."""

    def __init__(self, in_channels, decoder_channels, num_classes, image_size=128, pretrained=False, encoder=None):
        super().__init__()
        self.encoder = encoder or ConvNeXtV2PyramidEncoder(in_channels=in_channels, image_size=image_size, pretrained=pretrained)
        self.decoder = UnetDecoder(encoder_channels=self.encoder.out_channels, decoder_channels=decoder_channels, n_blocks=5, use_batchnorm=True, center=False, attention_type=None)
        self.segmentation_head = SegmentationHead(in_channels=decoder_channels[-1], out_channels=num_classes, activation=None, kernel_size=3)

def denorm_01(x):
    return x * 0.5 + 0.5

def gaussian_window(window_size, sigma, channel):
    gauss = torch.Tensor([float(torch.exp(torch.tensor(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)))) for x in range(window_size)])
    gauss = gauss / gauss.sum()
    window = gauss.unsqueeze(1) @ gauss.unsqueeze(0)
    window = window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, sigma=1.5, C1=0.01 ** 2, C2=0.03 ** 2):
    """
    img1, img2 shape: [B, 1, H, W], range [0,1]
    """
    with torch.cuda.amp.autocast(enabled=False):
        img1 = img1.float()
        img2 = img2.float()
        (_, C, _, _) = img1.shape
        window = gaussian_window(window_size, sigma, C).to(img1.device)
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=C)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=C)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=C) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=C) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=C) - mu1_mu2
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        ssim_map = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2) / denominator.clamp_min(1e-06)
        return ssim_map

class PrototypeBuffer(nn.Module):

    def __init__(self, num_classes, feat_dims):
        super().__init__()
        self.num_classes = num_classes
        self.momentum = 0.95
        self.num_levels = len(feat_dims)
        self.proto_mean = nn.ParameterList()
        self.proto_std = nn.ParameterList()
        for C in feat_dims:
            self.proto_mean.append(nn.Parameter(torch.zeros(num_classes, C), requires_grad=False))
            self.proto_std.append(nn.Parameter(torch.ones(num_classes, C), requires_grad=False))

    @torch.no_grad()
    def update(self, feat_list, labels, cloud_mask):
        """
        feat_list: list of multi-level features
                   e.g. [[B,C1,H1,W1], [B,C2,H2,W2], ...]
        labels: [B, H0, W0]  (original resolution)
        cloud_mask: [B,1,H0,W0]
        """
        B = labels.size(0)
        for (level, feat) in enumerate(feat_list):
            feat = feat.float()
            (B, C, H, W) = feat.shape
            lbl_lvl = F.interpolate(labels.unsqueeze(1).float(), size=(H, W), mode='nearest').long().squeeze(1)
            cloud_lvl = F.interpolate(cloud_mask.float(), size=(H, W), mode='nearest').squeeze(1)
            valid = 1 - cloud_lvl
            for c in range(self.num_classes):
                mask = (lbl_lvl == c).float() * valid
                count = mask.sum()
                min_count = max(4, min(64, int(0.002 * B * H * W)))
                if count < min_count:
                    continue
                feat_c = (feat * mask.unsqueeze(1)).sum([0, 2, 3]) / count
                feat_var = ((feat - feat_c.view(1, C, 1, 1)) ** 2 * mask.unsqueeze(1)).sum([0, 2, 3]) / count
                feat_std = feat_var.sqrt()
                self.proto_mean[level][c].mul_(self.momentum).add_(feat_c, alpha=1 - self.momentum)
                self.proto_std[level][c].mul_(self.momentum).add_(feat_std, alpha=1 - self.momentum)

def compute_centers_split(feat, seg, cloud_mask, num_cls, eps=1e-06):
    """
    feat:       [B,C,H,W]
    seg:        [B,H,W]
    cloud_mask: [B,1,H,W]
    return:
        mu_cloud: [B,K,C]
        mu_clear: [B,K,C]
        valid_c:  [B,K]
        valid_n:  [B,K]
    """
    (B, C, H, W) = feat.shape
    K = num_cls
    feat = feat.float().view(B, C, -1)
    seg = seg.view(B, -1)
    cloud = cloud_mask.view(B, -1).float()
    clear = 1.0 - cloud
    onehot = F.one_hot(seg, K).float()
    w_c = onehot * cloud.unsqueeze(-1)
    cnt_c = w_c.sum(1)
    mu_c = torch.einsum('bch,bhk->bkc', feat, w_c)
    mu_c = mu_c / (cnt_c.unsqueeze(-1) + eps)
    w_n = onehot * clear.unsqueeze(-1)
    cnt_n = w_n.sum(1)
    mu_n = torch.einsum('bch,bhk->bkc', feat, w_n)
    mu_n = mu_n / (cnt_n.unsqueeze(-1) + eps)
    valid_c = cnt_c > 5
    valid_n = cnt_n > 5
    return (mu_c, mu_n, valid_c, valid_n)

def cross_region_similarity(mu_c, mu_n):
    """
    mu_c: [B,K,C]  cloud
    mu_n: [B,K,C]  non-cloud
    return: [B,K,K]
    """
    mu_c = F.normalize(mu_c, dim=-1, eps=1e-06)
    mu_n = F.normalize(mu_n, dim=-1, eps=1e-06)
    return torch.matmul(mu_c, mu_n.transpose(1, 2))

def cloud_structure_loss(opt_feat, sar_feat, seg1, cloud_mask, num_cls, eps=1e-06):
    with torch.cuda.amp.autocast(enabled=False):
        (opt_c, opt_n, vc1, vn1) = compute_centers_split(opt_feat.float(), seg1, cloud_mask, num_cls)
        (sar_c, sar_n, vc2, vn2) = compute_centers_split(sar_feat.float(), seg1, cloud_mask, num_cls)
        S_opt = cross_region_similarity(opt_c, opt_n)
        S_sar = cross_region_similarity(sar_c, sar_n).detach()
        valid = vc1 & vn1 & vc2 & vn2
        mask = (valid.unsqueeze(2) & valid.unsqueeze(1)).float()
        return ((S_opt - S_sar).abs() * mask).sum() / mask.sum().clamp_min(eps)

def pool_regions(features, masks):
    with torch.autocast(device_type=features.device.type, enabled=False):
        if features.shape[-2:] != masks.shape[-2:]:
            masks = F.interpolate(masks.float(), size=features.shape[-2:], mode='nearest')
        flat = masks.flatten(2).float()
        area = flat.sum(-1).clamp_min(1e-06)
        return torch.bmm(flat, features.float().flatten(2).transpose(1, 2)) / area.unsqueeze(-1)

def regional_frequency_prior(opt, sar, masks, valid, threshold=0.6, temperature=0.15):
    """Preserve the old frequency cues, replacing only their hard projection."""
    with torch.autocast(device_type=opt.device.type, enabled=False):
        opt = opt.float().mean(1, keepdim=True)
        sar = sar.float()
        lp = opt.new_tensor([0.5, 0.5]).view(1, 1, 1, 2)
        hp = opt.new_tensor([0.5, -0.5]).view(1, 1, 1, 2)

        def wavelet(image):
            low = F.conv2d(image, lp, stride=(1, 2))
            high = F.conv2d(image, hp, stride=(1, 2))
            return (F.conv2d(low, lp.transpose(2, 3), stride=(2, 1)), F.conv2d(low, hp.transpose(2, 3), stride=(2, 1)), F.conv2d(high, lp.transpose(2, 3), stride=(2, 1)), F.conv2d(high, hp.transpose(2, 3), stride=(2, 1)))
        (optical, radar) = (wavelet(opt), wavelet(sar))
        cues = torch.cat([(optical[0] - radar[0]).abs(), -sum((band.square() for band in optical[1:])), sum(((a - b).abs() for (a, b) in zip(optical[1:], radar[1:])))], dim=1)
        regional = pool_regions(cues, masks)
        support = valid.bool().unsqueeze(-1)
        low = regional.masked_fill(~support, torch.inf).amin(1, keepdim=True)
        high = regional.masked_fill(~support, -torch.inf).amax(1, keepdim=True)
        has_valid = support.any(1, keepdim=True)
        low = torch.where(has_valid, low, torch.zeros_like(low))
        high = torch.where(has_valid, high, torch.ones_like(high))
        span = high - low
        normalized = torch.where(span > 1e-06, (regional - low) / span.clamp_min(1e-06), torch.full_like(regional, 0.5))
        normalized = normalized.masked_fill(~support, 0)
        score = (normalized * opt.new_tensor([0.4, 0.2, 0.4])).sum(-1)
        flat = masks.float().flatten(2) * valid.unsqueeze(-1)
        coverage = flat.sum(1, keepdim=True)
        projected = torch.bmm(score.unsqueeze(1), flat) / coverage.clamp_min(1)
        projected = torch.where(coverage > 0, projected, torch.full_like(projected, threshold))
        projected = projected.reshape(opt.shape[0], 1, *masks.shape[-2:])
        initial = torch.sigmoid((projected - threshold) / temperature)
        return (initial, projected, cues, score)

def normalized_region_inputs(cues, masks, valid):
    regional = pool_regions(cues, masks)
    support = valid.bool().unsqueeze(-1)
    low = regional.masked_fill(~support, torch.inf).amin(1, keepdim=True)
    high = regional.masked_fill(~support, -torch.inf).amax(1, keepdim=True)
    span = high - low
    normalized = torch.where(span > 1e-06, (regional - low) / span.clamp_min(1e-06), torch.full_like(regional, 0.5)).masked_fill(~support, 0)
    median = normalized.masked_fill(~support, torch.nan).nanmedian(1, keepdim=True).values
    return torch.cat([normalized, normalized - median], dim=-1).masked_fill(~support, 0)

class RegionCalibrator(nn.Module):

    def __init__(self, threshold=0.6, gamma=0.15, alpha=0.1, temperature=0.15):
        super().__init__()
        if gamma < 0 or alpha <= 0 or temperature <= 0:
            raise ValueError('Invalid bounded calibration configuration')
        (self.threshold, self.gamma) = (float(threshold), float(gamma))
        (self.alpha, self.temperature) = (float(alpha), float(temperature))
        self.mlp = nn.Sequential(nn.Linear(6, 16), nn.GELU(), nn.Linear(16, 1))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, inputs, initial, valid):
        uncertain = valid.bool() & ((initial - self.threshold).abs() <= self.gamma + 1e-07)
        delta = self.alpha * self.mlp(inputs).squeeze(-1).tanh() * uncertain
        score = initial + delta
        return {'score': score, 'delta': delta, 'uncertain': uncertain, 'probability': torch.sigmoid((score - self.threshold) / self.temperature), 'hard': (score > self.threshold) & valid.bool()}

def project_regions(states, masks):
    return torch.bmm(states.float().unsqueeze(1), masks.float().flatten(2)).reshape(masks.shape[0], 1, *masks.shape[-2:])

def reliable_region_mean(scores, eligible):
    """Center before reduction so identical scores retain an identical mean."""
    if scores.ndim != 2 or scores.shape != eligible.shape:
        raise ValueError('Expected matching BxN scores and eligibility')
    if scores.shape[1] == 0:
        return scores.new_zeros(scores.shape[0], 1)
    count = eligible.sum(1, keepdim=True)
    index = eligible.long().argmax(1, keepdim=True)
    anchor = scores.gather(1, index).masked_fill(count == 0, 0)
    residual = (scores - anchor).masked_fill(~eligible, 0)
    return anchor + residual.sum(1, keepdim=True) / count.clamp_min(1)

def adaptive_region_selection(scores, eligible):
    """Strictly above the per-image mean over eligible regions; no fallback."""
    return eligible & (scores > reliable_region_mean(scores, eligible))

def original_region_reliability(masks, valid, degradation, features):
    """Retain the original consistency/background/area score and eligibility."""
    area = masks.float().sum((-2, -1)).clamp_min(1e-06)
    means = [pool_regions(feature, masks) for feature in features]
    consistency = sum((F.cosine_similarity(means[a], means[b], dim=-1) for a in range(4) for b in range(a + 1, 4))) / 6
    optical = features[0].float()
    background = optical.sum((-2, -1)).unsqueeze(1) - means[0] * area.unsqueeze(-1)
    background = background / (optical.shape[-2] * optical.shape[-1] - area).clamp_min(1e-06).unsqueeze(-1)
    distinction = 1 - F.cosine_similarity(means[0], background, dim=-1)
    score = 0.4 * consistency + 0.3 * distinction + 0.3 * area / masks.shape[-2] / masks.shape[-1]
    eligible = valid.bool() & (pool_regions(degradation, masks).squeeze(-1) < 0.5)
    return (score, eligible)

class OpticalGatedSARFilter(nn.Module):
    """OPT supplies a gate; only projected SAR values enter the residual stack."""

    def __init__(self, in_channels, guide_channels, bottleneck_channels=64):
        super().__init__()
        hidden = min(bottleneck_channels, max(8, in_channels // 4))
        self.sar_proj = nn.Conv2d(in_channels, hidden, 1)
        self.guide_proj = nn.Conv2d(guide_channels, hidden, 1)
        self.fusion = nn.Sequential(nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden), nn.ReLU(inplace=True), nn.Conv2d(hidden, in_channels, 1))
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, sar_feat, guide_feat, reliable_mask):
        sar_low = self.sar_proj(sar_feat)
        gate = torch.tanh(self.guide_proj(guide_feat * reliable_mask)) * reliable_mask
        residual = self.fusion(sar_low * (1 + gate))
        return sar_feat + self.alpha * residual

class SAR_EdgeEnhancer(nn.Module):

    def __init__(self, sar_channels, guide_channels, num_enhance_levels=3, bottleneck_channels=64):
        super().__init__()
        self.blocks = nn.ModuleList([OpticalGatedSARFilter(sar_channels[i], guide_channels[i], bottleneck_channels) for i in range(num_enhance_levels)])

    def forward(self, sar_feats, guide_feats, cloud_mask):
        out = list(sar_feats)
        for (i, blk) in enumerate(self.blocks):
            reliable_mask = 1.0 - F.interpolate(cloud_mask.float(), size=sar_feats[i].shape[-2:], mode='nearest')
            out[i] = blk(sar_feats[i], guide_feats[i], reliable_mask)
        return out

class DRMSeg(nn.Module):
    """Final network: full prototype replacement and adaptive mean fusion."""

    def __init__(self, num_classes=6, image_size=128, pretrained=False,
                 gamma=0.15, alpha=0.1, threshold=0.6, temperature=0.15,
                 opt_ce_weight=0.5, sar_ce_weight=0.5, seg1_ce_weight=0.5):
        super().__init__()
        weights = (opt_ce_weight, sar_ce_weight, seg1_ce_weight)
        if any((not math.isfinite(value) or value < 0 for value in weights)):
            raise ValueError('CE weights must be finite and nonnegative.')
        self._init_input_normalization((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (0.5,), (0.5,))
        decoder_channels = (256, 128, 64, 32, 16)
        self.Unet_OPT = ConvNeXtV2UNetBranch(3, decoder_channels, num_classes, image_size, pretrained)
        self.Unet_SAR = ConvNeXtV2UNetBranch(1, decoder_channels, num_classes, image_size, pretrained)
        optical_channels = self.Unet_OPT.encoder.out_channels
        sar_channels = self.Unet_SAR.encoder.out_channels
        decoder_kwargs = dict(
            decoder_channels=decoder_channels, n_blocks=5,
            use_batchnorm=True, center=False, attention_type=None)
        self.SAR2OPT_Decoder = UnetDecoder(encoder_channels=optical_channels, **decoder_kwargs)
        self.SAR_Enhance_Decoder = UnetDecoder(encoder_channels=sar_channels, **decoder_kwargs)
        self.SAR2OPT_Decoder_Head = SegmentationHead(16, 3, activation='tanh', kernel_size=3)
        self.SAR_Enhance_Decoder_Head = SegmentationHead(16, 1, activation='tanh', kernel_size=3)
        self.Fusion_Segmentation_Head1 = SegmentationHead(32, num_classes, activation=None, kernel_size=3)
        self.Fusion_Segmentation_Head2 = SegmentationHead(64, num_classes, activation=None, kernel_size=3)
        self.prototype_buffer = PrototypeBuffer(num_classes, optical_channels)
        self.SAR_EdgeEnhancer = SAR_EdgeEnhancer(sar_channels, optical_channels)
        self.mask_refiner = RegionCalibrator(threshold, gamma, alpha, temperature)
        self.num_classes = num_classes
        self.cloud_threshold = threshold
        self.mask_temperature = temperature
        self.calibration_enabled = True
        (self.opt_ce_weight, self.sar_ce_weight, self.seg1_ce_weight) = weights
        self.ce_loss = nn.CrossEntropyLoss()

    def _init_input_normalization(self, opt_mean, opt_std, sar_mean, sar_std):
        self.register_buffer('_opt_mean', torch.as_tensor(opt_mean, dtype=torch.float32).view(1, -1, 1, 1), persistent=False)
        self.register_buffer('_opt_std', torch.as_tensor(opt_std, dtype=torch.float32).view(1, -1, 1, 1), persistent=False)
        self.register_buffer('_sar_mean', torch.as_tensor(sar_mean, dtype=torch.float32).view(1, -1, 1, 1), persistent=False)
        self.register_buffer('_sar_std', torch.as_tensor(sar_std, dtype=torch.float32).view(1, -1, 1, 1), persistent=False)

    @staticmethod
    def _denormalize(x, mean, std):
        return (x * std + mean).clamp_(0.0, 1.0)

    def repair_cached(self, result, degradation):
        restored = self.restore_cloud_region_multi(result['opt_features'], result['sar_features'], degradation, result['pseudo_labels'])
        semantic = self.SAR2OPT_Decoder(*restored)
        structural = self.SAR_Enhance_Decoder(*self.SAR_EdgeEnhancer(result['sar_features'], restored, degradation))
        return (semantic, structural)

    def decode_cached(self, result, representations):
        return self.Fusion_Segmentation_Head2(torch.cat([result['opt_decoder'], result['sar_decoder'], *representations], dim=1) * (1 + result['fusion_mask']))

    def predict(self, sar, opt, masks, valid):
        sar_features = self.Unet_SAR.encoder(sar)
        opt_features = self.Unet_OPT.encoder(opt)
        sar_decoder = self.Unet_SAR.decoder(*sar_features)
        opt_decoder = self.Unet_OPT.decoder(*opt_features)
        sar_logits = self.Unet_SAR.segmentation_head(sar_decoder)
        opt_logits = self.Unet_OPT.segmentation_head(opt_decoder)
        seg1 = self.Fusion_Segmentation_Head1(torch.cat([sar_decoder, opt_decoder], dim=1))
        with torch.autocast(device_type=opt.device.type, enabled=False):
            (_, _, cues, scores) = regional_frequency_prior(self._denormalize(opt, self._opt_mean, self._opt_std), self._denormalize(sar, self._sar_mean, self._sar_std), masks, valid, self.cloud_threshold, self.mask_temperature)
            inputs = normalized_region_inputs(cues, masks, valid)
            calibration = self.mask_refiner(inputs, scores, valid)
            initial_states = (scores > self.cloud_threshold) & valid.bool()
            initial = project_regions(initial_states, masks)
            if self.calibration_enabled:
                # Region decisions are hard forward, with a soft surrogate gradient.
                proxy = calibration['probability'] * calibration['uncertain']
                states = calibration['hard'].float() + (proxy - proxy.detach())
            else:
                states = initial_states.float()
            corrected = project_regions(states, masks)
        result = {
            'seg1': seg1, 'opt_logits': opt_logits, 'sar_logits': sar_logits,
            'opt_features': [value.detach() for value in opt_features],
            'sar_features': [value.detach() for value in sar_features],
            'pseudo_labels': seg1.detach().argmax(1),
            'sar_decoder': sar_decoder, 'opt_decoder': opt_decoder,
            'mask_initial': initial, 'mask_refined': corrected,
            'mask_residual': project_regions(calibration['delta'], masks),
            'calibration': calibration, 'region_scores': scores}
        (semantic, structural) = self.repair_cached(result, corrected)
        # M1 routes restoration; M0 retains the original fusion eligibility rule.
        fusion = self.reliable_fusion_mask(masks, valid, initial, [opt_decoder, sar_decoder, semantic, structural])
        result.update({'semantic': semantic, 'structural': structural, 'fusion_mask': fusion})
        result['seg2'] = self.decode_cached(result, (semantic, structural))
        return result

    def training_step(self, sar, opt, label, masks, valid):
        result = self.predict(sar, opt, masks, valid)
        opt_recon = self.SAR2OPT_Decoder_Head(result['semantic'])
        sar_recon = self.SAR_Enhance_Decoder_Head(result['structural'])
        mask_weight = result['mask_refined'].detach()
        loss_rec = F.l1_loss(opt_recon.float() * (1 - mask_weight), opt.float() * (1 - mask_weight))
        gray = 0.299 * opt[:, :1] + 0.587 * opt[:, 1:2] + 0.114 * opt[:, 2:3]
        reliable_ssim = (ssim(denorm_01(sar_recon), denorm_01(gray)) * (1 - mask_weight)).mean()
        loss_sar = 0.7 * F.l1_loss(sar_recon.float(), sar.float()) + 0.3 * (1 - reliable_ssim)
        loss_struct = cloud_structure_loss(result['semantic'], result['sar_decoder'].detach(), result['pseudo_labels'], mask_weight, self.num_classes)
        loss_seg = (
            self.sar_ce_weight * self.ce_loss(result['sar_logits'], label)
            + self.opt_ce_weight * self.ce_loss(result['opt_logits'], label)
            + self.seg1_ce_weight * self.ce_loss(result['seg1'], label)
            + self.ce_loss(result['seg2'], label))
        loss = loss_seg + 0.4 * loss_rec + 0.4 * loss_sar + 0.05 * loss_struct
        return (result, loss, {'seg': loss_seg.detach(), 'opt_rec': loss_rec.detach(), 'sar_rec': loss_sar.detach(), 'structure': loss_struct.detach()})

    @torch.no_grad()
    def update_memory(self, result):
        self.prototype_buffer.update(result['opt_features'], result['pseudo_labels'], result['mask_refined'].detach())

    def test(self, sar, opt, masks, valid):
        result = self.predict(sar, opt, masks, valid)
        return (result['seg2'], result['sar_logits'], result['opt_logits'])

    def forward(self, sar, opt, masks, valid):
        return self.predict(sar, opt, masks, valid)

    def restore_cloud_region_multi(self, feat_list, sar_feats, cloud_mask, labels):
        """Replace selected optical values with Seg1-indexed class prototypes."""
        restored = []
        for (feature, bank) in zip(feat_list, self.prototype_buffer.proto_mean):
            size = feature.shape[-2:]
            indices = F.interpolate(labels[:, None].float(), size=size, mode='nearest')[:, 0].long()
            prototype = bank[indices].permute(0, 3, 1, 2)
            mask = F.interpolate(cloud_mask.float(), size=size, mode='nearest')
            restored.append(feature * (1 - mask) + prototype * mask)
        return restored

    @torch.no_grad()
    def reliable_fusion_mask(self, masks, valid, degradation, features):
        """Enhance only eligible regions with consistency above their image mean."""
        (scores, eligible) = original_region_reliability(masks, valid, degradation, features)
        selected = adaptive_region_selection(scores, eligible)
        return ((masks * selected[:, :, None, None]).sum(1, keepdim=True) > 0).float()

@contextmanager
def paired_eval(model):
    modes = {module: module.training for module in model.modules()}
    model.eval()
    try:
        yield
    finally:
        for (module, training) in modes.items():
            module.training = training

@torch.no_grad()
def online_utility(model, result, masks, valid, labels, temperature=0.01, min_pixels=64, min_utility=0.0001):
    scores = result['region_scores']
    distance = (scores - model.cloud_threshold).abs()
    eligible = valid.bool() & result['calibration']['uncertain'] & (distance < model.mask_refiner.alpha - 1e-07) & (masks.sum((-2, -1)) >= min_pixels)
    indices = distance.masked_fill(~eligible, torch.inf).argmin(1)
    rows = torch.arange(masks.shape[0], device=masks.device)
    active = eligible[rows, indices]
    target = masks[rows, indices].unsqueeze(1) * active[:, None, None, None]
    initial = result['mask_refined'].detach()
    (on_mask, off_mask) = (torch.maximum(initial, target), initial * (1 - target))
    cached = {**result, 'opt_features': [value.float() for value in result['opt_features']], 'sar_features': [value.float() for value in result['sar_features']], 'opt_decoder': result['opt_decoder'].float(), 'sar_decoder': result['sar_decoder'].float(), 'fusion_mask': result['fusion_mask'].float()}
    with paired_eval(model), torch.autocast(device_type=masks.device.type, enabled=False):
        on = model.decode_cached(cached, model.repair_cached(cached, on_mask))
        off = model.decode_cached(cached, model.repair_cached(cached, off_mask))
    support = target[:, 0]
    values = ((F.cross_entropy(off.float(), labels, reduction='none') - F.cross_entropy(on.float(), labels, reduction='none')) * support).sum((-2, -1))
    values = values / support.sum((-2, -1)).clamp_min(1)
    changed = ((on - off).abs() * target).flatten(1).amax(1) > 1e-07
    accepted = active & changed & (values.abs() >= min_utility)
    if not torch.isfinite(values).all():
        raise FloatingPointError('Non-finite online utility target')
    return (indices, torch.sigmoid(values / temperature), accepted, values)
