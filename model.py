import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.decoders.unet import Unet
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.base import SegmentationHead
try:
    from .backbone import ConvNeXtV2PyramidEncoder, ConvNeXtV2UNetBranch
except ImportError:
    from backbone import ConvNeXtV2PyramidEncoder, ConvNeXtV2UNetBranch


def denorm_01(x):
    return x * 0.5 + 0.5

def gaussian_window(window_size, sigma, channel):
    gauss = torch.Tensor([
        float(torch.exp(torch.tensor(-(x - window_size//2)**2 / float(2*sigma**2))))
        for x in range(window_size)
    ])
    gauss = gauss/gauss.sum()
    window = gauss.unsqueeze(1) @ gauss.unsqueeze(0)
    window = window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, sigma=1.5, C1=0.01**2, C2=0.03**2):
    """
    img1, img2 shape: [B, 1, H, W], range [0,1]
    """
    # The SSIM denominator can underflow in AMP, especially for locally flat SAR.
    # Keep this auxiliary loss in FP32 while the network itself can still use AMP.
    with torch.cuda.amp.autocast(enabled=False):
        img1 = img1.float()
        img2 = img2.float()
        _, C, _, _ = img1.shape
        window = gaussian_window(window_size, sigma, C).to(img1.device)

        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=C)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=C)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=C) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=C) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=C) - mu1_mu2

        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / denominator.clamp_min(1e-6)
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
            # Prototypes are updated by momentum, not by the optimizer.
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

        for level, feat in enumerate(feat_list):
            # Summing a large feature map in FP16 can overflow after long training.
            feat = feat.float()
            B, C, H, W = feat.shape

            # Internal computation.
            lbl_lvl = F.interpolate(labels.unsqueeze(1).float(),
                                    size=(H, W),
                                    mode="nearest").long().squeeze(1)

            # Internal computation.
            cloud_lvl = F.interpolate(cloud_mask.float(),
                                      size=(H, W),
                                      mode="nearest").squeeze(1)

            valid = 1 - cloud_lvl  # Internal computation.

            for c in range(self.num_classes):
                # Internal computation.
                mask = (lbl_lvl == c).float() * valid

                count = mask.sum()
                # Keep high-resolution estimates stable while still allowing
                # rare classes to initialize at the coarsest feature levels.
                min_count = max(4, min(64, int(0.002 * B * H * W)))
                if count < min_count:
                    continue

                # Internal computation.
                feat_c = (feat * mask.unsqueeze(1)).sum([0, 2, 3]) / count

                # Internal computation.
                feat_var = (((feat - feat_c.view(1, C, 1, 1)) ** 2)
                            * mask.unsqueeze(1)).sum([0, 2, 3]) / count
                feat_std = feat_var.sqrt()

                self.proto_mean[level][c].mul_(self.momentum).add_(
                    feat_c, alpha=1 - self.momentum
                )
                self.proto_std[level][c].mul_(self.momentum).add_(
                    feat_std, alpha=1 - self.momentum
                )



class LearnableGuidedFilter(nn.Module):
    def __init__(self, in_channels, guide_channels, bottleneck_channels=64):
        super().__init__()

        hidden_channels = min(bottleneck_channels, max(8, in_channels // 4))

        self.sar_proj = nn.Conv2d(in_channels, hidden_channels, 1)
        self.guide_proj = nn.Conv2d(guide_channels, hidden_channels, 1)

        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, groups=hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, 1)
        )

        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, sar_feat, guide_feat, reliable_mask):
        sar_low = self.sar_proj(sar_feat)
        guide_feat = guide_feat * reliable_mask
        guide_feat = self.guide_proj(guide_feat)
        x = torch.cat([sar_low, guide_feat], dim=1)
        edge = self.fusion(x)
        return sar_feat + self.alpha * edge


class SAR_EdgeEnhancer(nn.Module):
    def __init__(self, sar_channels, guide_channels, num_enhance_levels=3, bottleneck_channels=64):
        super().__init__()

        self.blocks = nn.ModuleList([
            LearnableGuidedFilter(sar_channels[i], guide_channels[i], bottleneck_channels)
            for i in range(num_enhance_levels)
        ])

    def forward(self, sar_feats, guide_feats, cloud_mask):
        out = list(sar_feats)
        for i, blk in enumerate(self.blocks):
            reliable_mask = 1.0 - F.interpolate(
                cloud_mask.float(), size=sar_feats[i].shape[-2:], mode="nearest"
            )
            out[i] = blk(sar_feats[i], guide_feats[i], reliable_mask)
        return out

def sar_structure_field(sar_feat, k=5):
    """
    sar_feat: [B, C, H, W]
    return: [B, 2, H, W] displacement field
    """
    B, C, H, W = sar_feat.shape

    # Internal computation.
    gx = sar_feat[:, :, :, 1:] - sar_feat[:, :, :, :-1]
    gy = sar_feat[:, :, 1:, :] - sar_feat[:, :, :-1, :]

    gx = F.pad(gx, (0, 1, 0, 0))
    gy = F.pad(gy, (0, 0, 0, 1))

    # Internal computation.
    gx = gx.mean(dim=1, keepdim=True)
    gy = gy.mean(dim=1, keepdim=True)

    # Internal computation.
    gx = F.avg_pool2d(gx, k, stride=1, padding=k//2)
    gy = F.avg_pool2d(gy, k, stride=1, padding=k//2)

    # Internal computation.
    mag = torch.sqrt(gx**2 + gy**2 + 1e-6)
    gx = gx / (mag + 1e-6)
    gy = gy / (mag + 1e-6)

    # Internal computation.
    flow = torch.cat([gx, gy], dim=1) * 0.05
    return flow


def warp_prototype(proto_feat, flow):
    """
    proto_feat: [B, C, H, W]
    flow:       [B, 2, H, W]  (dx, dy)
    """
    B, C, H, W = proto_feat.shape

    # base grid
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=proto_feat.device),
        torch.linspace(-1, 1, W, device=proto_feat.device),
        indexing='ij'
    )
    base_grid = torch.stack([xx, yy], dim=-1)  # [H,W,2]
    base_grid = base_grid.unsqueeze(0).repeat(B, 1, 1, 1)

    # SAR structure warp
    flow = flow.permute(0, 2, 3, 1)
    grid = base_grid + flow

    warped = F.grid_sample(
        proto_feat,
        grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )
    return warped

def compute_centers_split(feat, seg, cloud_mask, num_cls, eps=1e-6):
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
    B, C, H, W = feat.shape
    K = num_cls

    feat = feat.float().view(B, C, -1)  # [B,C,HW]
    seg  = seg.view(B, -1)              # [B,HW]
    cloud = cloud_mask.view(B, -1).float()
    clear = 1.0 - cloud

    onehot = F.one_hot(seg, K).float()  # [B,HW,K]

    # -------- cloud --------
    w_c = onehot * cloud.unsqueeze(-1)
    cnt_c = w_c.sum(1)                  # [B,K]
    mu_c = torch.einsum('bch,bhk->bkc', feat, w_c)
    mu_c = mu_c / (cnt_c.unsqueeze(-1) + eps)

    # -------- clear --------
    w_n = onehot * clear.unsqueeze(-1)
    cnt_n = w_n.sum(1)
    mu_n = torch.einsum('bch,bhk->bkc', feat, w_n)
    mu_n = mu_n / (cnt_n.unsqueeze(-1) + eps)

    valid_c = cnt_c > 5
    valid_n = cnt_n > 5

    return mu_c, mu_n, valid_c, valid_n


def cross_region_similarity(mu_c, mu_n):
    """
    mu_c: [B,K,C]  cloud
    mu_n: [B,K,C]  non-cloud
    return: [B,K,K]
    """
    mu_c = F.normalize(mu_c, dim=-1, eps=1e-6)
    mu_n = F.normalize(mu_n, dim=-1, eps=1e-6)
    return torch.matmul(mu_c, mu_n.transpose(1, 2))


def cloud_structure_loss(
    opt_feat,
    sar_feat,
    seg1,
    cloud_mask,
    num_cls,
    eps=1e-6
):
    # Empty cloud/clear class centers are common. Evaluate this loss in FP32 so
    # normalization remains well defined under AMP.
    with torch.cuda.amp.autocast(enabled=False):
        opt_c, opt_n, vc1, vn1 = compute_centers_split(
            opt_feat.float(), seg1, cloud_mask, num_cls
        )
        sar_c, sar_n, vc2, vn2 = compute_centers_split(
            sar_feat.float(), seg1, cloud_mask, num_cls
        )

        S_opt = cross_region_similarity(opt_c, opt_n)
        S_sar = cross_region_similarity(sar_c, sar_n).detach()
        valid = vc1 & vn1 & vc2 & vn2
        mask = (valid.unsqueeze(2) & valid.unsqueeze(1)).float()
        return ((S_opt - S_sar).abs() * mask).sum() / mask.sum().clamp_min(eps)



class Segmentation(nn.Module):
    def __init__(
            self,
            sar_channels=1,
            opt_channels=3,
            num_classes=1000,
            backbone='resnet50',
            decoder_channels=(256, 128, 64, 32, 16),
            opt_mean=(0.5, 0.5, 0.5),
            opt_std=(0.5, 0.5, 0.5),
            sar_mean=(0.5,),
            sar_std=(0.5,),
    ):
        super(Segmentation, self).__init__()
        self._init_input_normalization(opt_mean, opt_std, sar_mean, sar_std)
        self.Unet_SAR = Unet(encoder_name=backbone,
                             encoder_depth=5,
                             encoder_weights="imagenet",
                             classes=num_classes,
                             in_channels=sar_channels,
                             decoder_channels=decoder_channels,
                             )

        self.Unet_OPT = Unet(encoder_name=backbone,
                             encoder_depth=5,
                             encoder_weights="imagenet",
                             classes=num_classes,
                             in_channels=opt_channels,
                             decoder_channels=decoder_channels,
                             )

        self.sar2opt_concat_levels = len(self.Unet_SAR.encoder.out_channels) - 2
        self.sar2opt_high_proj_channels = {4: 128, 5: 256}
        self.sar2opt_high_projs = nn.ModuleDict({
            str(level): nn.Conv2d(
                self.Unet_SAR.encoder.out_channels[level],
                out_channels,
                kernel_size=1,
            )
            for level, out_channels in self.sar2opt_high_proj_channels.items()
        })
        sar2opt_encoder_channels = [
            channels * 2 if i < self.sar2opt_concat_levels
            else channels + self.sar2opt_high_proj_channels.get(i, 0)
            for i, channels in enumerate(self.Unet_SAR.encoder.out_channels)
        ]

        self.SAR2OPT_Decoder = UnetDecoder(
            encoder_channels=sar2opt_encoder_channels,
            decoder_channels=decoder_channels,
            n_blocks=5,
            use_batchnorm=True,
            center=False,
            attention_type=None,
        )

        self.SAR2OPT_Decoder_Head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=opt_channels,
            activation='tanh',
            kernel_size=3,
        )

        self.Fusion_Segmentation_Head1 = SegmentationHead(
            in_channels=decoder_channels[-1]*2,
            out_channels=num_classes,
            activation=None,
            kernel_size=3,
        )

        self.Fusion_Segmentation_Head2 = SegmentationHead(
            in_channels=decoder_channels[-1] * 4,
            out_channels=num_classes,
            activation=None,
            kernel_size=3,
        )

        self.SAR_Enhance_Decoder = UnetDecoder(
            encoder_channels=self.Unet_SAR.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=5,
            use_batchnorm=True,
            center=False,
            attention_type=None,
        )

        self.SAR_Enhance_Decoder_Head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=sar_channels,
            activation='tanh',
            kernel_size=3,
        )

        self.num_classes = num_classes
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean')

        self.prototype_buffer = PrototypeBuffer(num_classes, self.Unet_OPT.encoder.out_channels)
        self.SAR_EdgeEnhancer = SAR_EdgeEnhancer(
            sar_channels=self.Unet_SAR.encoder.out_channels,
            guide_channels=self.Unet_OPT.encoder.out_channels,
            num_enhance_levels=3,
            bottleneck_channels=64,
        )

    def _init_input_normalization(self, opt_mean, opt_std, sar_mean, sar_std):
        self.register_buffer(
            "_opt_mean",
            torch.as_tensor(opt_mean, dtype=torch.float32).view(1, -1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_opt_std",
            torch.as_tensor(opt_std, dtype=torch.float32).view(1, -1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_sar_mean",
            torch.as_tensor(sar_mean, dtype=torch.float32).view(1, -1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_sar_std",
            torch.as_tensor(sar_std, dtype=torch.float32).view(1, -1, 1, 1),
            persistent=False,
        )

    @staticmethod
    def _denormalize(x, mean, std):
        return (x * std + mean).clamp_(0.0, 1.0)

    def build_sar2opt_features(self, sar_feats, restored_feats):
        out = []
        for i, (sar_feat, restored_feat) in enumerate(zip(sar_feats, restored_feats)):
            if i < self.sar2opt_concat_levels:
                out.append(torch.cat([sar_feat, restored_feat], dim=1))
            elif i in self.sar2opt_high_proj_channels:
                sar_proj = self.sar2opt_high_projs[str(i)](sar_feat)
                out.append(torch.cat([sar_proj, restored_feat], dim=1))
            else:
                out.append(restored_feat)
        return out

    def compute_cloud_instance_flags(self, img_opt, img_sar, sam_masks, valid_mask):
        """
        img_opt:   [B,3,H,W] normalized
        img_sar:   [B,1,H,W] normalized
        sam_masks: [B,N,H,W]
        valid_mask: [B,N]
        """
        B, N, H, W = sam_masks.shape
        eps = 1e-6

        # =============== 1) denorm to [0,1] ================
        opt01 = self._denormalize(img_opt, self._opt_mean, self._opt_std)
        sar01 = self._denormalize(img_sar, self._sar_mean, self._sar_std)

        # Convert to grayscale for wavelet (opt)
        gray = opt01.mean(1, keepdim=True)  # [B,1,H,W]

        # =============== 2) Haar wavelet transform ================
        # Wavelet kernels
        lp = torch.tensor([1 / 2, 1 / 2], device=opt01.device).view(1, 1, 1, 2)
        hp = torch.tensor([1 / 2, -1 / 2], device=opt01.device).view(1, 1, 1, 2)

        # --- Row transform (H, W -> H, W/2)
        LL_r = F.conv2d(gray, lp, stride=(1, 2))  # [B,1,H,W/2]
        LH_r = F.conv2d(gray, hp, stride=(1, 2))

        LL_sar_r = F.conv2d(sar01, lp, stride=(1, 2))
        LH_sar_r = F.conv2d(sar01, hp, stride=(1, 2))

        # --- Column transform (H, W/2 -> H/2, W/2)
        lp_t = lp.transpose(2, 3)
        hp_t = hp.transpose(2, 3)

        LL = F.conv2d(LL_r, lp_t, stride=(2, 1))  # [B,1,H/2,W/2]
        HL = F.conv2d(LL_r, hp_t, stride=(2, 1))
        LH = F.conv2d(LH_r, lp_t, stride=(2, 1))
        HH = F.conv2d(LH_r, hp_t, stride=(2, 1))

        LL_sar = F.conv2d(LL_sar_r, lp_t, stride=(2, 1))
        HL_sar = F.conv2d(LL_sar_r, hp_t, stride=(2, 1))
        LH_sar = F.conv2d(LH_sar_r, lp_t, stride=(2, 1))
        HH_sar = F.conv2d(LH_sar_r, hp_t, stride=(2, 1))


        # ============ 3) flatten SAM mask to wavelet size ============
        sam_w = F.interpolate(sam_masks, size=LL.shape[-2:], mode='nearest')  # [B,N,h,w]
        m = sam_w.view(B, N, -1)
        area = m.sum(-1) + eps  # [B,N]

        # ============ 4) compute wavelet stats per instance =============
        def inst_mean(feat):  # feat: [B,1,h,w]
            f = feat.view(B, 1, -1)  # [B,1,hw]
            num = (f * m).sum(-1)  # [B,N]
            return num / area

        # low freq diff
        low_diff = (LL - LL_sar).abs()
        low_diff_mean = inst_mean(low_diff)

        # high freq smoothness (cloud => smoother => high freq low)
        hf_energy = (LH ** 2 + HL ** 2 + HH ** 2)
        hf_smooth = -inst_mean(hf_energy)  # Internal computation.

        # Internal computation.
        hf_diff = (LH - LH_sar).abs() + (HL - HL_sar).abs() + (HH - HH_sar).abs()
        hf_diff_mean = inst_mean(hf_diff)

        # ============= 5) normalize =============
        def norm(x):
            valid = valid_mask.bool()
            xmin = x.masked_fill(~valid, torch.inf).min(1, keepdim=True)[0]
            xmax = x.masked_fill(~valid, -torch.inf).max(1, keepdim=True)[0]
            has_valid = valid.any(1, keepdim=True)
            xmin = torch.where(has_valid, xmin, torch.zeros_like(xmin))
            xmax = torch.where(has_valid, xmax, torch.ones_like(xmax))
            span = xmax - xmin
            normalized = (x - xmin) / span.clamp_min(eps)
            normalized = torch.where(
                span > eps, normalized, torch.full_like(normalized, 0.5)
            )
            return torch.where(valid, normalized, torch.zeros_like(normalized))

        f1 = norm(low_diff_mean)
        f2 = norm(hf_smooth)
        f3 = norm(hf_diff_mean)

        # ============= 6) final cloud score =============
        cloud_score = 0.4 * f1 + 0.2 * f2 + 0.4 * f3

        cloud_score = cloud_score - (1 - valid_mask) * 9999

        # Exposed for inference-time threshold sensitivity analysis. The
        # checkpoint is unchanged; the default preserves the original behavior.
        cloud_threshold = getattr(self, "cloud_threshold", 0.3)
        is_cloud = (cloud_score > cloud_threshold).float()

        cloud_mask = (sam_masks * is_cloud[:, :, None, None]).sum(1, keepdim=True)
        cloud_mask = (cloud_mask > 0).float()

        # Test-time mask perturbations for causal robustness analysis. These
        # modes leave the checkpoint and candidate region provider unchanged.
        mask_ablation = getattr(self, "mask_ablation", "full")
        if mask_ablation == "no-mask":
            is_cloud = torch.zeros_like(is_cloud)
            cloud_mask = torch.zeros_like(cloud_mask)
        elif mask_ablation == "random-mask":
            shuffled = torch.zeros_like(is_cloud)
            for b in range(B):
                valid_indices = torch.nonzero(valid_mask[b] > 0, as_tuple=False).flatten()
                if valid_indices.numel() > 0:
                    permutation = valid_indices[torch.randperm(valid_indices.numel(), device=is_cloud.device)]
                    shuffled[b, valid_indices] = is_cloud[b, permutation]
            is_cloud = shuffled
            cloud_mask = (sam_masks * is_cloud[:, :, None, None]).sum(1, keepdim=True)
            cloud_mask = (cloud_mask > 0).float()
        elif mask_ablation == "shifted-mask":
            shift_h = max(1, H // 8)
            shift_w = max(1, W // 8)
            cloud_mask = torch.roll(cloud_mask, shifts=(shift_h, shift_w), dims=(-2, -1))

        return is_cloud, cloud_mask


    def restore_cloud_region_multi(self, feat_list, sar_feats, cloud_mask, labels):
        out = []
        for level, feat in enumerate(feat_list):
            B, C, H, W = feat.shape

            # Internal computation.
            label = F.interpolate(
                labels.unsqueeze(1).float(),
                size=(H, W),
                mode="nearest"
            ).long().squeeze(1)

            # Internal computation.
            if level:

               cloud_down = F.interpolate(cloud_mask.float(), size=(H, W), mode="nearest")
            else:
                cloud_down = cloud_mask.float()  # [B,1,H,W]

            # Internal computation.
            proto_m = self.prototype_buffer.proto_mean[level][label]  # [B,H,W,C]
            proto_s = self.prototype_buffer.proto_std[level][label]  # [B,H,W,C]

            proto_m = proto_m.permute(0, 3, 1, 2)  # Internal computation.
            proto_s = proto_s.permute(0, 3, 1, 2)



            if level < 0:
                sar_feat_level = sar_feats[level]  # Internal computation.
                flow = sar_structure_field(sar_feat_level)

                proto_warped = warp_prototype(proto_m, flow.detach())

                restored_cloud_feat = proto_m + 0.5 * proto_warped.detach()

            else :
                restored_cloud_feat = proto_m


            # new
            phi = 0.8
            final_feat = feat * (1.0 - cloud_down * 1.0) + restored_cloud_feat * (cloud_down * (1.0 - phi)) + feat * (
                        cloud_down * phi)

            out.append(final_feat)

        return out


    def multimodal_topk_instances(self,
                                  sam_masks, valid_mask, is_cloud,
                                  feat_opt, feat_sar, feat_sar2opt, feat_sar_enhanced,
                                  K=5):
        """
        sam_masks:     [B,N,H,W]
        valid_mask:    [B,N]
        is_cloud:      [B,N]
        feat_opt:      [B,C,H',W']
        feat_sar:      [B,C,H',W']
        """

        B, N, H, W = sam_masks.shape
        _, C, h, w = feat_opt.shape
        eps = 1e-6
        device = sam_masks.device

        # Internal computation.
        is_cloud = is_cloud.float()
        valid_mask = valid_mask.float()

        with torch.no_grad():
            # ---------------------------------------------------
            # Internal computation.
            # ---------------------------------------------------
            # Internal computation.
            m = sam_masks

            # ---------------------------------------------------
            # Internal computation.
            # ---------------------------------------------------
            area = m.sum((2, 3)) + eps  # [B,N]
            area_norm = area / (h * w)  # [B,N]

            # ---------------------------------------------------
            # Internal computation.
            # ---------------------------------------------------
            def region_mean(feat: torch.Tensor) -> torch.Tensor:  # feat: [B,C,h,w]
                f = feat.unsqueeze(1)  # [B,1,C,h,w]
                # r_sum: [B,N,C]
                r_sum = (f * m.unsqueeze(2)).sum((3, 4))
                # r: [B,N,C]
                r = r_sum / area.unsqueeze(-1)
                return r

            r_opt = region_mean(feat_opt)
            r_sar = region_mean(feat_sar)
            r_s2o = region_mean(feat_sar2opt)
            r_se = region_mean(feat_sar_enhanced)



            # ---------------------------------------------------
            # Internal computation.
            #    C = CosineSim(r_opt, r_sar)
            # ---------------------------------------------------
            C1 = F.cosine_similarity(r_opt, r_sar, dim=-1)  # [B,N]
            C2 = F.cosine_similarity(r_opt, r_s2o, dim=-1)
            C3 = F.cosine_similarity(r_sar, r_s2o, dim=-1)
            C4 = F.cosine_similarity(r_opt, r_se, dim=-1)
            C5 = F.cosine_similarity(r_sar, r_se, dim=-1)
            C6 = F.cosine_similarity(r_s2o, r_se, dim=-1)
            C = (C1 + C2 + C3 + C4 + C5 + C6) / 6

            # ---------------------------------------------------
            # Internal computation.
            #    D = 1 - CosineSim(r_opt, r_opt_bg)
            # ---------------------------------------------------
            def region_background_mean(feat: torch.Tensor) -> torch.Tensor:
                # Internal computation.
                # Internal computation.
                m_bg = 1.0 - m
                # Internal computation.
                area_bg = m_bg.sum((2, 3)) + eps

                # Internal computation.
                f = feat.unsqueeze(1)
                r_bg_sum = (f * m_bg.unsqueeze(2)).sum((3, 4))

                # Internal computation.
                r_bg = r_bg_sum / area_bg.unsqueeze(-1)
                return r_bg

            r_opt_bg = region_background_mean(feat_opt)

            # D = 1 - CosineSim(r_opt, r_opt_bg)
            D = 1.0 - F.cosine_similarity(r_opt, r_opt_bg, dim=-1)  # [B, N]

            # ---------------------------------------------------
            # Internal computation.
            #    Q = 0.5 * C + 0.2 * D + 0.3 * A
            # ---------------------------------------------------
            Q = (0.4 * C +
                 0.3 * D +
                 0.3 * area_norm)

            # ---------------------------------------------------
            # 7) Exclude degraded and invalid candidates from reliable fusion.
            # ---------------------------------------------------
            eligible = (valid_mask > 0.5) & (is_cloud < 0.5)
            Q = Q.masked_fill(~eligible, torch.finfo(Q.dtype).min)

            # ---------------------------------------------------
            # 8) Top-K index
            # ---------------------------------------------------
            K = min(K, N)
            # Internal computation.
            _, idx = torch.topk(Q, K, dim=1)  # [B,K]

            # ---------------------------------------------------
            # Internal computation.
            # ---------------------------------------------------
            # Internal computation.
            batch_indices = torch.arange(B, device=device).unsqueeze(1)

            # Internal computation.
            # Internal computation.
            selected_is_eligible = eligible[batch_indices, idx]
            topk = sam_masks[batch_indices, idx] * selected_is_eligible[:, :, None, None]

            # ---------------------------------------------------
            # Internal computation.
            # ---------------------------------------------------
            # Internal computation.
            topk_mask = (topk.sum(1, keepdim=True) > 0).float()  # [B, 1, H, W]

        return topk_mask


    def trainSeg1(self, img_sar, img_opt, label, SAM_mask, valid_mask):

        is_cloud, cloud_mask = self.compute_cloud_instance_flags(
            img_opt, img_sar, SAM_mask, valid_mask
        )

        sar_features = self.Unet_SAR.encoder(img_sar)
        sar_feature_cp = [x.clone().detach() for x in sar_features]
        sar_decoder_feature = self.Unet_SAR.decoder(*sar_features)
        sar_decoder_feature_cp = sar_decoder_feature.clone()
        sar_logits = self.Unet_SAR.segmentation_head(sar_decoder_feature)
        loss_ce_sar = self.ce_loss(sar_logits, label)

        opt_features = self.Unet_OPT.encoder(img_opt)
        opt_feature_cp = [x.clone().detach() for x in opt_features]
        opt_decoder_feature = self.Unet_OPT.decoder(*opt_features)
        opt_decoder_feature_cp = opt_decoder_feature.clone()
        opt_logits = self.Unet_OPT.segmentation_head(opt_decoder_feature)
        loss_ce_opt = self.ce_loss(opt_logits, label)

        fusion_logit1 = self.Fusion_Segmentation_Head1(torch.cat([
            sar_decoder_feature_cp,
            opt_decoder_feature_cp
        ], dim=1))

        loss_ce_fusion1 = self.ce_loss(fusion_logit1, label)

        restored_feats = self.restore_cloud_region_multi(opt_feature_cp, sar_feature_cp, cloud_mask, torch.argmax(fusion_logit1, dim=1))

        sar2opt = self.build_sar2opt_features(sar_feature_cp, restored_feats)

        sar_features_enhanced = self.SAR_EdgeEnhancer(sar_feature_cp, restored_feats, cloud_mask)

        sar2opt_decoder_feature = self.SAR2OPT_Decoder(*sar2opt)

        sar_enhanced_decoder_feature = self.SAR_Enhance_Decoder(*sar_features_enhanced)

        opt_recon = self.SAR2OPT_Decoder_Head(
            sar2opt_decoder_feature
        )

        sar_enhanced = self.SAR_Enhance_Decoder_Head(sar_enhanced_decoder_feature)

        loss_rec = F.l1_loss(opt_recon.float() * (1 - cloud_mask), img_opt.float() * (1 - cloud_mask))
        opt_gray = 0.299 * img_opt[:, 0:1] + 0.587 * img_opt[:, 1:2] + 0.114 * img_opt[:, 2:3]
        reliable_mask = 1.0 - cloud_mask
        reliable_ssim = (
            ssim(denorm_01(sar_enhanced), denorm_01(opt_gray)) * reliable_mask
        ).mean()
        loss_enhanced = (
            0.7 * F.l1_loss(sar_enhanced.float(), img_sar.float())
            + 0.3 * (1 - reliable_ssim)
        )

        with torch.no_grad():
            self.prototype_buffer.update(opt_feature_cp, torch.argmax(fusion_logit1, dim=1), cloud_mask)

        rho = 0.5
        loss = loss_ce_sar + loss_ce_opt + loss_ce_fusion1 + rho * loss_rec + rho * loss_enhanced

        return fusion_logit1, loss

    def trainSeg2(self, img_sar, img_opt, label, SAM_mask, valid_mask):

        is_cloud, cloud_mask = self.compute_cloud_instance_flags(
            img_opt, img_sar, SAM_mask, valid_mask
        )

        sar_features = self.Unet_SAR.encoder(img_sar)
        sar_feature_cp = [x.clone().detach() for x in sar_features]
        sar_decoder_feature = self.Unet_SAR.decoder(*sar_features)
        sar_decoder_feature_cp = sar_decoder_feature.clone()
        sar_logits = self.Unet_SAR.segmentation_head(sar_decoder_feature)
        loss_ce_sar = self.ce_loss(sar_logits, label)

        opt_features = self.Unet_OPT.encoder(img_opt)
        opt_feature_cp = [x.clone().detach() for x in opt_features]
        opt_decoder_feature = self.Unet_OPT.decoder(*opt_features)
        opt_decoder_feature_cp = opt_decoder_feature.clone()
        opt_logits = self.Unet_OPT.segmentation_head(opt_decoder_feature)
        loss_ce_opt = self.ce_loss(opt_logits, label)

        fusion_logit1 = self.Fusion_Segmentation_Head1(torch.cat([
            sar_decoder_feature_cp,
            opt_decoder_feature_cp
        ], dim=1))

        loss_ce_fusion1 = self.ce_loss(fusion_logit1, label)

        restored_feats = self.restore_cloud_region_multi(opt_feature_cp, sar_feature_cp, cloud_mask, torch.argmax(fusion_logit1, dim=1))

        sar2opt = self.build_sar2opt_features(sar_feature_cp, restored_feats)

        sar_features_enhanced = self.SAR_EdgeEnhancer(sar_feature_cp, restored_feats, cloud_mask)

        sar2opt_decoder_feature = self.SAR2OPT_Decoder(*sar2opt)
        sar2opt_decoder_feature_cp = sar2opt_decoder_feature.clone()

        sar_enhanced_decoder_feature = self.SAR_Enhance_Decoder(*sar_features_enhanced)
        sar_enhanced_decoder_feature_cp = sar_enhanced_decoder_feature.clone()

        loss_struct = cloud_structure_loss(
            sar2opt_decoder_feature_cp,
            sar_decoder_feature_cp.detach(),
            torch.argmax(fusion_logit1, dim=1),
            cloud_mask,
            self.num_classes
        )


        opt_recon = self.SAR2OPT_Decoder_Head(
            sar2opt_decoder_feature
        )

        sar_enhanced = self.SAR_Enhance_Decoder_Head(sar_enhanced_decoder_feature)

        loss_rec = F.l1_loss(opt_recon.float() * (1 - cloud_mask), img_opt.float() * (1 - cloud_mask))
        opt_gray = 0.299 * img_opt[:, 0:1] + 0.587 * img_opt[:, 1:2] + 0.114 * img_opt[:, 2:3]
        reliable_mask = 1.0 - cloud_mask
        reliable_ssim = (
            ssim(denorm_01(sar_enhanced), denorm_01(opt_gray)) * reliable_mask
        ).mean()
        loss_enhanced = (
            0.7 * F.l1_loss(sar_enhanced.float(), img_sar.float())
            + 0.3 * (1 - reliable_ssim)
        )

        with torch.no_grad():
            self.prototype_buffer.update(opt_feature_cp, torch.argmax(fusion_logit1, dim=1), cloud_mask)

        topk_mask = self.multimodal_topk_instances(
            SAM_mask,
            valid_mask,
            is_cloud,
            opt_decoder_feature_cp,
            sar_decoder_feature_cp,
            sar2opt_decoder_feature_cp,
            sar_enhanced_decoder_feature_cp,
            K=getattr(self, "top_k", 5)
        )

        fusion_logit2 = self.Fusion_Segmentation_Head2(torch.cat([opt_decoder_feature_cp,
                                                                  sar_decoder_feature_cp,
                                                                  sar2opt_decoder_feature_cp,
                                                                  sar_enhanced_decoder_feature_cp
                                                                  ], dim=1) * (1 + topk_mask))

        loss_ce_fusion2 = self.ce_loss(fusion_logit2, label)

        rho = 0.4
        phi = 0.4
        alpha = 0.5
        beta = 0.5
        loss = alpha * loss_ce_sar + alpha * loss_ce_opt + beta * loss_ce_fusion1 + loss_ce_fusion2 + rho * loss_rec + phi * loss_enhanced + .05 * loss_struct

        return fusion_logit2, loss

    def test(self, img_sar, img_opt, SAM_mask, valid_mask):

        is_cloud, cloud_mask = self.compute_cloud_instance_flags(
            img_opt, img_sar, SAM_mask, valid_mask
        )

        sar_features = self.Unet_SAR.encoder(img_sar)
        sar_feature_cp = [x.clone().detach() for x in sar_features]
        sar_decoder_feature = self.Unet_SAR.decoder(*sar_features)
        sar_decoder_feature_cp = sar_decoder_feature.clone()
        sar_logits = self.Unet_SAR.segmentation_head(sar_decoder_feature)

        opt_features = self.Unet_OPT.encoder(img_opt)
        opt_feature_cp = [x.clone().detach() for x in opt_features]
        opt_decoder_feature = self.Unet_OPT.decoder(*opt_features)
        opt_decoder_feature_cp = opt_decoder_feature.clone()
        opt_logits = self.Unet_OPT.segmentation_head(opt_decoder_feature)

        fusion_logit1 = self.Fusion_Segmentation_Head1(torch.cat([sar_decoder_feature_cp,
                                                                opt_decoder_feature_cp,
                                                                ], dim=1)
                                                     )
        restored_feats = self.restore_cloud_region_multi(opt_feature_cp, sar_feature_cp, cloud_mask, torch.argmax(fusion_logit1, dim=1))

        sar2opt = self.build_sar2opt_features(sar_feature_cp, restored_feats)
        sar2opt_decoder_feature = self.SAR2OPT_Decoder(*sar2opt)
        sar2opt_decoder_feature_cp = sar2opt_decoder_feature.clone()

        sar_features_enhanced = self.SAR_EdgeEnhancer(sar_feature_cp, restored_feats, cloud_mask)
        sar_enhanced_decoder_feature = self.SAR_Enhance_Decoder(*sar_features_enhanced)
        sar_enhanced_decoder_feature_cp = sar_enhanced_decoder_feature.clone()

        topk_mask = self.multimodal_topk_instances(
            SAM_mask,
            valid_mask,
            is_cloud,
            opt_decoder_feature_cp,
            sar_decoder_feature_cp,
            sar2opt_decoder_feature_cp,
            sar_enhanced_decoder_feature_cp,
            K=getattr(self, "top_k", 5)
        )

        fusion_logit2 = self.Fusion_Segmentation_Head2(torch.cat([opt_decoder_feature_cp,
                                                                sar_decoder_feature_cp,
                                                                sar2opt_decoder_feature_cp,
                                                                sar_enhanced_decoder_feature_cp
                                                                ], dim=1) * (1 + topk_mask))

        return fusion_logit2, sar_logits, opt_logits

    def forward(self, img_sar, img_opt, label, SAM_mask, valid_mask):
        return self.test(img_sar, img_opt, SAM_mask, valid_mask)


class ConvNeXtV2Segmentation(Segmentation):
    """Dual-modality DRM-Seg with ConvNeXtV2-Tiny encoders."""

    def __init__(self, sar_channels=1, opt_channels=3, num_classes=6,
                 image_size=128, pretrained=False,
                 decoder_channels=(256, 128, 64, 32, 16),
                 opt_mean=(0.5, 0.5, 0.5), opt_std=(0.5, 0.5, 0.5),
                 sar_mean=(0.5,), sar_std=(0.5,), **_):
        nn.Module.__init__(self)
        self._init_input_normalization(opt_mean, opt_std, sar_mean, sar_std)
        opt_encoder = ConvNeXtV2PyramidEncoder(opt_channels, image_size, pretrained)
        sar_encoder = ConvNeXtV2PyramidEncoder(sar_channels, image_size, pretrained)
        self.Unet_OPT = ConvNeXtV2UNetBranch(
            opt_channels, decoder_channels, num_classes, image_size, pretrained, opt_encoder
        )
        self.Unet_SAR = ConvNeXtV2UNetBranch(
            sar_channels, decoder_channels, num_classes, image_size, pretrained, sar_encoder
        )

        sar_pyramid = tuple(self.Unet_SAR.encoder.out_channels)
        opt_pyramid = tuple(self.Unet_OPT.encoder.out_channels)
        self.sar2opt_concat_levels = len(sar_pyramid) - 2
        self.sar2opt_high_proj_channels = {4: 96, 5: 192}
        self.sar2opt_high_projs = nn.ModuleDict({
            str(level): nn.Conv2d(sar_pyramid[level], channels, 1)
            for level, channels in self.sar2opt_high_proj_channels.items()
        })
        sar2opt_pyramid = [
            sar_pyramid[i] + opt_pyramid[i]
            if i < self.sar2opt_concat_levels
            else opt_pyramid[i] + self.sar2opt_high_proj_channels.get(i, 0)
            for i in range(len(sar_pyramid))
        ]
        decoder_kwargs = dict(decoder_channels=decoder_channels, n_blocks=5,
                              use_batchnorm=True, center=False, attention_type=None)
        self.SAR2OPT_Decoder = UnetDecoder(encoder_channels=sar2opt_pyramid, **decoder_kwargs)
        self.SAR_Enhance_Decoder = UnetDecoder(encoder_channels=sar_pyramid, **decoder_kwargs)
        self.SAR2OPT_Decoder_Head = SegmentationHead(
            in_channels=decoder_channels[-1], out_channels=opt_channels,
            activation="tanh", kernel_size=3
        )
        self.SAR_Enhance_Decoder_Head = SegmentationHead(
            in_channels=decoder_channels[-1], out_channels=sar_channels,
            activation="tanh", kernel_size=3
        )
        self.Fusion_Segmentation_Head1 = SegmentationHead(
            in_channels=decoder_channels[-1] * 2, out_channels=num_classes,
            activation=None, kernel_size=3
        )
        self.Fusion_Segmentation_Head2 = SegmentationHead(
            in_channels=decoder_channels[-1] * 4, out_channels=num_classes,
            activation=None, kernel_size=3
        )
        self.num_classes = num_classes
        self.top_k = 5
        self.ce_loss = nn.CrossEntropyLoss()
        self.prototype_buffer = PrototypeBuffer(num_classes, opt_pyramid)
        self.SAR_EdgeEnhancer = SAR_EdgeEnhancer(
            sar_channels=sar_pyramid, guide_channels=opt_pyramid,
            num_enhance_levels=3, bottleneck_channels=64
        )


class DualConvNeXtV2Segmentation(ConvNeXtV2Segmentation):
    """Public model name used by the standalone train and test scripts."""

    pass


if __name__ == "__main__":
    model = DualConvNeXtV2Segmentation(pretrained=False)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.3f}M")


