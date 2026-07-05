"""NullSwap generator + PatchGAN discriminator (Wang et al. 2025, arXiv:2503.18678v1).

v11:
- Image scale unified to [-1, 1] across the pipeline. Dataloader emits [-1, 1];
  G/D no longer rescale internally; G output is the raw tanh result in [-1, 1].
  infer.py rescales [-1, 1] → [0, 255] uint8 only for storage.
- Discriminator restructured to 5 conv layers (4 ConvBlock + 1 bare Conv logit
  head) per paper's "ConvBlock" wording. Replaces the LeakyReLU PatchGAN.
- PerturbationBlock: added the second ConvBlock (post-SEResBlock) per paper
  Fig. 2 — ConvBlock → SEResBlock×M → ConvBlock → +noise.

v10:
- Removed v6 D4 multi-level skips. Paper Section 3.2 specifies CloakingBlock
  Stage 1 as "SEResBlock + DeConv + Conv + DeConv (no skip)" — the "two
  reconstructions at different dimension levels" refers to feature-level
  (Stage 1) vs image-level (Stage 2), not U-Net skips at intermediate
  resolutions.

Retained deviations (forced by training dynamics, see git history):
- Stage 2 final layer is bare Conv → tanh instead of Conv+BN+ReLU.
  BN-final pins output std to 1, blocking pixel values across the full output
  range; tanh is the working substitute.
- Direct synthesis (not residual I'_s = ReLU(I_s + δ)): δ ≈ 0 init gives a
  zero-gradient saddle (MSE grad = 0 and cos=1 grad = 0 simultaneously).

Input/Output: 256x256 RGB float tensor in [-1, 1]."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from blocks import ConvBlock, SEResBlock, DeconvBlock


class IdentityExtractor(nn.Module):
    """ConvBlock + max-pool + L=4 SEResBlocks → identity feature matrix.
    Input 256x256x3 → 128x128x64."""

    def __init__(self, in_ch=3, base_ch=64, L=4):
        super().__init__()
        self.stem = ConvBlock(in_ch, base_ch, kernel_size=7, stride=1, padding=3, use_pool=True)
        self.blocks = nn.Sequential(*[SEResBlock(base_ch, base_ch) for _ in range(L)])

    def forward(self, x):
        return self.blocks(self.stem(x))


class PerturbationBlock(nn.Module):
    """ConvBlock (128→64 spatial) + M=3 SEResBlocks + ConvBlock (post-aggregation refinement)
    + adaptive random noise injection: RandNoise = β · (α · RandNoise + η), α, β learnable
    scalars, η learnable per-channel. Paper Fig. 2 shows two ConvBlocks in this block: one
    before and one after the SEResBlock chain."""

    def __init__(self, in_ch=64, out_ch=128, M=3):
        super().__init__()
        self.refine = ConvBlock(in_ch, out_ch, stride=2)        # 128→64 spatial, 64→128 ch
        self.blocks = nn.Sequential(*[SEResBlock(out_ch, out_ch) for _ in range(M)])
        self.post = ConvBlock(out_ch, out_ch, stride=1)         # post-aggregation ConvBlock (paper Fig. 2)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.1))
        self.eta = nn.Parameter(torch.zeros(out_ch, 1, 1))

    def forward(self, ident_feats):
        p = self.post(self.blocks(self.refine(ident_feats)))
        noise = torch.randn_like(p)
        rand = self.beta * (self.alpha * noise + self.eta)
        return p + rand


class FeatureBlock(nn.Module):
    """3 ConvBlocks + N=5 SEResBlocks.
    Input 256x256x3 → 64x64x256 bottleneck. The 4× spatial downsample (matching the
    2 DeConvBlocks in CloakingBlock Stage 1) is achieved by 2 stride-2 ConvBlocks +
    1 stride-1 ConvBlock for channel adjustment."""

    def __init__(self, in_ch=3, base=64, N=5):
        super().__init__()
        self.c1 = ConvBlock(in_ch, base, stride=2)         # 256→128, 64ch
        self.c2 = ConvBlock(base, base * 2, stride=2)      # 128→64,  128ch
        self.c3 = ConvBlock(base * 2, base * 4, stride=1)  # 64→64,   256ch
        self.blocks = nn.Sequential(*[SEResBlock(base * 4, base * 4) for _ in range(N)])

    def forward(self, x):
        return self.blocks(self.c3(self.c2(self.c1(x))))   # 64x64x256


class CloakingBlock(nn.Module):
    """Paper Section 3.2: two-stage reconstruction with no intermediate skips.

    Stage 1 — feature-level (64×64 → 256×256):
        concat(γ·perturbation, feat_bn) → SEResBlock → DeConv → ConvBlock → DeConv.
    Stage 2 — image-level (256×256):
        concat(original_input, Stage1_out) → 3 sequential ConvBlocks → I'_s.
        The 3rd "ConvBlock" is bare Conv → tanh instead of Conv+BN+ReLU; BN-final
        pins output std to 1 and blocks the full output range (empirically MSE
        never decreases otherwise). Output is in [-1, 1]."""

    def __init__(self, pert_ch=128, feat_ch=256, in_img_ch=3, out_ch=3):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(0.1))

        # Stage 1: SEResBlock + DeConv + ConvBlock + DeConv (no intermediate skips).
        self.combine = SEResBlock(pert_ch + feat_ch, feat_ch)            # 64x64x384 → 64x64x256
        self.up1 = DeconvBlock(feat_ch, feat_ch // 2)                    # 64→128, 256→128
        self.conv_mid = ConvBlock(feat_ch // 2, feat_ch // 2)            # 128→128, 128→128
        self.up2 = DeconvBlock(feat_ch // 2, feat_ch // 4)               # 128→256, 128→64

        # Stage 2: concat(I_s, fused) + 3 ConvBlocks (last is tanh-rescale, see class docstring).
        self.post1 = ConvBlock(in_img_ch + feat_ch // 4, feat_ch // 4)   # 67 → 64
        self.post2 = ConvBlock(feat_ch // 4, feat_ch // 8)               # 64 → 32
        self.post3 = nn.Conv2d(feat_ch // 8, out_ch, kernel_size=3, padding=1, bias=True)
        nn.init.kaiming_normal_(self.post3.weight, nonlinearity='linear')
        self.post3.weight.data.mul_(0.1)
        nn.init.zeros_(self.post3.bias)

    def forward(self, perturbation, feat_bn, original_input_m11):
        # Stage 1: feature-level reconstruction.
        x = torch.cat([perturbation * self.gamma, feat_bn], dim=1)       # 64x64 x 384
        x = self.combine(x)                                              # 64x64 x 256
        x = self.up1(x)                                                  # 128x128 x 128
        x = self.conv_mid(x)                                             # 128x128 x 128
        x = self.up2(x)                                                  # 256x256 x 64

        # Stage 2: image-level reconstruction.
        x = torch.cat([original_input_m11, x], dim=1)                    # 256x256 x 67
        x = self.post1(x)
        x = self.post2(x)
        return torch.tanh(self.post3(x))                                 # [-1, 1]


class NullSwapGenerator(nn.Module):
    """NullSwap generator. Input/Output: image tensor in [-1, 1]."""

    def __init__(self):
        super().__init__()
        self.ident = IdentityExtractor()
        self.pert = PerturbationBlock()
        self.feat = FeatureBlock()
        self.cloak = CloakingBlock()

    def forward(self, x_m11):
        ident = self.ident(x_m11)
        perturbation = self.pert(ident)
        feat_bn = self.feat(x_m11)
        return self.cloak(perturbation, feat_bn, x_m11)


class PatchDiscriminator(nn.Module):
    """5-layer ConvBlock discriminator. Hidden 4 layers are ConvBlock (Conv+BN+ReLU
    per generator's ConvBlock spec), final layer is bare Conv logit head. Strides
    2,2,2,1,1 give a 70×70 receptive field at 256×256 input. Input in [-1, 1]."""

    def __init__(self, in_ch=3, base=64):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(in_ch, base, kernel_size=4, stride=2, padding=1),         # 256→128, 64ch
            ConvBlock(base, base * 2, kernel_size=4, stride=2, padding=1),      # 128→64, 128ch
            ConvBlock(base * 2, base * 4, kernel_size=4, stride=2, padding=1),  # 64→32, 256ch
            ConvBlock(base * 4, base * 8, kernel_size=4, stride=1, padding=1),  # 32→31, 512ch
            nn.Conv2d(base * 8, 1, kernel_size=4, stride=1, padding=1),         # 31→30, 1ch
        )

    def forward(self, x_m11):
        return self.net(x_m11)
