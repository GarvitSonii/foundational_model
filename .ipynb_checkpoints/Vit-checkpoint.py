import math
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# 1) 3D Patch Embedding (spatial 2D + spectral groups)
# -----------------------------
class Spectral3DPatchEmbed(nn.Module):
    """
    Turn (B, C, H, W) into tokens across space and spectral groups.
    - Split spectral channels into groups of size k (stride k).
    - Extract non-overlapping 2D patches of size (ph, pw) for each spectral group.
    - Project patches -> embed_dim.
    Positional embeddings:
      - spatial_pe: (1, Np, D) for spatial tokens
      - spectral_pe: (1, Ng, D) for spectral groups
    Output tokens are shaped as (B, Ng*Np, D).
    """
    def __init__(
        self,
        in_ch: int,
        embed_dim: int = 768,
        patch_size: Tuple[int, int] = (8, 8),
        spec_group: int = 3,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.embed_dim = embed_dim
        self.ph, self.pw = patch_size
        self.k = spec_group

        # We'll unfold 2D patches; projection happens per spectral group.
        patch_vec = self.ph * self.pw * self.k  # k channels per token * ph*pw
        self.proj = nn.Linear(patch_vec, embed_dim)

        # Lazy buffers (set at first forward, then registered)
        self.register_buffer("spatial_pe", None, persistent=False)
        self.register_buffer("spectral_pe", None, persistent=False)

    def _build_pos_embed(self, Hp: int, Wp: int, Ng: int, device):
        Np = Hp * Wp
        spatial_pe = nn.Parameter(torch.zeros(1, Np, self.embed_dim, device=device))
        spectral_pe = nn.Parameter(torch.zeros(1, Ng, self.embed_dim, device=device))
        nn.init.trunc_normal_(spatial_pe, std=0.02)
        nn.init.trunc_normal_(spectral_pe, std=0.02)
        # Register as buffers so they move with .to(device) but are not optimized by default
        # (you can switch to Parameters if you prefer to train them explicitly)
        self.register_buffer("spatial_pe", spatial_pe.data, persistent=False)
        self.register_buffer("spectral_pe", spectral_pe.data, persistent=False)

    @staticmethod
    def _pad_channels(x: torch.Tensor, k: int) -> Tuple[torch.Tensor, int]:
        # pad channel dimension so it is divisible by k
        B, C, H, W = x.shape
        rem = C % k
        if rem == 0:
            return x, C
        pad = k - rem
        x = F.pad(x, (0, 0, 0, 0, 0, pad))  # pad channels at end
        return x, C + pad

    def forward(self, x: torch.Tensor):
        """
        x: (B, C, H, W)
        returns:
          tokens: (B, Ng*Np, D)
          Hp, Wp: patch grid size
          Ng: number of spectral groups
        """
        B, C, H, W = x.shape
        assert H % self.ph == 0 and W % self.pw == 0, "H,W must be divisible by patch size"

        x, Cp = self._pad_channels(x, self.k)
        Ng = Cp // self.k   # number of spectral groups
        Hp, Wp = H // self.ph, W // self.pw
        Np = Hp * Wp

        # group spectral channels
        x = x.view(B, Ng, self.k, H, W)   # (B, Ng, k, H, W)
        # unfold 2D patches: do per-group unfold
        # We'll rearrange to (B, Ng, k*ph*pw, Np)
        unfold = nn.Unfold(kernel_size=(self.ph, self.pw), stride=(self.ph, self.pw))
        x = x.reshape(B * Ng, self.k, H, W)   # merge group
        patches = unfold(x)                   # (B*Ng, k*ph*pw, Np)
        patches = patches.transpose(1, 2)     # (B*Ng, Np, k*ph*pw)
        patches = self.proj(patches)          # (B*Ng, Np, D)

        # restore (B, Ng, Np, D)
        tokens = patches.view(B, Ng, Np, self.embed_dim)

        # build positional embeddings once per spatial/spectral size
        if (self.spatial_pe is None) or (self.spatial_pe.shape[1] != Np) or (self.spectral_pe.shape[1] != Ng):
            self._build_pos_embed(Hp, Wp, Ng, tokens.device)

        tokens = tokens + self.spatial_pe.unsqueeze(1) + self.spectral_pe.unsqueeze(2)  # broadcast
        tokens = tokens.reshape(B, Ng * Np, self.embed_dim)  # flatten to seq

        return tokens, Hp, Wp, Ng


# -----------------------------
# 2) ViT encoder (vanilla)
# -----------------------------
class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=12, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, v, k = qkv.permute(2, 0, 3, 1, 4)  # (3,B,heads,N,head_dim); careful—k,v swapped is ok if consistent
        # Use standard q,k,v order: fix to q,k,v
        q, k, v = qkv[..., 0, :, :], qkv[..., 1, :, :], qkv[..., 2, :, :]
        q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder3D(nn.Module):
    """
    ViT encoder on top of Spectral3DPatchEmbed.
    Returns:
      seq (B, N, D) and a reshaped 2D feature map (B, D, Hp, Wp) from the last layer.
    """
    def __init__(
        self,
        in_ch: int = 4,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        patch_size: Tuple[int, int] = (8, 8),
        spec_group: int = 3,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.patch = Spectral3DPatchEmbed(in_ch, embed_dim, patch_size, spec_group)
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, drop, attn_drop) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        tokens, Hp, Wp, Ng = self.patch(x)            # (B, Ng*Np, D)
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        B, N, D = tokens.shape
        feat2d = tokens.transpose(1, 2).reshape(B, D, Hp, Wp)  # (B, D, Hp, Wp)
        return tokens, feat2d, (Hp, Wp)


# -----------------------------
# 3) Neck: make a feature pyramid from ViT grid
# -----------------------------
class SimpleFPNNeck(nn.Module):
    """
    From a single-scale ViT grid (B, D, Hp, Wp) create 4 pyramid levels:
    P2 ~ 1/4, P3 ~ 1/8, P4 ~ 1/16, P5 ~ 1/32 of input.
    Given patch_size=8, Hp=H/8, so we treat that as P3 and build P2 via upsample, P4,P5 via downsample.
    """
    def __init__(self, in_dim: int, out_dims: List[int] = [256, 256, 256, 256]):
        super().__init__()
        assert len(out_dims) == 4
        self.lateral3 = nn.Conv2d(in_dim, out_dims[1], 1)  # base level (P3)
        self.to_p2 = nn.Conv2d(out_dims[1], out_dims[0], 3, padding=1)
        self.to_p4 = nn.Conv2d(out_dims[1], out_dims[2], 3, stride=2, padding=1)
        self.to_p5 = nn.Conv2d(out_dims[2], out_dims[3], 3, stride=2, padding=1)

    def forward(self, feat: torch.Tensor) -> List[torch.Tensor]:
        # feat is ViT grid at ~1/8 input (if patch=8)
        p3 = self.lateral3(feat)                  # (B, C3, Hp, Wp)
        p2 = F.interpolate(p3, scale_factor=2, mode="bilinear", align_corners=False)
        p2 = self.to_p2(p2)
        p4 = self.to_p4(p3)
        p5 = self.to_p5(p4)
        return [p2, p3, p4, p5]  # low->high


# -----------------------------
# 4) UPerNet Head (PPM + FPN fuse)
# -----------------------------
class PPM(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bins=(1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(b),
                nn.Conv2d(in_dim, out_dim, 1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True),
            ) for b in bins
        ])
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim + len(bins)*out_dim, out_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        h, w = x.shape[-2:]
        pyramids = [x]
        for stage in self.stages:
            pyramids.append(F.interpolate(stage(x), size=(h, w), mode="bilinear", align_corners=False))
        x = torch.cat(pyramids, dim=1)
        return self.conv(x)


class UPerHead(nn.Module):
    """
    Minimal UPerNet-style head:
      - PPM on top (P5)
      - Top-down FPN fusion to P4, P3, P2
      - Final segmentation conv
    """
    def __init__(self, in_dims: List[int], ppm_dim: int = 256, num_classes: int = 2):
        super().__init__()
        assert len(in_dims) == 4
        self.ppm = PPM(in_dims[3], ppm_dim)  # P5 in -> ppm
        # lateral convs to unify channels
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, ppm_dim, 1, bias=False) for c in in_dims[:3]
        ])
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ppm_dim, ppm_dim, 3, padding=1, bias=False),
                nn.BatchNorm2d(ppm_dim),
                nn.ReLU(inplace=True)
            ) for _ in range(3)
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(ppm_dim * 4, ppm_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(ppm_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(ppm_dim, num_classes, 1)

    def forward(self, feats: List[torch.Tensor]):
        # feats: [P2, P3, P4, P5]
        p2, p3, p4, p5 = feats
        p5 = self.ppm(p5)
        # top-down
        p4 = self._td(self.lateral_convs[2](p4), p5)
        p3 = self._td(self.lateral_convs[1](p3), p4)
        p2 = self._td(self.lateral_convs[0](p2), p3)

        # smooth
        p4 = self.fpn_convs[2](p4)
        p3 = self.fpn_convs[1](p3)
        p2 = self.fpn_convs[0](p2)

        # upsample & fuse to P2 size
        h, w = p2.shape[-2:]
        p3u = F.interpolate(p3, size=(h, w), mode="bilinear", align_corners=False)
        p4u = F.interpolate(p4, size=(h, w), mode="bilinear", align_corners=False)
        p5u = F.interpolate(p5, size=(h, w), mode="bilinear", align_corners=False)
        x = torch.cat([p2, p3u, p4u, p5u], dim=1)
        x = self.fuse(x)
        return self.classifier(x)

    @staticmethod
    def _td(lateral: torch.Tensor, top: torch.Tensor):
        h, w = lateral.shape[-2:]
        top = F.interpolate(top, size=(h, w), mode="bilinear", align_corners=False)
        return lateral + top


# -----------------------------
# 5) Full model
# -----------------------------
class CloudSegSpectralViT(nn.Module):
    """
    End-to-end model: ViT encoder (3D tokens) + simple FPN neck + UPer head.
    Upsamples logits back to input HxW.
    """
    def __init__(
        self,
        in_ch: int = 4,                # e.g., RGB+NIR for Cloud95
        num_classes: int = 2,          # cloud / not-cloud
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        patch_size: Tuple[int, int] = (8, 8),
        spec_group: int = 3,
        neck_dims: List[int] = [256, 256, 256, 256],
        ppm_dim: int = 256,
    ):
        super().__init__()
        self.encoder = ViTEncoder3D(
            in_ch=in_ch,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            patch_size=patch_size,
            spec_group=spec_group,
        )
        self.neck = SimpleFPNNeck(in_dim=embed_dim, out_dims=neck_dims)
        self.decode = UPerHead(in_dims=neck_dims, ppm_dim=ppm_dim, num_classes=num_classes)
        self.patch_size = patch_size

    def forward(self, x):
        """
        x: (B, C, H, W)  with H,W divisible by patch_size
        """
        B, C, H, W = x.shape
        _, grid, _ = self.encoder(x)         # (B, D, Hp, Wp)
        feats = self.neck(grid)              # [P2,P3,P4,P5] at ~1/4..1/32 of input
        logits = self.decode(feats)          # (B, num_classes, ~H/4, ~W/4)
        # Upsample to input resolution
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        return logits


# -----------------------------
# 6) Quick test
# -----------------------------
if __name__ == "__main__":
    torch.set_grad_enabled(False)
    B, C, H, W = 2, 4, 256, 256   # Cloud95: adjust H,W to your tiles (e.g., 128 or 256)
    model = CloudSegSpectralViT(
        in_ch=C,
        num_classes=2,
        embed_dim=768,
        depth=8,            # you can start smaller for speed
        num_heads=8,
        patch_size=(8, 8),
        spec_group=3,       # k=3 like the paper; works with 4 channels (will pad 1 channel internally)
    )
    x = torch.randn(B, C, H, W)
    y = model(x)
    print("logits:", y.shape)  # (B, 2, H, W)
