# snn_models.py
"""
SNN versions of UNet-like modules using SpikingJelly.
- TemporalSpikingUNet: runs input for `T` timesteps, accumulates output spikes and returns averaged logits.
- The implementation replaces ReLU activations with LIF neurons (surrogate gradients).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# SpikingJelly imports
from spikingjelly.activation_based import neuron, surrogate

# ---------------------------
# Building blocks
# ---------------------------
class SNNDoubleConv(nn.Module):
    """
    Two conv layers + batchnorm (or group norm) followed by LIF neurons.
    Each conv is followed by a neuron (membrane) which persists across timesteps.
    """
    def __init__(self, in_ch, out_ch, use_gn=False, groups=8, lif_kwargs=None):
        super().__init__()
        if use_gn:
            norm1 = nn.GroupNorm(groups, out_ch)
            norm2 = nn.GroupNorm(groups, out_ch)
        else:
            norm1 = nn.BatchNorm2d(out_ch)
            norm2 = nn.BatchNorm2d(out_ch)

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = norm1
        self.lif1 = neuron.LIFNode(surrogate_function=surrogate.Sigmoid(), detach_reset=True, **(lif_kwargs or {}))

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = norm2
        self.lif2 = neuron.LIFNode(surrogate_function=surrogate.Sigmoid(), detach_reset=True, **(lif_kwargs or {}))

    def forward_step(self, x):
        """
        forward for a single timestep (x shape: B,C,H,W)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.lif1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.lif2(x)
        return x

    def reset(self):
        # Reset internal LIF nodes' membrane potentials
        if hasattr(self.lif1, 'reset'):
            self.lif1.reset()
        if hasattr(self.lif2, 'reset'):
            self.lif2.reset()

class SNNUpBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c, lif_kwargs=None):
        super().__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_c, out_c, kernel_size=1, bias=False))
        self.dec = SNNDoubleConv(out_c + skip_c, out_c, use_gn=False, lif_kwargs=lif_kwargs)

    def forward_step(self, x, skip):
        u = self.up(x)
        cat = torch.cat([u, skip], dim=1)
        return self.dec.forward_step(cat)

    def reset(self):
        self.dec.reset()

# ---------------------------
# Temporal Spiking UNet (small, similar to UNetSmall in your models.py)
# ---------------------------
class TemporalSpikingUNet(nn.Module):
    def __init__(self, in_ch=4, out_ch=1, time_steps=8, lif_kwargs=None, device='cpu'):
        """
        time_steps: number of timesteps to run the SNN for each forward pass.
        lif_kwargs: dict of kwargs passed to LIFNode (tau, v_threshold, etc.)
        """
        super().__init__()
        self.T = time_steps
        self.device = device

        # channels similar to UNetSmall
        ch = [24, 48, 96, 192]
        self.down1 = SNNDoubleConv(in_ch, ch[0], use_gn=True, lif_kwargs=lif_kwargs)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = SNNDoubleConv(ch[0], ch[1], use_gn=True, lif_kwargs=lif_kwargs)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = SNNDoubleConv(ch[1], ch[2], use_gn=True, lif_kwargs=lif_kwargs)
        self.pool3 = nn.MaxPool2d(2)
        self.bottom = SNNDoubleConv(ch[2], ch[3], use_gn=True, lif_kwargs=lif_kwargs)

        self.up3 = SNNUpBlock(ch[3], ch[2], ch[2], lif_kwargs=lif_kwargs)
        self.up2 = SNNUpBlock(ch[2], ch[1], ch[1], lif_kwargs=lif_kwargs)
        self.up1 = SNNUpBlock(ch[1], ch[0], ch[0], lif_kwargs=lif_kwargs)
        self.head_conv = nn.Conv2d(ch[0], out_ch, kernel_size=1)  # logits per timestep

        # For proper device placement
        self.to(device)

    def reset(self):
        # reset all internal LIF nodes
        for m in self.modules():
            if m is self:
                continue
            if hasattr(m, 'reset') and callable(m.reset):
                try:
                    m.reset()
                except TypeError:
                    # Some modules maybe accept arguments; ignore
                    pass

    def forward(self, x):
        """
        x: (B, C, H, W) continuous-valued input image
        Returns: logits (B, out_ch, H, W) - averaged over time (sum of spikes / T)
        """
        B = x.shape[0]
        device = next(self.parameters()).device
        # Encode: simple repetition encoding (can replace with Poisson/other encoding)
        # create input per timestep: shape (T, B, C, H, W)
        # using a deterministic repeat here; for probabilistic spike encoding use PoissonEncoder.
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)

        # reset membrane states
        self.reset()

        out_accum = 0.0
        for t in range(self.T):
            xt = x_seq[t]

            # Encoder
            c1 = self.down1.forward_step(xt)
            p1 = self.pool1(c1)
            c2 = self.down2.forward_step(p1)
            p2 = self.pool2(c2)
            c3 = self.down3.forward_step(p2)
            p3 = self.pool3(c3)
            cb = self.bottom.forward_step(p3)

            # Decoder
            d3 = self.up3.forward_step(cb, c3)
            d2 = self.up2.forward_step(d3, c2)
            d1 = self.up1.forward_step(d2, c1)

            logits_t = self.head_conv(d1)  # logits from this timestep (B, out_ch, H, W)

            # accumulate (we will average later)
            out_accum = out_accum + logits_t

        out = out_accum / float(self.T)
        return out
