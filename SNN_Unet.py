import torch
import torch.nn as nn
import torch.nn.functional as F

# # ----------------------------
# # Surrogate spike function (fast sigmoid window)
# # ----------------------------
# class SurrogateSpike(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, mem_minus_thr, lens):
#         # mem_minus_thr: membrane - threshold
#         ctx.save_for_backward(mem_minus_thr)
#         ctx.lens = lens
#         out = (mem_minus_thr > 0).float()
#         return out

#     @staticmethod
#     def backward(ctx, grad_output):
#         mem_minus_thr, = ctx.saved_tensors
#         lens = ctx.lens
#         # triangular surrogate gradient: max(0, 1 - |x/lens|)
#         grad = torch.clamp(1.0 - torch.abs(mem_minus_thr / lens), min=0.0)
#         return grad_output * grad / lens, None


# spike_fn = SurrogateSpike.apply

# # ----------------------------
# # Simple LIF neuron module (per-layer stateful)
# # ----------------------------
# class LIFLayer(nn.Module):
#     def __init__(self, beta=0.9, threshold=1.0, lens=0.5, init_zero=True):
#         """
#         beta: membrane decay (0..1)
#         threshold: spike threshold
#         lens: surrogate gradient width
#         """
#         super().__init__()
#         self.beta = beta
#         self.threshold = threshold
#         self.lens = lens
#         # membrane & spike states will be created at runtime per-batch/shape
#         self.register_buffer('mem', None, persistent=False)
#         self.register_buffer('spk', None, persistent=False)
#         self.init_zero = init_zero

#     def reset_state(self, x):
#         # x is a tensor from previous conv: shape (B, C, H, W)
#         if x is None:
#             self.mem = None
#             self.spk = None
#             return
#         device = x.device
#         shape = x.shape
#         if (self.mem is None) or (self.mem.shape != shape):
#             if self.init_zero:
#                 self.mem = torch.zeros_like(x, device=device)
#             else:
#                 self.mem = torch.randn_like(x, device=device) * 0.01
#             self.spk = torch.zeros_like(x, device=device)

#     def forward(self, input_current):
#         """
#         Single time-step forward: input_current is the current (pre-activation)
#         Returns spike (0/1) and updates internal membrane state.
#         """
#         if self.mem is None or self.mem.shape != input_current.shape:
#             self.reset_state(input_current)

#         # integrate (leaky)
#         self.mem = self.mem * self.beta + input_current

#         # spike generation (surrogate)
#         mem_minus_thr = self.mem - self.threshold
#         spk = spike_fn(mem_minus_thr, self.lens)

#         # reset by subtracting threshold on spike (soft reset)
#         self.mem = self.mem - spk * self.threshold

#         self.spk = spk
#         return spk, self.mem

# # ----------------------------
# # Spiking DoubleConv block (Conv -> LIF -> Conv -> LIF)
# # We implement it so it can be stepped across T timesteps.
# # ----------------------------
# class SpikingDoubleConv(nn.Module):
#     def __init__(self, in_ch, out_ch, kernel_size=3, padding=1,
#                  beta=0.9, threshold=1.0, lens=0.5):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding)
#         self.lif1 = LIFLayer(beta=beta, threshold=threshold, lens=lens)
#         self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding=padding)
#         self.lif2 = LIFLayer(beta=beta, threshold=threshold, lens=lens)

#     def reset_state(self):
#         self.lif1.reset_state(torch.zeros(1))  # will be reinitialized on first forward per actual shape
#         self.lif2.reset_state(torch.zeros(1))

#     def forward_step(self, x):
#         # x: input spikes/current at this timestep (B, C, H, W)
#         # apply conv -> LIF -> conv -> LIF for a single timestep
#         x1 = self.conv1(x)
#         spk1, mem1 = self.lif1(x1)
#         x2 = self.conv2(spk1)  # pass spike as input current to next conv
#         spk2, mem2 = self.lif2(x2)
#         return spk2, mem2  # return spike and membrane (mem used for readout if desired)

# # ----------------------------
# # Spiking U-Net
# # ----------------------------
# class SpikingUNet(nn.Module):
#     def __init__(self, in_channels=1, out_channels=1, features=[64,128,256,512],
#                  beta=0.9, threshold=1.0, lens=0.5, T=20, readout='mem'):
#         """
#         readout: 'mem' (sum of membranes), 'spike' (sum of spikes) or 'last_mem'
#         T: number of timesteps to run
#         """
#         super().__init__()
#         self.T = T
#         self.readout = readout

#         # Encoder blocks
#         self.downs = nn.ModuleList()
#         prev = in_channels
#         for f in features:
#             self.downs.append(SpikingDoubleConv(prev, f, beta=beta, threshold=threshold, lens=lens))
#             prev = f

#         # Bottleneck
#         self.bottleneck = SpikingDoubleConv(prev, prev*2, beta=beta, threshold=threshold, lens=lens)

#         # Decoder blocks (upsampling via ConvTranspose2d)
#         rev = features[::-1]
#         self.ups_convtrans = nn.ModuleList()
#         self.ups_blocks = nn.ModuleList()
#         up_in = prev*2
#         for f in rev:
#             self.ups_convtrans.append(nn.ConvTranspose2d(up_in, f, kernel_size=2, stride=2))
#             # after concat channels become f + f -> pass that into SpikingDoubleConv which expects in_ch to match
#             self.ups_blocks.append(SpikingDoubleConv(f + f, f, beta=beta, threshold=threshold, lens=lens))
#             up_in = f

#         self.pool = nn.MaxPool2d(2, 2)
#         self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

#     def reset_states(self):
#         for m in self.modules():
#             if isinstance(m, SpikingDoubleConv):
#                 m.lif1.mem = None
#                 m.lif2.mem = None

#     def forward(self, x):
#         """
#         x: tensor (B, C, H, W) - static image input
#         Returns logits aggregated across time (B, out_ch, H, W)
#         """
#         device = x.device
#         B = x.shape[0]
#         # Convert input to Poisson spike trains (rate encoding)
#         # p = clamp(x, 0, 1) is assumed; if not normalized, scale appropriately before calling model
#         prob = x.clamp(min=0.0, max=1.0)  # expected input in [0,1] for images
#         # Initialize readout accumulators
#         if self.readout == 'mem':
#             out_acc = None  # accumulate membrane potentials
#         else:
#             out_acc = None  # accumulate spikes

#         # Reset states of all LIFs
#         self.reset_states()

#         # We'll store encoder skip activations per timestep; for memory efficiency we accumulate
#         # To handle skip connections we must keep per-encoder-layer spike tensors per timestep or use same-time concatenation.
#         # Simpler approach: at each timestep compute encoder activations and push them to stacks for concatenation.
#         # We'll store skip lists of length = number of encoder layers; each entry will be the spike at that timestep.
#         num_enc = len(self.downs)
#         # For aggregation we will sum spikes/mem across time and do UNet operations per timestep
#         # Run per-timestep forward
#         for t in range(self.T):
#             # generate Poisson spikes for input
#             rand = torch.rand_like(prob)
#             inp_spk = (rand <= prob).float()  # (B, C, H, W)

#             x_t = inp_spk
#             skip_spikes = []
#             # Encoder forward through spiking double convs
#             for enc in self.downs:
#                 spk, mem = enc.forward_step(x_t)
#                 skip_spikes.append(spk)
#                 # pooling operates on spikes (maxpool)
#                 x_t = self.pool(spk)

#             # Bottleneck
#             spk_bot, mem_bot = self.bottleneck.forward_step(x_t)
#             # Decoder: up + concat with corresponding skip (same timestep)
#             x_t = spk_bot
#             for upconv, decblk, skip in zip(self.ups_convtrans, self.ups_blocks, reversed(skip_spikes)):
#                 x_t = upconv(x_t)
#                 # shapes may mismatch by 1 due to pooling; center crop skip to match
#                 if x_t.shape[2:] != skip.shape[2:]:
#                     sh, sw = skip.shape[2:]
#                     th, tw = x_t.shape[2:]
#                     start_h = (sh - th) // 2
#                     start_w = (sw - tw) // 2
#                     skip = skip[:, :, start_h:start_h+th, start_w:start_w+tw]
#                 # concat spikes along channel dim
#                 x_cat = torch.cat([skip, x_t], dim=1)
#                 spk_dec, mem_dec = decblk.forward_step(x_cat)
#                 x_t = spk_dec

#             # final 1x1 conv: operate on spikes or membranes? We'll use spikes as input current
#             final_input = x_t
#             if self.readout == 'mem':
#                 # For membrane readout, we want to accumulate the membrane potential before spike reset.
#                 # Our SpikingDoubleConv returns last mem in its second return value only during step calls,
#                 # but here we don't have that from final conv. So we treat final conv as linear readout applied to spikes.
#                 logits_t = self.final_conv(final_input)  # (B, out_ch, H, W)
#                 if out_acc is None:
#                     out_acc = logits_t
#                 else:
#                     out_acc = out_acc + logits_t
#             else:
#                 logits_t = self.final_conv(final_input)
#                 if out_acc is None:
#                     out_acc = logits_t
#                 else:
#                     out_acc = out_acc + logits_t

#         # After T timesteps, aggregate
#         # For rate-based logits, average by T
#         out = out_acc / float(self.T)
#         # Optionally, if using membrane readout you'd return out directly as logits and apply BCE/CE externally.
#         return out


import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron

class SpikingDoubleConv(nn.Module):
    """
    Same as your DoubleConv but with LIF neurons replacing ReLU.
    Each conv -> GN -> LIF
    """
    def __init__(self, in_ch, out_ch, groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(groups, out_ch)
        self.lif1 = neuron.LIFNode()        # stateful neuron

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(groups, out_ch)
        self.lif2 = neuron.LIFNode()        # stateful neuron

    def forward(self, x):
        # x: (N, C, H, W)
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.lif1(x)    # produces spikes and updates membrane

        x = self.conv2(x)
        x = self.gn2(x)
        x = self.lif2(x)
        return x

    def reset(self):
        # reset state of the LIF nodes in this block
        self.lif1.reset()
        self.lif2.reset()


class SpikingUNetSmall(nn.Module):
    """
    Spiking version of UNetSmall.
    Forward runs the network for T timesteps and returns either:
      - spike-rate map averaged over T timesteps (default), or
      - the last-timestep spikes if return_last=True.
    """
    def __init__(self, in_ch=4, out_ch=1, timesteps=8):
        super().__init__()
        self.timesteps = timesteps
        ch = [24, 48, 96, 192]

        self.down1 = SpikingDoubleConv(in_ch, ch[0])
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = SpikingDoubleConv(ch[0], ch[1])
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = SpikingDoubleConv(ch[1], ch[2])
        self.pool3 = nn.MaxPool2d(2)
        self.bottom = SpikingDoubleConv(ch[2], ch[3])

        def up_module(in_c, skip_c, out_c):
            m = nn.ModuleDict({
                "up": nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(in_c, out_c, 1, bias=False)
                ),
                "dec": SpikingDoubleConv(out_c + skip_c, out_c)
            })
            return m

        self.up3 = up_module(ch[3], ch[2], ch[2])
        self.up2 = up_module(ch[2], ch[1], ch[1])
        self.up1 = up_module(ch[1], ch[0], ch[0])
        # head is a plain conv (maps spikes to logits). If you want spikes out also,
        # you can attach another LIFNode after this conv.
        self.head = nn.Conv2d(ch[0], out_ch, 1)

    def reset(self):
        """Reset membrane states for all LIF nodes in the network."""
        for m in self.modules():
            if isinstance(m, SpikingDoubleConv):
                m.reset()
        # If you later add LIF nodes outside SpikingDoubleConv, reset them here too.

    def forward_single_timestep(self, x):
        """Forward pass for a single timestep. x shape: (N, C, H, W)"""
        c1 = self.down1(x); p1 = self.pool1(c1)
        c2 = self.down2(p1); p2 = self.pool2(c2)
        c3 = self.down3(p2); p3 = self.pool3(c3)
        cb = self.bottom(p3)

        u3 = self.up3["up"](cb); d3 = self.up3["dec"](torch.cat([u3, c3], dim=1))
        u2 = self.up2["up"](d3); d2 = self.up2["dec"](torch.cat([u2, c2], dim=1))
        u1 = self.up1["up"](d2); d1 = self.up1["dec"](torch.cat([u1, c1], dim=1))
        out = self.head(d1)   # shape: (N, out_ch, H, W)
        return out

    def forward(self, x, timesteps=None, return_last=False, reset=True):
        """
        x: (N, C, H, W) static input (will be fed every timestep)
        timesteps: override self.timesteps if provided
        return_last: if True returns spikes/logits of last timestep, else returns time-averaged output
        reset: whether to call reset() before running (usually True)
        """
        if timesteps is None:
            timesteps = self.timesteps

        if reset:
            # clear neuron states before simulating a new sequence / batch
            self.reset()

        # accumulate outputs (e.g., sum of logits across timesteps)
        acc = None
        last = None
        for t in range(timesteps):
            out_t = self.forward_single_timestep(x)  # out_t are raw conv outputs; depending on your design, these are analog-valued
            # note: if you want spikes at the head, you can add a LIFNode here and call it.
            if acc is None:
                acc = out_t
            else:
                acc = acc + out_t
            last = out_t

        if return_last:
            return last
        # return rate (average across timesteps)
        return acc / float(timesteps)


# Example usage:
if __name__ == "__main__":
    # pip install spikingjelly before running
    model = SpikingUNetSmall(in_ch=4, out_ch=1, timesteps=6)
    img = torch.randn(2, 4, 128, 128)   # batch of 2
    out = model(img)   # shape (2,1,128,128) (time-averaged)
    print(out.shape)
