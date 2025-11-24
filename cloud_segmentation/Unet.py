import torch
import torch.nn as nn
import torch.nn.functional as F

# class DoubleConv(nn.Module):
#     """(conv => ReLU) * 2"""
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True)
#         )
#     def forward(self, x):
#         return self.net(x)

# class UNet(nn.Module):
#     def __init__(self, in_channels=1, out_channels=1, features=[64,128,256,512]):
#         super().__init__()
#         self.downs = nn.ModuleList()
#         self.ups = nn.ModuleList()
#         # Encoder (downs)
#         prev_ch = in_channels
#         for f in features:
#             self.downs.append(DoubleConv(prev_ch, f))
#             prev_ch = f
#         # Bottleneck
#         self.bottleneck = DoubleConv(prev_ch, prev_ch*2)
#         # Decoder (ups)
#         rev_features = features[::-1]
#         up_in_ch = prev_ch*2
#         for f in rev_features:
#             self.ups.append(nn.ConvTranspose2d(up_in_ch, f, kernel_size=2, stride=2))
#             self.ups.append(DoubleConv(up_in_ch, f))  # after concat channels = f + f = 2f
#             up_in_ch = f
#         # Final 1x1 conv
#         self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

#     def forward(self, x):
#         skip_connections = []
#         for down in self.downs:
#             x = down(x)
#             skip_connections.append(x)
#             x = self.pool(x)

#         x = self.bottleneck(x)
#         skip_connections = skip_connections[::-1]

#         up_idx = 0
#         for i in range(0, len(self.ups), 2):
#             upconv = self.ups[i]
#             dconv = self.ups[i+1]
#             x = upconv(x)                 # upsample
#             skip = skip_connections[up_idx]
#             up_idx += 1
#             # If shapes mismatch (due to odd sizes), center crop skip to x's size
#             if x.shape[2:] != skip.shape[2:]:
#                 # simple center crop
#                 sh, sw = skip.shape[2:]
#                 th, tw = x.shape[2:]
#                 start_h = (sh - th) // 2
#                 start_w = (sw - tw) // 2
#                 skip = skip[:, :, start_h:start_h+th, start_w:start_w+tw]
#             x = torch.cat([skip, x], dim=1)
#             x = dconv(x)

#         return self.final_conv(x)




import torch, torch.nn as nn, torch.nn.functional as F
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, groups=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class UNetSmall(nn.Module):
    def __init__(self, in_ch=4, out_ch=1):
        super().__init__()
        # slimmer channels help a lot on CPU
        ch = [24, 48, 96, 192]
        self.down1 = DoubleConv(in_ch, ch[0])
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(ch[0], ch[1])
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(ch[1], ch[2])
        self.pool3 = nn.MaxPool2d(2)
        self.bottom = DoubleConv(ch[2], ch[3])

        def up(in_c, skip_c, out_c):
            return nn.ModuleDict({
                "up": nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                    nn.Conv2d(in_c, out_c, 1, bias=False)),
                "dec": DoubleConv(out_c + skip_c, out_c)
            })
        self.up3 = up(ch[3], ch[2], ch[2])
        self.up2 = up(ch[2], ch[1], ch[1])
        self.up1 = up(ch[1], ch[0], ch[0])
        self.head = nn.Conv2d(ch[0], out_ch, 1)

    def forward(self, x):
        c1 = self.down1(x); p1 = self.pool1(c1)
        c2 = self.down2(p1); p2 = self.pool2(c2)
        c3 = self.down3(p2); p3 = self.pool3(c3)
        cb = self.bottom(p3)

        u3 = self.up3["up"](cb); d3 = self.up3["dec"](torch.cat([u3, c3], 1))
        u2 = self.up2["up"](d3); d2 = self.up2["dec"](torch.cat([u2, c2], 1))
        u1 = self.up1["up"](d2); d1 = self.up1["dec"](torch.cat([u1, c1], 1))
        return self.head(d1)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import snntorch as snn
# import snntorch.utils as snn_utils

# # --- Spiking Neuron Parameters ---
# # Beta is the decay rate for the membrane potential. Typically between 0 and 1.
# BETA = 0.9 
# # T is the number of simulation time steps. 
# TIME_STEPS = 10 

# class DoubleConvSpiking(nn.Module):
#     """
#     Spiking version of DoubleConv, replacing ReLU with snntorch.Leaky neurons.
#     The snn.Leaky neurons manage their own internal membrane state (mem) across time steps.
#     """
#     def __init__(self, in_ch, out_ch, groups=8, beta=BETA):
#         super().__init__()
        
#         # nn.Sequential is used, and the snn.Leaky neuron manages its own internal state.
#         self.net = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
#             nn.GroupNorm(groups, out_ch),
#             # Replace nn.ReLU with a Leaky Integrate-and-Fire (LIF) neuron.
#             # output=True ensures the forward pass returns the spike output (1s or 0s).
#             # init_hidden=True ensures the neuron manages its own internal state.
#             snn.Leaky(beta=beta, threshold=1.0, init_hidden=True, output=True),
            
#             nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
#             nn.GroupNorm(groups, out_ch),
#             snn.Leaky(beta=beta, threshold=1.0, init_hidden=True, output=True),
#         )

#     def forward(self, x): 
#         """
#         Runs one step of the double convolution block. 
#         The input x should be either a spike tensor or a rate-coded input.
#         """
#         return self.net(x)

# class SNNUNetSmall(nn.Module):
#     """
#     A Spiking Neural Network UNet implementation.
#     The forward pass processes the input over T time steps and accumulates the output spikes.
#     """
#     def __init__(self, in_ch=4, out_ch=1, beta=BETA, T=TIME_STEPS):
#         super().__init__()
#         self.T = T
#         self.beta = beta
        
#         # Channels, same as original
#         ch = [24, 48, 96, 192]
        
#         # Encoder (Downsampling path)
#         self.down1 = DoubleConvSpiking(in_ch, ch[0], beta=beta)
#         self.pool1 = nn.MaxPool2d(2)
#         self.down2 = DoubleConvSpiking(ch[0], ch[1], beta=beta)
#         self.pool2 = nn.MaxPool2d(2)
#         self.down3 = DoubleConvSpiking(ch[1], ch[2], beta=beta)
#         self.pool3 = nn.MaxPool2d(2)
#         self.bottom = DoubleConvSpiking(ch[2], ch[3], beta=beta)

#         # Decoder helper function to create upsampling blocks
#         def up(in_c, skip_c, out_c):
#             # The 'up' block is standard CNN layers: Upsample + 1x1 Conv
#             # The 'dec' block is the spiking DoubleConv
#             return nn.ModuleDict({
#                 "up": nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
#                                     nn.Conv2d(in_c, out_c, 1, bias=False)),
#                 "dec": DoubleConvSpiking(out_c + skip_c, out_c, beta=beta)
#             })
            
#         # Decoder (Upsampling path)
#         self.up3 = up(ch[3], ch[2], ch[2])
#         self.up2 = up(ch[2], ch[1], ch[1])
#         self.up1 = up(ch[1], ch[0], ch[0])
        
#         # Readout layer: Standard convolutional layer that accumulates the spikes over time
#         self.head = nn.Conv2d(ch[0], out_ch, 1)

#     def forward(self, x):
#         # 1. Reset the internal state (membrane potential) of all snn.Leaky neurons
#         snn_utils.reset(self) 

#         # 2. Initialize the accumulated output
#         final_output_sum = 0
        
#         # 3. Time Loop: Process the data over T time steps
#         for t in range(self.T):
#             # Input encoding: For a static image (x), we feed it repeatedly.
#             # The LIF neuron in the first layer integrates this input across time.
#             input_spk = x 

#             # --- Encoder ---
#             c1_spk = self.down1(input_spk) # c1_spk is the spike output of the block
#             p1_spk = self.pool1(c1_spk)
            
#             c2_spk = self.down2(p1_spk)
#             p2_spk = self.pool2(c2_spk)

#             c3_spk = self.down3(p2_spk)
#             p3_spk = self.pool3(c3_spk)

#             cb_spk = self.bottom(p3_spk)

#             # --- Decoder ---
#             # UP 3: Upsample bottom spikes, concatenate with skip connection spikes (c3_spk)
#             u3 = self.up3["up"](cb_spk)
#             d3_spk = self.up3["dec"](torch.cat([u3, c3_spk], 1))

#             # UP 2
#             u2 = self.up2["up"](d3_spk)
#             d2_spk = self.up2["dec"](torch.cat([u2, c2_spk], 1))

#             # UP 1
#             u1 = self.up1["up"](d2_spk)
#             d1_spk = self.up1["dec"](torch.cat([u1, c1_spk], 1))
            
#             # --- Readout ---
#             # The final layer sums the spike activity (or uses a standard conv layer)
#             current_output = self.head(d1_spk)
#             final_output_sum += current_output
        
#         # The final output is the accumulated spike activity over T timesteps, 
#         # divided by T for normalization (optional, depending on loss function)
#         return final_output_sum / self.T

# # --- Example Usage ---
# def test_snn_model():
#     # Model configuration
#     in_channels = 4  # e.g., image + 3 masks
#     out_channels = 1 # e.g., output mask
    
#     # Create an instance of the SNN UNet
#     model = SNNUNetSmall(in_ch=in_channels, out_ch=out_channels, T=TIME_STEPS, beta=BETA)
    
#     # Create dummy input data: [Batch_Size, Channels, Height, Width]
#     # UNet input is usually a power of 2, e.g., 64x64 or 128x128
#     dummy_input = torch.randn(2, in_channels, 64, 64) 
    
#     print(f"SNN-UNet created with {TIME_STEPS} timesteps and beta={BETA}.")
#     print(f"Input size: {dummy_input.shape}")

#     # Pass the dummy input through the SNN (processes over T timesteps internally)
#     output = model(dummy_input)
    
#     print(f"Output size (Accumulated Spikes): {output.shape}")

# if __name__ == '__main__':
#     test_snn_model()