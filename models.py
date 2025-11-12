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

class DoubleConvm(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class UNetm(nn.Module):
    def __init__(self, in_ch=4, out_ch=1):
        super().__init__()
        self.down1 = DoubleConvm(in_ch, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConvm(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConvm(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.bottom = DoubleConvm(128, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec3 = DoubleConvm(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec2 = DoubleConvm(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.dec1 = DoubleConvm(64, 32)
        self.head = nn.Conv2d(32, out_ch, 1)

    def forward(self, x):
        c1 = self.down1(x)         # (B,32,H,W)
        p1 = self.pool1(c1)        # (B,32,H/2,W/2)
        c2 = self.down2(p1)        # (B,64, ...)
        p2 = self.pool2(c2)
        c3 = self.down3(p2)        # (B,128,...)
        p3 = self.pool3(c3)
        cb = self.bottom(p3)       # (B,256,...)
        u3 = self.up3(cb)          # -> match c3
        d3 = self.dec3(torch.cat([u3, c3], dim=1))
        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, c2], dim=1))
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, c1], dim=1))
        return self.head(d1)       # logits (B,1,H,W)