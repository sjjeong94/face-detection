"""
[reference]
https://github.com/qubvel/segmentation_models.pytorch
"""

import torch
import torch.nn as nn
import torchvision


class MobileNetV2(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.features = torchvision.models.mobilenet_v2(
            pretrained=pretrained).features

    def get_stages(self):
        return [
            nn.Identity(),
            self.features[:2],
            self.features[2:4],
            self.features[4:7],
            self.features[7:14],
            self.features[14:],
        ]

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(len(stages)):
            x = stages[i](x)
            features.append(x)

        return features


class ConvBN(nn.Module):
    def __init__(self, in_ch, out_ch, k_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k_size,
                              stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return self.bn(self.conv(x))


class FPN(nn.Module):
    def __init__(self, in_ch2, in_ch3, in_ch4, in_ch5, out_ch=256):
        super().__init__()
        self.up = nn.UpsamplingNearest2d(scale_factor=2)
        self.act = nn.ReLU(inplace=True)
        self.inner2 = ConvBN(in_ch2, out_ch, 1, 1)
        self.inner3 = ConvBN(in_ch3, out_ch, 1, 1)
        self.inner4 = ConvBN(in_ch4, out_ch, 1, 1)
        self.inner5 = ConvBN(in_ch5, out_ch, 1, 1)
        self.layer2 = ConvBN(out_ch, out_ch, 3, 1, 1)
        self.layer3 = ConvBN(out_ch, out_ch, 3, 1, 1)
        self.layer4 = ConvBN(out_ch, out_ch, 3, 1, 1)

    def forward(self, i2, i3, i4, i5):
        i2 = self.act(self.inner2(i2))
        i3 = self.act(self.inner3(i3))
        i4 = self.act(self.inner4(i4))
        o5 = self.act(self.inner5(i5))
        o4 = self.act(self.layer4(i4 + self.up(o5)))
        o3 = self.act(self.layer3(i3 + self.up(o4)))
        o2 = self.act(self.layer2(i2 + self.up(o3)))
        return o2, o3, o4, o5


class Head(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.act = nn.ReLU(inplace=True)
        self.conv1 = ConvBN(in_ch, mid_ch, 3, 1, 1)
        self.conv2 = ConvBN(mid_ch, mid_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(mid_ch, out_ch, 1, 1)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)
        return x


class Detector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = MobileNetV2()
        self.neck = FPN(24, 32, 96, 1280, 256)

        self.head_r = Head(256, 256, 4)
        self.head_k = Head(256, 256, 1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(*x[-4:])

        out_o = torch.exp(self.head_r(x[-4]))
        out_k = self.head_k(x[-4])

        out = torch.concat([out_o, out_k], axis=1)
        return out


if __name__ == '__main__':
    x = torch.randn(8, 3, 320, 320)
    backbone = MobileNetV2()
    neck = FPN(24, 32, 96, 1280, 256)
    head = Head(256, 256, 91)
    outs = backbone(x)
    print(backbone)
    for out in outs:
        print(out.shape)
    print()

    outs = neck(*outs[-4:])
    for out in outs:
        print(out.shape)
    print()

    out = head(outs[-4])
    print(out.shape)
    print()

    detector = Detector()
    out = detector(x)
    print(out.shape)
    print()
    # print(detector)

    torch.save(detector.state_dict(), 'net.pth')
