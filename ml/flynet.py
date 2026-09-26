"""FlyNet: the small "student" that learns from SAM's reviewed drafts and runs in the browser.

U-Net shape: an encoder shrinks the image into features (what is where), a decoder grows them back to a
per-pixel label map, and skip connections hand fine detail from encoder to decoder at each scale.
  encoder  = MobileNetV3-small pretrained on ImageNet (already knows edges/textures → needs few labels)
  decoder  = 4 × (upsample 2×, concatenate the matching encoder features, 2 conv layers) + final 2× + 1×1 conv
  drop     = an INPUT tensor multiplied into the bottleneck. Passing random 0/1.25 masks = MC dropout;
             passing all ones = a normal deterministic prediction. Keeping randomness outside the graph means
             the exported ONNX file is plain maths that runs on every backend (WebGPU, WASM, Node).

self-check: ml/.venv/bin/python ml/flynet.py
"""
import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

N_CLASSES, DROP_C, SIZE = 4, 576, 512


def block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class FlyNet(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        f = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT if pretrained else None).features
        # stages end at strides 2, 4, 8, 16, 32 with 16, 16, 24, 48, 576 channels
        self.stages = nn.ModuleList([f[:1], f[1:2], f[2:4], f[4:9], f[9:]])
        widths, skips = [96, 48, 32, 16], [48, 24, 16, 16]
        cin = DROP_C
        self.up = nn.ModuleList()
        for w, s in zip(widths, skips):
            self.up.append(block(cin + s, w))
            cin = w
        self.head = nn.Conv2d(cin, N_CLASSES, 1)

    def forward(self, image: torch.Tensor, drop: torch.Tensor) -> torch.Tensor:
        feats, x = [], image
        for st in self.stages:
            x = st(x)
            feats.append(x)
        x = feats[-1] * drop  # [B,576,H/32,W/32] × [B,576,1,1]
        for up, skip in zip(self.up, reversed(feats[:-1])):
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = up(torch.cat([x, skip], 1))
        x = nn.functional.interpolate(x, size=image.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(x)  # logits; export.py appends the softmax


def rand_drop(batch: int, p: float = 0.2, device="cpu") -> torch.Tensor:
    return (torch.rand(batch, DROP_C, 1, 1, device=device) >= p).float() / (1 - p)


def no_drop(batch: int, device="cpu") -> torch.Tensor:
    return torch.ones(batch, DROP_C, 1, 1, device=device)


if __name__ == "__main__":
    net = FlyNet().eval()
    x = torch.randn(2, 3, SIZE, SIZE)
    with torch.no_grad():
        feats = x
        for st in net.stages:
            feats = st(feats)
        assert feats.shape[1] == DROP_C, feats.shape
        out = net(x, rand_drop(2)).softmax(1)
    assert out.shape == (2, N_CLASSES, SIZE, SIZE), out.shape
    assert torch.allclose(out.sum(1), torch.ones(2, SIZE, SIZE), atol=1e-5)
    n = sum(p.numel() for p in net.parameters())
    print(f"ok: {n / 1e6:.2f} M params ≈ {n * 4 / 1e6:.1f} MB fp32, output {tuple(out.shape)}")
