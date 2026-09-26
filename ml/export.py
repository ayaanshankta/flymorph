"""Export a trained FlyNet to ONNX for the browser (onnxruntime-web) and the CLI (onnxruntime-node).

Graph I/O (must match packages/core/src/adult.ts):
  image [B,3,512,512] float32 (ImageNet-normalised) · drop [B,576,1,1] float32 → probs [B,4,512,512] softmax
Checks the exported file against PyTorch on the same random input before copying it into the web app.

usage: ml/.venv/bin/python ml/export.py ml/runs/<run>/best.pt
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from flynet import DROP_C, SIZE, FlyNet, rand_drop

WEB = Path(__file__).parents[1] / "apps/web/public/flynet.onnx"


class WithSoftmax(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, image, drop):
        return self.net(image, drop).softmax(1)


def main(ckpt_path: str) -> None:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net = FlyNet(pretrained=False).eval()
    net.load_state_dict(ck["model"])
    model = WithSoftmax(net).eval()
    out = Path(ckpt_path).parent / "flynet.onnx"
    x, d = torch.randn(2, 3, SIZE, SIZE), rand_drop(2)
    torch.onnx.export(model, (x, d), out, input_names=["image", "drop"], output_names=["probs"], opset_version=17,
                      dynamic_axes={"image": {0: "b"}, "drop": {0: "b"}, "probs": {0: "b"}}, dynamo=False)

    with torch.no_grad():
        ref = model(x, d).numpy()
    got = ort.InferenceSession(str(out)).run(None, {"image": x.numpy(), "drop": d.numpy()})[0]
    diff = float(np.abs(ref - got).max())
    assert got.shape == (2, 4, SIZE, SIZE) and diff < 1e-4, (got.shape, diff)
    assert DROP_C == 576
    shutil.copy(out, WEB)
    print(f"ok: max |onnx − torch| = {diff:.2e}, {out.stat().st_size / 1e6:.1f} MB → {WEB}  (labels: {ck['labels']})")


if __name__ == "__main__":
    main(sys.argv[1])
