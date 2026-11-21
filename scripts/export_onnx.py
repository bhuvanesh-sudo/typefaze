"""Export a TypeFaze checkpoint for the browser.

    python scripts/export_onnx.py --ckpt runs/ckpt_ema.pt --out web/public/models --fp16

Writes encoder.onnx, unet.onnx (one velocity step, dynamic batch), null_style.bin and meta.json.
Only one U-Net step is exported; the 12-step Euler loop runs in JavaScript.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx
import torch

from typefaze.flow import load_checkpoint
from typefaze.model import GlyphUNet, StyleEncoder


class EncoderWrap(torch.nn.Module):
    def __init__(self, enc: StyleEncoder):
        super().__init__()
        self.enc = enc

    def forward(self, refs, ref_ids, ref_mask):
        return self.enc(refs, ref_ids, ref_mask)


def to_fp16(path: Path, min_elems: int = 1024) -> None:
    """Store large weights as fp16 and cast them back to float32 inside the graph.

    The download halves, but all arithmetic stays float32: results match PyTorch closely and the model
    runs on the WASM backend too, which has patchy fp16 operator support.
    """
    from onnx import TensorProto, helper, numpy_helper

    model = onnx.load(str(path))
    g = model.graph
    casts = []
    for init in g.initializer:
        if init.data_type != TensorProto.FLOAT:
            continue
        arr = numpy_helper.to_array(init)
        if arr.size < min_elems:
            continue
        name = init.name
        init.CopyFrom(numpy_helper.from_array(arr.astype("float16"), name=name + "__fp16"))
        casts.append(helper.make_node("Cast", [name + "__fp16"], [name], to=TensorProto.FLOAT, name=name + "__cast"))
    nodes = casts + list(g.node)
    del g.node[:]
    g.node.extend(nodes)
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--wasm-steps", type=int, default=4, help="fewer steps on the slow CPU fallback")
    ap.add_argument("--cfg", type=float, default=1.0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    unet, enc = GlyphUNet().eval(), StyleEncoder().eval()
    step = load_checkpoint(args.ckpt, unet, enc)

    refs = torch.zeros(1, 5, 1, 64, 64)
    ref_ids = torch.zeros(1, 5, dtype=torch.long)
    ref_mask = torch.ones(1, 5, dtype=torch.bool)
    torch.onnx.export(EncoderWrap(enc), (refs, ref_ids, ref_mask), str(out / "encoder.onnx"),
                      input_names=["refs", "ref_ids", "ref_mask"], output_names=["style"],
                      opset_version=17, dynamo=False)

    B = 2
    torch.onnx.export(unet, (torch.zeros(B, 1, 64, 64), torch.zeros(B), torch.zeros(B, dtype=torch.long), torch.zeros(B, 256)),
                      str(out / "unet.onnx"),
                      input_names=["x_t", "t", "char_id", "style"], output_names=["v"],
                      dynamic_axes={n: {0: "batch"} for n in ["x_t", "t", "char_id", "style", "v"]},
                      opset_version=17, dynamo=False)

    if args.fp16:
        to_fp16(out / "encoder.onnx")
        to_fp16(out / "unet.onnx")

    unet.null_style.detach().float().numpy().tofile(out / "null_style.bin")
    (out / "meta.json").write_text(json.dumps({"steps": args.steps, "wasmSteps": args.wasm_steps, "cfg": args.cfg, "version": step}))
    total = sum(p.stat().st_size for p in out.iterdir())
    print(f"exported step {step} to {out} ({total / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
