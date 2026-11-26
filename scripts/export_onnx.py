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
from typefaze.model_v2 import GlyphUNetV2, RefTokenEncoder


class EncoderV2Wrap(torch.nn.Module):
    """v2 encoder for the browser: always 5 valid references, so the token mask is dropped."""

    def __init__(self, enc: RefTokenEncoder):
        super().__init__()
        self.enc = enc

    def forward(self, refs, ref_ids, ref_mask):
        style, tokens, _ = self.enc(refs, ref_ids, ref_mask)
        return style, tokens


class UNetV2Wrap(torch.nn.Module):
    """v2 U-Net step for the browser: tokens come in once (batch 1) and are shared by all letters."""

    def __init__(self, unet: GlyphUNetV2):
        super().__init__()
        self.unet = unet

    def forward(self, x_t, t, char_id, style, tokens):
        return self.unet(x_t, t, char_id, style, tokens, None)


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


def simplify(path: Path) -> None:
    from onnxsim import simplify as onnxsim_simplify

    model, ok = onnxsim_simplify(onnx.load(str(path)))
    if not ok:
        raise RuntimeError(f"onnx simplifier could not verify {path.name}")
    onnx.save(model, str(path))


@torch.no_grad()  # tracing with autograd on kept the whole batch-52 graph alive (~6 GB peak vs ~1.6 GB)
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--batch", type=int, default=52, help="fixed U-Net batch size (all 52 letters at once)")
    ap.add_argument("--wasm-steps", type=int, default=4, help="fewer steps on the slow CPU fallback")
    ap.add_argument("--cfg", type=float, default=1.0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    arch = torch.load(args.ckpt, map_location="cpu", weights_only=False).get("config", {}).get("arch", "v1")
    if arch == "v2":
        unet, enc = GlyphUNetV2().eval(), RefTokenEncoder().eval()
    else:
        unet, enc = GlyphUNet().eval(), StyleEncoder().eval()
    step = load_checkpoint(args.ckpt, unet, enc)

    refs = torch.zeros(1, 5, 1, 64, 64)
    ref_ids = torch.zeros(1, 5, dtype=torch.long)
    ref_mask = torch.ones(1, 5, dtype=torch.bool)
    B = args.batch
    unet_args = (torch.zeros(B, 1, 64, 64), torch.zeros(B), torch.zeros(B, dtype=torch.long), torch.zeros(B, 256))
    if arch == "v2":
        torch.onnx.export(EncoderV2Wrap(enc), (refs, ref_ids, ref_mask), str(out / "encoder.onnx"),
                          input_names=["refs", "ref_ids", "ref_mask"], output_names=["style", "tokens"],
                          opset_version=17, dynamo=False)
        n_tok = 5 * 64
        torch.onnx.export(UNetV2Wrap(unet), unet_args + (torch.zeros(1, n_tok, unet.null_token.numel()),),
                          str(out / "unet.onnx"), input_names=["x_t", "t", "char_id", "style", "tokens"],
                          output_names=["v"], opset_version=17, dynamo=False)
        unet.null_token.detach().float().expand(n_tok, -1).contiguous().numpy().tofile(out / "null_tokens.bin")
    else:
        torch.onnx.export(EncoderWrap(enc), (refs, ref_ids, ref_mask), str(out / "encoder.onnx"),
                          input_names=["refs", "ref_ids", "ref_mask"], output_names=["style"],
                          opset_version=17, dynamo=False)
        # Fixed batch: every shape is known at export time, so the simplifier can fold away the Shape/Gather/
        # Reshape chains that would otherwise run on the CPU (and stall the GPU) on every step in the browser.
        torch.onnx.export(unet, unet_args, str(out / "unet.onnx"),
                          input_names=["x_t", "t", "char_id", "style"], output_names=["v"],
                          opset_version=17, dynamo=False)
    simplify(out / "encoder.onnx")
    simplify(out / "unet.onnx")

    if args.fp16:
        to_fp16(out / "encoder.onnx")
        to_fp16(out / "unet.onnx")

    unet.null_style.detach().float().numpy().tofile(out / "null_style.bin")
    (out / "meta.json").write_text(json.dumps({"arch": arch, "steps": args.steps, "wasmSteps": args.wasm_steps, "cfg": args.cfg, "batch": args.batch, "version": step}))
    total = sum(p.stat().st_size for p in out.iterdir())
    print(f"exported step {step} to {out} ({total / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
