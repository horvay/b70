#!/usr/bin/env python3
"""Reference for the C++ vision tower: HF's image processor and Qwen3_5VisionModel on the CPU, on the pack's bf16 weights.

Run in the SGLang image, no GPU: b70 run python3 tests/q35v_ref.py <model dir> <image> <out dir> [max_pixels]
Writes raw little-endian files the check reads: meta.json, pixels.f32 [N, 1536 -> two equal frames, one kept],
stream0.f32 (row 0 after the patch embedding and each block, [28, 1152]), out.f32 [N / 4, 5120].
"""
import json
import pathlib
import sys

import torch
from PIL import Image
from safetensors import safe_open
from transformers import AutoImageProcessor
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel


def main() -> None:
    model_dir, image, out = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
    max_pixels = int(sys.argv[4]) if len(sys.argv) > 4 else 4194304
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)

    proc = AutoImageProcessor.from_pretrained(model_dir)
    img = Image.open(image).convert("RGB")
    enc = proc(images=[img], size={"longest_edge": max_pixels, "shortest_edge": 65536}, return_tensors="pt")
    pixels, grid = enc["pixel_values"], enc["image_grid_thw"]
    print("grid", grid.tolist(), "pixels", pixels.shape)

    cfg = json.load(open(f"{model_dir}/config.json"))["vision_config"]
    vc = Qwen3_5VisionConfig(**{k: v for k, v in cfg.items() if k != "model_type"})
    vc._attn_implementation = "eager"
    model = Qwen3_5VisionModel(vc).to(torch.bfloat16).eval()

    index = json.load(open(f"{model_dir}/model.safetensors.index.json"))["weight_map"]
    state = {}
    for key, shard in index.items():
        if key.startswith("model.visual."):
            with safe_open(f"{model_dir}/{shard}", "pt") as f:
                state[key[len("model.visual."):]] = f.get_tensor(key)
    print(model.load_state_dict(state, strict=False))

    streams = []
    def hook(_m, _i, o):
        x = o[0] if isinstance(o, tuple) else o
        streams.append(x[0].float().clone())
    model.patch_embed.register_forward_hook(lambda m, i, o: streams.append(o[0].float().clone()))
    for blk in model.blocks:
        blk.register_forward_hook(hook)

    with torch.no_grad():
        result = model(pixels.to(torch.bfloat16), grid_thw=grid)
    final = result.pooler_output.float()
    print("out", final.shape, "rms", final.pow(2).mean().sqrt().item())

    # how much of any difference is the checkpoint's own rounding: the same tower in fp32, and its merger alone
    with torch.no_grad():
        m32 = model.to(torch.float32)
        r32 = m32(pixels.to(torch.float32), grid_thw=grid).pooler_output
        print("bf16 vs fp32 output rel err", ((final - r32).pow(2).sum() / r32.pow(2).sum()).sqrt().item())
        r32.numpy().tofile(out / "out32.f32")

    # one frame of each patch: the processor doubles it along the temporal axis
    n = pixels.shape[0]
    frames = pixels.view(n, 3, 2, 16, 16)
    assert torch.equal(frames[:, :, 0], frames[:, :, 1])
    frames[:, :, 0].float().numpy().tofile(out / "pixels.f32")
    torch.stack(streams).numpy().tofile(out / "stream0.f32")
    final.numpy().tofile(out / "out.f32")
    json.dump({"grid": grid[0].tolist(), "N": n, "image": image, "max_pixels": max_pixels}, open(out / "meta.json", "w"))


if __name__ == "__main__":
    main()
