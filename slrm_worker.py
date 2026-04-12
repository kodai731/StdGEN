"""S-LRM subprocess worker for mesh generation from multiview images.

Usage:
    python slrm_worker.py --mv-dir /path/to/multiview --output-dir /path/to/output
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import torch


def _log(msg: str):
    sys.stderr.write(f"[slrm_worker] {msg}\n")
    sys.stderr.flush()


def load_model(device: str):
    from omegaconf import OmegaConf
    from slrm.utils.train_util import instantiate_from_config

    config = OmegaConf.load("configs/mesh-slrm-infer.yaml")
    model = instantiate_from_config(config.model_config)
    state_dict = torch.load(config.infer_config.model_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    model.init_flexicubes_geometry(device, fovy=30.0, is_ortho=model.is_ortho)
    return model, config.infer_config


def load_multiview_images(mv_dir: str, device: str):
    from PIL import Image
    from torchvision.transforms import v2

    imgs = []
    for j in range(6):
        path = os.path.join(mv_dir, "level0", f"color_{j}.png")
        img = Image.open(path)

        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img = bg

        img = img.convert("RGB").resize((320, 320), Image.LANCZOS)
        arr = np.array(img).astype(np.float32) / 255.0
        imgs.append(torch.from_numpy(arr).permute(2, 0, 1))

    images = torch.stack(imgs, dim=0).unsqueeze(0).to(device)
    images = v2.functional.resize(images, (320, 320), interpolation=3, antialias=True).clamp(0, 1)
    return images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mv-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    t0 = time.monotonic()

    _log(f"loading S-LRM model")
    model, infer_config = load_model(device)
    _log(f"model loaded in {time.monotonic() - t0:.1f}s")

    images = load_multiview_images(args.mv_dir, device)
    input_cameras = torch.tensor(np.load("slrm/cameras.npy")).to(device)
    _log(f"images loaded: {images.shape}")

    from slrm.utils.mesh_util import save_obj, save_glb

    with torch.no_grad():
        torch.cuda.empty_cache()
        planes = model.forward_planes(images, input_cameras.float())
        _log(f"forward_planes done")

        for j, level_id in enumerate([0, 3, 4, 2]):
            mesh_out = model.extract_mesh(
                planes,
                use_texture_map=False,
                levels=torch.tensor([level_id]).to(device),
                **infer_config,
            )
            vertices, faces, vertex_colors = mesh_out
            vertices = vertices[:, [1, 2, 0]]

            save_obj(vertices, faces, vertex_colors, os.path.join(output_dir, f"mesh_{j}.obj"))
            save_glb(vertices, faces, vertex_colors, os.path.join(output_dir, f"mesh_{j}.glb"))
            _log(f"mesh_{j}: V={len(vertices)} F={len(faces)}")

    del model, planes
    gc.collect()
    torch.cuda.empty_cache()

    elapsed = time.monotonic() - t0
    _log(f"done in {elapsed:.1f}s")
    result = {"status": "ok", "output_dir": output_dir, "elapsed_ms": elapsed * 1000}
    print(json.dumps(result))


if __name__ == "__main__":
    main()
