from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.expanduser("~/.cache/torch/inductor"))

READY_MARKER = "__READY__"
SHUTDOWN_COMMAND = "__SHUTDOWN__"
RESULT_MARKER_START = "__RESULT__"
RESULT_MARKER_END = "__END_RESULT__"


def load_multiview_pipeline(device):
    import torch
    from multiview.pipeline_multiclass import StableUnCLIPImg2ImgPipeline

    pipeline = StableUnCLIPImg2ImgPipeline.from_pretrained(
        "./ckpt/StdGEN-multiview-1024",
        torch_dtype=torch.float32,
    )
    pipeline.to(device)
    pipeline.enable_vae_slicing()

    t0 = time.monotonic()
    pipeline.unet = torch.compile(pipeline.unet, mode="default")
    sys.stderr.write(f"[torch.compile] multiview UNet compiled ({time.monotonic() - t0:.1f}s)\n")

    return pipeline


def load_slrm_model(device):
    import torch
    from omegaconf import OmegaConf
    from slrm.utils.train_util import instantiate_from_config

    config = OmegaConf.load("configs/mesh-slrm-infer.yaml")
    model = instantiate_from_config(config.model_config)
    state_dict = torch.load(config.infer_config.model_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    model.init_flexicubes_geometry(device, fovy=30.0, is_ortho=model.is_ortho)

    t0 = time.monotonic()
    model = torch.compile(model, mode="default")
    sys.stderr.write(f"[torch.compile] S-LRM compiled ({time.monotonic() - t0:.1f}s)\n")

    return model, config.infer_config


def load_multiview_embeddings():
    import torch

    prompt_dir = "./multiview/fixed_prompt_embeds_6view"
    normal_embeds = torch.load(f"{prompt_dir}/normal_embeds.pt")
    color_embeds = torch.load(f"{prompt_dir}/clr_embeds.pt")
    return normal_embeds, color_embeds


def load_all(device: str) -> dict:
    sys.stderr.write("Loading StdGEN models...\n")
    sys.stderr.flush()

    t0 = time.monotonic()
    multiview_pipeline = load_multiview_pipeline(device)
    normal_embeds, color_embeds = load_multiview_embeddings()
    slrm_model, slrm_infer_config = load_slrm_model(device)

    elapsed = time.monotonic() - t0
    sys.stderr.write(f"All StdGEN models loaded in {elapsed:.1f}s\n")
    sys.stderr.flush()

    return {
        "device": device,
        "multiview_pipeline": multiview_pipeline,
        "normal_embeds": normal_embeds,
        "color_embeds": color_embeds,
        "slrm_model": slrm_model,
        "slrm_infer_config": slrm_infer_config,
    }


def run_multiview(ctx, input_image_path: str, work_dir: str, seed: int):
    import torch
    import numpy as np
    from PIL import Image
    from torchvision import transforms
    from einops import rearrange

    device = ctx["device"]
    pipeline = ctx["multiview_pipeline"]
    normal_embeds = ctx["normal_embeds"]
    color_embeds = ctx["color_embeds"]

    img = Image.open(input_image_path).convert("RGB")
    new_img = Image.new("RGB", (1024, 1024), (255, 255, 255))
    w, h = img.size
    new_w = int(w / h * 1024)
    img = img.resize((new_w, 1024))
    offset = (1024 - new_w) // 2
    new_img.paste(img, (offset, 0))

    tform = transforms.Compose([
        transforms.Resize(1024),
        transforms.CenterCrop((1024, 576)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2.0 - 1),
    ])
    cond = tform(new_img)
    imgs_in = torch.stack([cond] * 6, dim=0).unsqueeze(0)
    imgs_in = torch.cat([imgs_in] * 2, dim=0).to(device)
    imgs_in_flat = rearrange(imgs_in, "B Nv C H W -> (B Nv) C H W")

    prompt_embeddings = torch.cat([normal_embeds, color_embeds], dim=0).to(device)

    generator = torch.Generator(device=device).manual_seed(seed)

    with torch.no_grad():
        unet_out = pipeline(
            imgs_in_flat, None,
            prompt_embeds=prompt_embeddings,
            generator=generator,
            guidance_scale=3.0,
            output_type="pt",
            num_images_per_prompt=1,
            height=1024, width=576,
            num_inference_steps=40,
            eta=1.0,
            num_levels=3,
        )

    views = ["front", "front_right", "right", "back", "left", "front_left"]
    mv_dir = os.path.join(work_dir, "multiview")

    for level in range(3):
        out = unet_out[level].images
        bsz = out.shape[0] // 2
        normals_pred = out[:bsz]
        images_pred = out[bsz:]

        level_dir = os.path.join(mv_dir, f"level{level}")
        os.makedirs(level_dir, exist_ok=True)

        for j in range(6):
            normal_np = normals_pred[j].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).cpu().to(torch.uint8).numpy()
            color_np = images_pred[j].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).cpu().to(torch.uint8).numpy()

            _save_padded(normal_np, os.path.join(level_dir, f"normal_{j}.png"))
            _save_padded(color_np, os.path.join(level_dir, f"color_{j}.png"))

    gc.collect()
    torch.cuda.empty_cache()
    return mv_dir


def _save_padded(arr, path):
    from PIL import Image

    im = Image.fromarray(arr)
    if im.size[0] != im.size[1]:
        size = max(im.size)
        new_im = Image.new("RGB", (size, size), (255, 255, 255))
        new_im.paste(im, ((size - im.size[0]) // 2, (size - im.size[1]) // 2))
        im = new_im
    im = im.resize((1024, 1024), Image.LANCZOS)
    im.save(path)


def run_slrm(ctx, mv_dir: str, work_dir: str):
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    from torchvision.transforms import v2
    from slrm.utils.mesh_util import save_obj, save_glb

    device = ctx["device"]
    model = ctx["slrm_model"]
    infer_config = ctx["slrm_infer_config"]

    imgs = []
    for j in range(6):
        path = os.path.join(mv_dir, "level0", f"color_{j}.png")
        img = plt.imread(path)
        img = Image.fromarray(np.uint8(img * 255.0))
        img = img.resize((320, 320), Image.LANCZOS)
        img = np.array(img).astype(np.float32) / 255.0
        imgs.append(torch.from_numpy(img).permute(2, 0, 1))

    images = torch.stack(imgs, dim=0).unsqueeze(0).to(device)
    images = v2.functional.resize(images, (320, 320), interpolation=3, antialias=True).clamp(0, 1)
    input_cameras = torch.tensor(np.load("slrm/cameras.npy")).to(device)

    slrm_dir = os.path.join(work_dir, "slrm")
    os.makedirs(slrm_dir, exist_ok=True)

    with torch.no_grad():
        torch.cuda.empty_cache()
        planes = model.forward_planes(images, input_cameras.float())

        for j, level_id in enumerate([0, 3, 4, 2]):
            mesh_out = model.extract_mesh(
                planes,
                use_texture_map=False,
                levels=torch.tensor([level_id]).to(device),
                **infer_config,
            )
            vertices, faces, vertex_colors = mesh_out
            vertices = vertices[:, [1, 2, 0]]

            obj_path = os.path.join(slrm_dir, f"mesh_{j}.obj")
            glb_path = os.path.join(slrm_dir, f"mesh_{j}.glb")
            save_obj(vertices, faces, vertex_colors, obj_path)
            save_glb(vertices, faces, vertex_colors, glb_path)

    gc.collect()
    torch.cuda.empty_cache()
    return slrm_dir


def run_refine(mv_dir: str, slrm_dir: str, work_dir: str):
    import torch
    import cv2
    import numpy as np
    import trimesh
    from PIL import Image
    from sklearn.neighbors import KDTree

    from refine.mesh_refine import geo_refine
    from refine.func import make_star_cameras_orthographic
    from refine.render import NormalsRenderer, calc_vertex_normals
    from infer_refine import (
        calc_horizontal_offset,
        calc_horizontal_offset2,
        filter_fixed_mesh_by_proximity,
        get_distract_mask,
        save_py3dmesh_with_trimesh_fast,
        _unload_sam,
    )

    fixed_v, fixed_f = None, None
    last_colors, last_normals = None, None
    last_front_color = None

    mv, proj = make_star_cameras_orthographic(8, 1, r=1.2)
    mv = mv[[4, 3, 2, 0, 6, 5]]
    renderer = NormalsRenderer(mv, proj, (1024, 1024))

    refine_dir = os.path.join(work_dir, "refined")
    os.makedirs(refine_dir, exist_ok=True)

    name_to_level = [(3, 2), (1, 1), (2, 0)]

    for name_idx, level in name_to_level:
        gc.collect()
        torch.cuda.empty_cache()

        mesh = trimesh.load(os.path.join(slrm_dir, f"mesh_{name_idx}.obj"))
        new_mesh = mesh.split(only_watertight=False)
        new_mesh = [j for j in new_mesh if len(j.vertices) >= 300]
        mesh = trimesh.Scene(new_mesh).to_geometry()
        mesh_v, mesh_f = mesh.vertices, mesh.faces

        if last_colors is None:
            images = renderer.render(
                torch.tensor(mesh_v, device="cuda").float(),
                torch.ones_like(torch.from_numpy(mesh_v), device="cuda").float(),
                torch.tensor(mesh_f, device="cuda"),
            )
            mask = (images[..., 3] < 0.9).cpu().numpy()

        colors, normals = [], []
        for i in range(6):
            color = cv2.imread(os.path.join(mv_dir, f"level{level}", f"color_{i}.png"))[..., ::-1]
            normal = cv2.imread(os.path.join(mv_dir, f"level{level}", f"normal_{i}.png"))[..., ::-1]

            if last_colors is not None:
                offset = calc_horizontal_offset(np.array(last_colors[i]), color)
            else:
                offset = calc_horizontal_offset2(mask[i], color)

            if offset != 0:
                color = np.roll(color, offset, axis=1)
                normal = np.roll(normal, offset, axis=1)

            colors.append(Image.fromarray(color))
            normals.append(Image.fromarray(normal))

        distract_mask, distract_bbox = None, None
        if last_front_color is not None and level == 0:
            _, distract_bbox, _, distract_mask = get_distract_mask(
                last_front_color,
                np.array(colors[0]).astype(np.float32) / 255.0,
                outside_ratio=0.20,
            )
            _unload_sam()

        last_front_color = np.array(colors[0]).astype(np.float32) / 255.0

        if last_colors is None:
            from copy import deepcopy
            last_colors, last_normals = deepcopy(colors), deepcopy(normals)

        if fixed_v is not None and level == 1:
            kdtree_anchor = KDTree(fixed_v.numpy())
            kdtree_mesh_v = KDTree(mesh_v)
            _, idx_anchor = kdtree_anchor.query(mesh_v, k=1)
            _, idx_mesh_v = kdtree_mesh_v.query(mesh_v, k=25)
            idx_anchor = idx_anchor.squeeze()
            neighbors = torch.tensor(mesh_v).cuda()[idx_mesh_v]
            neighbor_dists = torch.norm(neighbors - torch.tensor(mesh_v).cuda()[:, None], dim=-1)
            neighbor_dists[neighbor_dists > 0.06] = 114514.0
            neighbor_weights = torch.exp(-neighbor_dists * 1.0)
            neighbor_weights = neighbor_weights / neighbor_weights.sum(dim=1, keepdim=True)
            fv_gpu = fixed_v.cuda()
            ff_gpu = fixed_f.cuda()
            anchors = fv_gpu[idx_anchor]
            anchor_normals = calc_vertex_normals(fv_gpu, ff_gpu)[idx_anchor]
            dis_anchor = torch.clamp(((anchors - torch.tensor(mesh_v).cuda()) * anchor_normals).sum(-1), min=0) + 0.01
            vec_anchor = dis_anchor[:, None] * anchor_normals
            vec_anchor = vec_anchor[idx_mesh_v]
            weighted_vec_anchor = (vec_anchor * neighbor_weights[:, :, None]).sum(1)
            mesh_v += weighted_vec_anchor.cpu().numpy()
            del fv_gpu, ff_gpu, anchors, anchor_normals, neighbors, neighbor_dists, neighbor_weights
            torch.cuda.empty_cache()

        gc.collect()
        torch.cuda.empty_cache()

        mesh_v_t = torch.tensor(mesh_v, device="cuda", dtype=torch.float32)
        mesh_f_t = torch.tensor(mesh_f, device="cuda")

        level_fixed_v, level_fixed_f = fixed_v, fixed_f
        if level == 0 and fixed_v is not None:
            level_fixed_v, level_fixed_f = filter_fixed_mesh_by_proximity(
                fixed_v, fixed_f, mesh_v_t.cpu(), margin=0.15,
            )

        new_mesh_result, simp_v, simp_f = geo_refine(
            mesh_v_t, mesh_f_t, colors, normals,
            fixed_v=level_fixed_v, fixed_f=level_fixed_f,
            distract_mask=distract_mask, distract_bbox=distract_bbox,
        )

        save_py3dmesh_with_trimesh_fast(
            new_mesh_result,
            os.path.join(refine_dir, f"out_{level}.glb"),
            apply_sRGB_to_LinearRGB=True,
        )

        if fixed_v is None:
            fixed_v, fixed_f = simp_v.cpu(), simp_f.cpu()
        else:
            fixed_f = torch.cat([fixed_f, simp_f.cpu() + fixed_v.shape[0]], dim=0)
            fixed_v = torch.cat([fixed_v, simp_v.cpu()], dim=0)

        del new_mesh_result, simp_v, simp_f, colors, normals, level_fixed_v, level_fixed_f
        gc.collect()
        torch.cuda.empty_cache()

    del renderer, last_colors, last_normals, fixed_v, fixed_f
    gc.collect()
    torch.cuda.empty_cache()

    return refine_dir


def combine_refined_glbs(refine_dir: str, output_path: str):
    import numpy as np
    from pygltflib import (
        GLTF2, Mesh, Primitive, Accessor, BufferView, Buffer,
        Material, PbrMetallicRoughness, Attributes,
    )

    all_positions, all_colors, all_indices = [], [], []
    vertex_offset = 0
    total_vertices, total_faces = 0, 0

    for level in [0, 1, 2]:
        glb = GLTF2().load(os.path.join(refine_dir, f"out_{level}.glb"))
        prim = glb.meshes[0].primitives[0]
        binary = glb.binary_blob()

        def read_accessor(acc_idx):
            acc = glb.accessors[acc_idx]
            bv = glb.bufferViews[acc.bufferView]
            start = bv.byteOffset + (acc.byteOffset or 0)
            return binary[start:start + bv.byteLength]

        indices = np.frombuffer(read_accessor(prim.indices), dtype=np.uint32)
        positions = np.frombuffer(read_accessor(prim.attributes.POSITION), dtype=np.float32).reshape(-1, 3)
        col_data = read_accessor(prim.attributes.COLOR_0)
        colors = np.frombuffer(col_data, dtype=np.uint8).reshape(-1, 4)

        all_indices.append(indices + vertex_offset)
        all_positions.append(positions)
        all_colors.append(colors)
        vertex_offset += len(positions)
        total_vertices += len(positions)
        total_faces += len(indices) // 3

    positions = np.concatenate(all_positions).astype(np.float32)
    colors = np.concatenate(all_colors).astype(np.uint8)
    indices = np.concatenate(all_indices).astype(np.uint32)

    idx_bytes = indices.tobytes()
    pos_bytes = positions.tobytes()
    col_bytes = colors.tobytes()
    blob = idx_bytes + pos_bytes + col_bytes

    out = GLTF2()
    out.asset.generator = "StdGEN-combined"
    out.scene = 0
    out.scenes = [{"nodes": [0]}]
    out.nodes = [{"mesh": 0}]
    out.buffers = [Buffer(byteLength=len(blob))]
    out.bufferViews = [
        BufferView(buffer=0, byteOffset=0, byteLength=len(idx_bytes), target=34963),
        BufferView(buffer=0, byteOffset=len(idx_bytes), byteLength=len(pos_bytes), target=34962),
        BufferView(buffer=0, byteOffset=len(idx_bytes) + len(pos_bytes), byteLength=len(col_bytes), target=34962),
    ]
    out.accessors = [
        Accessor(bufferView=0, componentType=5125, count=len(indices), type="SCALAR"),
        Accessor(
            bufferView=1, componentType=5126, count=len(positions), type="VEC3",
            max=positions.max(axis=0).tolist(), min=positions.min(axis=0).tolist(),
        ),
        Accessor(bufferView=2, componentType=5121, count=len(colors), type="VEC4", normalized=True),
    ]
    out.materials = [Material(
        pbrMetallicRoughness=PbrMetallicRoughness(
            baseColorFactor=[1.0, 1.0, 1.0, 1.0],
            metallicFactor=0.0,
            roughnessFactor=1.0,
        ),
        doubleSided=True,
    )]
    out.meshes = [Mesh(primitives=[Primitive(
        attributes=Attributes(POSITION=1, COLOR_0=2),
        indices=0,
        material=0,
    )])]
    out.set_binary_blob(blob)
    out.save(output_path)

    return total_vertices, total_faces


def process_request(ctx, request: dict) -> dict:
    import torch

    image_path = request["image"]
    output_path = request["output"]
    seed = request.get("seed", 42)
    work_dir = request.get("work_dir", "/tmp/stdgen_work")
    os.makedirs(work_dir, exist_ok=True)

    t0 = time.monotonic()

    t_mv = time.monotonic()
    mv_dir = run_multiview(ctx, image_path, work_dir, seed)
    multiview_ms = (time.monotonic() - t_mv) * 1000.0
    sys.stderr.write(f"Multiview: {multiview_ms:.0f}ms\n")
    sys.stderr.flush()

    del ctx["multiview_pipeline"]
    gc.collect()
    torch.cuda.empty_cache()

    t_slrm = time.monotonic()
    slrm_dir = run_slrm(ctx, mv_dir, work_dir)
    slrm_ms = (time.monotonic() - t_slrm) * 1000.0
    sys.stderr.write(f"S-LRM: {slrm_ms:.0f}ms\n")
    sys.stderr.flush()

    del ctx["slrm_model"]
    gc.collect()
    torch.cuda.empty_cache()

    t_refine = time.monotonic()
    refine_dir = run_refine(mv_dir, slrm_dir, work_dir)
    refine_ms = (time.monotonic() - t_refine) * 1000.0
    sys.stderr.write(f"Refine: {refine_ms:.0f}ms\n")
    sys.stderr.flush()

    vertex_count, face_count = combine_refined_glbs(refine_dir, output_path)

    sys.stderr.write("Reloading multiview + S-LRM...\n")
    sys.stderr.flush()
    ctx["multiview_pipeline"] = load_multiview_pipeline(ctx["device"])
    ctx["slrm_model"], ctx["slrm_infer_config"] = load_slrm_model(ctx["device"])

    total_ms = (time.monotonic() - t0) * 1000.0

    return {
        "status": "ok",
        "vertex_count": vertex_count,
        "face_count": face_count,
        "total_ms": total_ms,
        "multiview_ms": multiview_ms,
        "slrm_ms": slrm_ms,
        "refine_ms": refine_ms,
    }


def daemon_main():
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    t0 = time.monotonic()
    ctx = load_all(device)
    sys.stderr.write(f"StdGEN ready ({time.monotonic() - t0:.1f}s)\n")
    sys.stderr.flush()

    print(READY_MARKER, flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line == SHUTDOWN_COMMAND:
            break

        try:
            request = json.loads(line)
            result = process_request(ctx, request)
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            result = {"status": "error", "message": str(e)}

        print(f"{RESULT_MARKER_START}{json.dumps(result)}{RESULT_MARKER_END}", flush=True)

    sys.stderr.write("StdGEN worker shutting down\n")
    sys.stderr.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--daemon", action="store_true")
    args = parser.parse_args()

    if args.daemon:
        daemon_main()
