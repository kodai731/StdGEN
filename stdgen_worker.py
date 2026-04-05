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
os.environ.setdefault("PYTORCH_ALLOC_CONF", "max_split_size_mb:128")

READY_MARKER = "__READY__"
SHUTDOWN_COMMAND = "__SHUTDOWN__"
RESULT_MARKER_START = "__RESULT__"
RESULT_MARKER_END = "__END_RESULT__"


def load_multiview_pipeline(device):
    import torch
    import torch._dynamo
    from multiview.pipeline_multiclass import StableUnCLIPImg2ImgPipeline

    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.force_parameter_static_shapes = False

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



REFINE_WORKER_SCRIPT = Path(__file__).parent / "refine_worker.py"
REFINE_WORKER_PYTHON = Path(__file__).parent / ".venv" / "bin" / "python"
REFINE_WORKER_TIMEOUT = 600
HANG_DETECT_SECONDS = 30


def _run_refine_subprocess(stage: str, level_work_dir: str, params: dict | None = None):
    import select
    import subprocess
    import threading

    cmd = [
        str(REFINE_WORKER_PYTHON),
        str(REFINE_WORKER_SCRIPT),
        "--stage", stage,
        "--work-dir", level_work_dir,
    ]
    if params:
        cmd.extend(["--params-json", json.dumps(params)])

    proc = subprocess.Popen(
        cmd,
        cwd=str(REFINE_WORKER_SCRIPT.parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    last_activity = time.monotonic()
    lock = threading.Lock()

    def read_stderr():
        nonlocal last_activity
        assert proc.stderr is not None
        for line in proc.stderr:
            with lock:
                last_activity = time.monotonic()
            stderr_lines.append(line)
            sys.stderr.write(f"[refine/{stage}] {line}")
            sys.stderr.flush()

    def read_stdout():
        nonlocal last_activity
        assert proc.stdout is not None
        for line in proc.stdout:
            with lock:
                last_activity = time.monotonic()
            stdout_lines.append(line)

    t_err = threading.Thread(target=read_stderr, daemon=True)
    t_out = threading.Thread(target=read_stdout, daemon=True)
    t_err.start()
    t_out.start()

    while proc.poll() is None:
        t_err.join(timeout=1.0)
        with lock:
            idle_sec = time.monotonic() - last_activity
        if idle_sec > HANG_DETECT_SECONDS:
            sys.stderr.write(
                f"[hang_detect] {stage}: no output for {idle_sec:.0f}s, "
                f"killing PID {proc.pid}\n"
            )
            sys.stderr.flush()
            proc.kill()
            proc.wait()
            raise RuntimeError(
                f"refine_worker {stage} hang detected: "
                f"no output for {idle_sec:.0f}s (threshold={HANG_DETECT_SECONDS}s)"
            )

    t_err.join(timeout=5)
    t_out.join(timeout=5)

    if proc.returncode != 0:
        stderr_text = "".join(stderr_lines)
        raise RuntimeError(
            f"refine_worker {stage} failed (exit={proc.returncode}): {stderr_text[-500:]}"
        )

    for line in reversed(stdout_lines):
        stripped = line.strip()
        if stripped.startswith("{"):
            return json.loads(stripped)

    stdout_text = "".join(stdout_lines)
    raise RuntimeError(
        f"refine_worker {stage}: no JSON result in stdout: {stdout_text[-200:]}"
    )



def run_refine(mv_dir: str, slrm_dir: str, work_dir: str):
    tmp_dir = f"/tmp/StdGEN/{os.getpid()}"
    os.makedirs(tmp_dir, exist_ok=True)

    name_to_level = [(3, 2), (1, 1), (2, 0)]

    for mesh_idx, level in name_to_level:
        params = {
            "slrm_dir": slrm_dir,
            "mv_dir": mv_dir,
            "level": level,
            "mesh_idx": mesh_idx,
            "tmp_dir": tmp_dir,
        }

        sys.stderr.write(f"[run_refine] level={level} start\n")
        sys.stderr.flush()
        _run_refine_subprocess("refine_level", tmp_dir, params)
        sys.stderr.write(f"[run_refine] level={level} done\n")
        sys.stderr.flush()

    refine_dir = os.path.join(tmp_dir, "refined")
    return refine_dir


PART_NAMES = ["hair", "body", "full"]
MIN_PART_VERTICES = 300


def _read_glb_geometry(glb_path: str):
    import numpy as np
    from pygltflib import GLTF2

    glb = GLTF2().load(glb_path)
    prim = glb.meshes[0].primitives[0]
    binary = glb.binary_blob()

    def read_accessor(acc_idx):
        acc = glb.accessors[acc_idx]
        bv = glb.bufferViews[acc.bufferView]
        start = bv.byteOffset + (acc.byteOffset or 0)
        return binary[start:start + bv.byteLength]

    indices = np.frombuffer(read_accessor(prim.indices), dtype=np.uint32).copy()
    positions = np.frombuffer(
        read_accessor(prim.attributes.POSITION), dtype=np.float32,
    ).reshape(-1, 3).copy()
    colors = np.frombuffer(
        read_accessor(prim.attributes.COLOR_0), dtype=np.uint8,
    ).reshape(-1, 4).copy()

    return positions, colors, indices


def _filter_noise_fragments(positions, colors, indices):
    import numpy as np
    import trimesh

    mesh = trimesh.Trimesh(vertices=positions, faces=indices.reshape(-1, 3))
    components = mesh.split(only_watertight=False)

    kept = [c for c in components if len(c.vertices) >= MIN_PART_VERTICES]
    if not kept:
        return positions, colors, indices

    all_pos, all_col, all_idx = [], [], []
    offset = 0
    for comp in kept:
        original_indices = comp.metadata.get("face_index")
        original_vertex_mask = np.zeros(len(positions), dtype=bool)
        if original_indices is not None:
            original_faces = indices.reshape(-1, 3)[original_indices]
            original_vertex_mask[np.unique(original_faces)] = True

        if original_vertex_mask.any():
            old_to_new = np.full(len(positions), -1, dtype=np.int64)
            new_verts = positions[original_vertex_mask]
            new_cols = colors[original_vertex_mask]
            old_to_new[original_vertex_mask] = np.arange(len(new_verts))
            new_faces = old_to_new[original_faces.ravel()].reshape(-1, 3)
        else:
            new_verts = np.array(comp.vertices, dtype=np.float32)
            new_cols = np.full((len(new_verts), 4), 200, dtype=np.uint8)
            new_faces = np.array(comp.faces, dtype=np.int64)

        all_pos.append(new_verts)
        all_col.append(new_cols)
        all_idx.append(new_faces.ravel().astype(np.uint32) + offset)
        offset += len(new_verts)

    return (
        np.concatenate(all_pos).astype(np.float32),
        np.concatenate(all_col).astype(np.uint8),
        np.concatenate(all_idx).astype(np.uint32),
    )


OVERLAP_DISTANCE = 0.015


def _remove_overlapping_faces(positions, colors, indices, priority_positions):
    import numpy as np
    from sklearn.neighbors import KDTree

    if len(priority_positions) == 0:
        return positions, colors, indices

    faces = indices.reshape(-1, 3)
    kdtree = KDTree(priority_positions)
    dists, _ = kdtree.query(positions, k=1)
    dists = dists.squeeze()

    close_mask = dists < OVERLAP_DISTANCE
    face_all_close = close_mask[faces].all(axis=1)
    kept_faces = faces[~face_all_close]

    if len(kept_faces) == 0:
        return positions, colors, indices

    used_verts = np.unique(kept_faces)
    old_to_new = np.full(len(positions), -1, dtype=np.int64)
    old_to_new[used_verts] = np.arange(len(used_verts))

    new_positions = positions[used_verts]
    new_colors = colors[used_verts]
    new_indices = old_to_new[kept_faces.ravel()].astype(np.uint32)

    return new_positions, new_colors, new_indices


def combine_refined_glbs(refine_dir: str, output_path: str):
    import numpy as np
    from pygltflib import (
        GLTF2, Mesh, Node, Primitive, Accessor, BufferView, Buffer,
        Material, PbrMetallicRoughness, Attributes, Scene,
    )

    level_data = []
    for level, part_name in enumerate(PART_NAMES):
        glb_path = os.path.join(refine_dir, f"out_{level}.glb")
        positions, colors, indices = _read_glb_geometry(glb_path)
        positions, colors, indices = _filter_noise_fragments(positions, colors, indices)
        level_data.append((positions, colors, indices))

    for level in range(len(PART_NAMES)):
        higher_priority = [level_data[j][0] for j in range(level)]
        if higher_priority:
            priority_positions = np.concatenate(higher_priority)
            positions, colors, indices = level_data[level]
            level_data[level] = _remove_overlapping_faces(
                positions, colors, indices, priority_positions,
            )

    parts_info = []
    blob_parts = []
    accessors = []
    buffer_views = []
    meshes = []
    nodes = []
    byte_offset = 0
    total_vertices, total_faces = 0, 0

    for level, part_name in enumerate(PART_NAMES):
        positions, colors, indices = level_data[level]

        num_verts = len(positions)
        num_faces = len(indices) // 3
        total_vertices += num_verts
        total_faces += num_faces

        idx_bytes = indices.tobytes()
        pos_bytes = positions.tobytes()
        col_bytes = colors.tobytes()

        bv_base = len(buffer_views)
        buffer_views.append(BufferView(
            buffer=0, byteOffset=byte_offset,
            byteLength=len(idx_bytes), target=34963,
        ))
        byte_offset += len(idx_bytes)

        buffer_views.append(BufferView(
            buffer=0, byteOffset=byte_offset,
            byteLength=len(pos_bytes), target=34962,
        ))
        byte_offset += len(pos_bytes)

        buffer_views.append(BufferView(
            buffer=0, byteOffset=byte_offset,
            byteLength=len(col_bytes), target=34962,
        ))
        byte_offset += len(col_bytes)

        acc_base = len(accessors)
        accessors.append(Accessor(
            bufferView=bv_base, componentType=5125,
            count=len(indices), type="SCALAR",
        ))
        accessors.append(Accessor(
            bufferView=bv_base + 1, componentType=5126,
            count=num_verts, type="VEC3",
            max=positions.max(axis=0).tolist(),
            min=positions.min(axis=0).tolist(),
        ))
        accessors.append(Accessor(
            bufferView=bv_base + 2, componentType=5121,
            count=num_verts, type="VEC4", normalized=True,
        ))

        mesh_idx = len(meshes)
        meshes.append(Mesh(
            name=part_name,
            primitives=[Primitive(
                attributes=Attributes(POSITION=acc_base + 1, COLOR_0=acc_base + 2),
                indices=acc_base,
                material=0,
            )],
        ))

        nodes.append(Node(name=part_name, mesh=mesh_idx))
        blob_parts.extend([idx_bytes, pos_bytes, col_bytes])

        parts_info.append({
            "name": part_name,
            "vertex_count": num_verts,
            "face_count": num_faces,
        })

    blob = b"".join(blob_parts)

    out = GLTF2()
    out.asset.generator = "StdGEN-multipart"
    out.scene = 0
    out.scenes = [Scene(nodes=list(range(len(nodes))))]
    out.nodes = nodes
    out.buffers = [Buffer(byteLength=len(blob))]
    out.bufferViews = buffer_views
    out.accessors = accessors
    out.materials = [Material(
        pbrMetallicRoughness=PbrMetallicRoughness(
            baseColorFactor=[1.0, 1.0, 1.0, 1.0],
            metallicFactor=0.0,
            roughnessFactor=1.0,
        ),
        doubleSided=True,
    )]
    out.meshes = meshes
    out.set_binary_blob(blob)
    out.save(output_path)

    return total_vertices, total_faces, parts_info


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

    vertex_count, face_count, parts_info = combine_refined_glbs(refine_dir, output_path)

    sys.stderr.write("Reloading multiview + S-LRM...\n")
    sys.stderr.flush()
    ctx["multiview_pipeline"] = load_multiview_pipeline(ctx["device"])
    ctx["slrm_model"], ctx["slrm_infer_config"] = load_slrm_model(ctx["device"])

    total_ms = (time.monotonic() - t0) * 1000.0

    return {
        "status": "ok",
        "vertex_count": vertex_count,
        "face_count": face_count,
        "parts": parts_info,
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
