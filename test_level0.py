"""Test Stage 3 Level 0 only, with cached fixed_v/fixed_f from Level 2+1.

First run: processes Level 2 → 1 → saves fixed_v/fixed_f → runs Level 0
Subsequent runs: loads cached fixed_v/fixed_f → runs Level 0 directly

Usage:
    .venv/bin/python test_level0.py
    .venv/bin/python test_level0.py --rebuild-cache
"""
import os

os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.expanduser("~/.cache/torch/inductor"))

import argparse
import gc
import time

import cv2
import numpy as np
import torch
import trimesh
from PIL import Image

from refine.func import make_star_cameras_orthographic
from refine.mesh_refine import geo_refine
from refine.render import NormalsRenderer, calc_vertex_normals
from sklearn.neighbors import KDTree

from infer_refine import (
    calc_horizontal_offset,
    calc_horizontal_offset2,
    filter_fixed_mesh_by_proximity,
    get_distract_mask,
    save_py3dmesh_with_trimesh_fast,
    _unload_sam,
)
from vram_monitor import init_log, log_vram, set_stage

DATA_DIR = "/tmp/StdGEN"
CACHE_PATH = os.path.join(DATA_DIR, "fixed_cache.pt")
OUTPUT_DIR = os.path.join(DATA_DIR, "refined")


def build_fixed_cache():
    mv_root_dir = os.path.join(DATA_DIR, "multiview", "test")
    fixed_v, fixed_f = None, None
    last_colors, last_normals = None, None

    mv, proj = make_star_cameras_orthographic(8, 1, r=1.2)
    mv = mv[[4, 3, 2, 0, 6, 5]]
    renderer = NormalsRenderer(mv, proj, (1024, 1024))

    for name_idx, level in zip([3, 1], [2, 1]):
        gc.collect()
        torch.cuda.empty_cache()
        set_stage(f"decompose_level{level}", step=0)
        log_vram(f"start level{level}")

        mesh = trimesh.load(os.path.join(DATA_DIR, f"test_{name_idx}.obj"))
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
            color = cv2.imread(os.path.join(mv_root_dir, f"level{level}", f"color_{i}.png"))[..., ::-1]
            normal = cv2.imread(os.path.join(mv_root_dir, f"level{level}", f"normal_{i}.png"))[..., ::-1]

            if last_colors is not None:
                offset = calc_horizontal_offset(np.array(last_colors[i]), color)
            else:
                offset = calc_horizontal_offset2(mask[i], color)

            if offset != 0:
                color = np.roll(color, offset, axis=1)
                normal = np.roll(normal, offset, axis=1)

            colors.append(Image.fromarray(color))
            normals.append(Image.fromarray(normal))

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

        set_stage(f"geo_refine_level{level}")
        log_vram(f"before geo_refine level{level}")
        new_mesh, simp_v, simp_f = geo_refine(mesh_v_t, mesh_f_t, colors, normals, fixed_v=fixed_v, fixed_f=fixed_f)
        log_vram(f"after geo_refine level{level}")

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        save_py3dmesh_with_trimesh_fast(new_mesh, os.path.join(OUTPUT_DIR, f"out_{level}.glb"), apply_sRGB_to_LinearRGB=True)

        if fixed_v is None:
            fixed_v, fixed_f = simp_v.cpu(), simp_f.cpu()
        else:
            fixed_f = torch.cat([fixed_f, simp_f.cpu() + fixed_v.shape[0]], dim=0)
            fixed_v = torch.cat([fixed_v, simp_v.cpu()], dim=0)

        del new_mesh, simp_v, simp_f, colors, normals
        gc.collect()
        torch.cuda.empty_cache()

    last_front_color = np.array(last_colors[0]).astype(np.float32) / 255.0

    torch.save({
        "fixed_v": fixed_v,
        "fixed_f": fixed_f,
        "last_colors": [np.array(c) for c in last_colors],
        "last_front_color": last_front_color,
    }, CACHE_PATH)
    print(f"Cache saved: fixed_v={fixed_v.shape[0]}, fixed_f={fixed_f.shape[0]}")

    del last_colors, last_normals, renderer
    gc.collect()
    torch.cuda.empty_cache()

    return fixed_v, fixed_f, last_front_color


def run_level0(fixed_v, fixed_f, last_front_color):
    mv_root_dir = os.path.join(DATA_DIR, "multiview", "test")
    name_idx = 2
    level = 0

    gc.collect()
    torch.cuda.empty_cache()
    set_stage("decompose_level0", step=0)
    log_vram("start level0")

    mesh = trimesh.load(os.path.join(DATA_DIR, f"test_{name_idx}.obj"))
    new_mesh = mesh.split(only_watertight=False)
    new_mesh = [j for j in new_mesh if len(j.vertices) >= 300]
    mesh = trimesh.Scene(new_mesh).to_geometry()
    mesh_v, mesh_f = mesh.vertices, mesh.faces

    colors, normals = [], []
    for i in range(6):
        color = cv2.imread(os.path.join(mv_root_dir, f"level{level}", f"color_{i}.png"))[..., ::-1]
        normal = cv2.imread(os.path.join(mv_root_dir, f"level{level}", f"normal_{i}.png"))[..., ::-1]
        colors.append(Image.fromarray(color))
        normals.append(Image.fromarray(normal))

    original_mask, distract_bbox, _, distract_mask = get_distract_mask(
        last_front_color,
        np.array(colors[0]).astype(np.float32) / 255.0,
        outside_ratio=0.20,
    )
    _unload_sam()
    log_vram("after SAM unload")

    gc.collect()
    torch.cuda.empty_cache()

    mesh_v_t = torch.tensor(mesh_v, device="cuda", dtype=torch.float32)
    mesh_f_t = torch.tensor(mesh_f, device="cuda")

    level_fixed_v, level_fixed_f = filter_fixed_mesh_by_proximity(
        fixed_v, fixed_f, mesh_v_t.cpu(), margin=0.15,
    )
    print(f"fixed_v filtered: {fixed_v.shape[0]} -> {level_fixed_v.shape[0]}")
    log_vram("after proximity filter")

    set_stage("geo_refine_level0")
    log_vram("before geo_refine level0")
    t0 = time.monotonic()
    result_mesh, simp_v, simp_f = geo_refine(
        mesh_v_t, mesh_f_t, colors, normals,
        fixed_v=level_fixed_v, fixed_f=level_fixed_f,
        distract_mask=distract_mask, distract_bbox=distract_bbox,
    )
    elapsed = time.monotonic() - t0
    log_vram("after geo_refine level0")
    print(f"Level 0 completed in {elapsed:.1f}s")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_py3dmesh_with_trimesh_fast(result_mesh, os.path.join(OUTPUT_DIR, "out_0.glb"), apply_sRGB_to_LinearRGB=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()

    init_log()

    if os.path.exists(CACHE_PATH) and not args.rebuild_cache:
        print(f"Loading cached fixed_v/fixed_f from {CACHE_PATH}")
        cache = torch.load(CACHE_PATH, weights_only=False)
        fixed_v = cache["fixed_v"]
        fixed_f = cache["fixed_f"]
        last_front_color = cache["last_front_color"]
        print(f"Loaded: fixed_v={fixed_v.shape[0]}, fixed_f={fixed_f.shape[0]}")
    else:
        print("Building cache (Level 2 + 1)...")
        fixed_v, fixed_f, last_front_color = build_fixed_cache()

    run_level0(fixed_v, fixed_f, last_front_color)


if __name__ == "__main__":
    main()
