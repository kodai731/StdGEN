"""Subprocess worker for refine stages.

Each invocation runs a single refine stage, then exits so the OS reclaims
all GPU memory (including nvdiffrast RasterizeCudaContext).

Usage:
    python refine_worker.py --stage reconstruct_stage1 --work-dir /tmp/refine_work
    python refine_worker.py --stage run_mesh_refine --work-dir /tmp/refine_work
    python refine_worker.py --stage color_projection --work-dir /tmp/refine_work
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np


def _load_images(image_dir: str):
    from PIL import Image

    images = []
    for i in range(6):
        images.append(Image.open(os.path.join(image_dir, f"{i}.png")))
    return images


def _save_images(images, image_dir: str):
    os.makedirs(image_dir, exist_ok=True)
    for i, img in enumerate(images):
        img.save(os.path.join(image_dir, f"{i}.png"))


def _decimate_mesh(vertices, faces, target_faces=40000):
    import pymeshlab as ml

    ms = ml.MeshSet()
    ms.add_mesh(ml.Mesh(vertices, faces))
    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=target_faces,
        preserveboundary=True,
        preservenormal=True,
        preservetopology=True,
        qualitythr=0.5,
    )
    m = ms.current_mesh()
    return m.vertex_matrix().astype("float32"), m.face_matrix()


MAX_INPUT_FACES = 40000


def _build_fixed_adjacency(faces, num_verts):
    import torch
    from refine.remesh import calc_edges

    edges, _ = calc_edges(faces)
    E = edges.shape[0]

    src_indices = torch.cat([edges[:, 1], edges[:, 0]], dim=0)
    dst_indices = torch.cat([edges[:, 0], edges[:, 1]], dim=0)

    neighbor_count = torch.zeros(
        num_verts, 1, device=faces.device, dtype=torch.float32,
    )
    neighbor_count.scatter_add_(
        0,
        dst_indices.unsqueeze(1),
        torch.ones(2 * E, 1, device=faces.device, dtype=torch.float32),
    )
    neighbor_count = neighbor_count.clamp(min=1)

    return src_indices, dst_indices, neighbor_count


def _run_reconstruct_stage1(work_dir: str, params: dict):
    import torch
    from refine.mesh_refine import reconstruct_stage1, simple_remove, erode_alpha
    from PIL import Image

    v_np = np.load(os.path.join(work_dir, "mesh_v.npy"))
    f_np = np.load(os.path.join(work_dir, "mesh_f.npy"))

    target_faces = params.get("max_input_faces", MAX_INPUT_FACES)
    if len(f_np) > target_faces:
        sys.stderr.write(
            f"[reconstruct_stage1] decimating {len(f_np)} -> {target_faces} faces\n"
        )
        sys.stderr.flush()
        v_np, f_np = _decimate_mesh(v_np, f_np, target_faces)

    mesh_v = torch.tensor(v_np, device="cuda", dtype=torch.float32)
    mesh_f = torch.tensor(f_np, device="cuda")

    normal_ls = _load_images(os.path.join(work_dir, "normals"))
    rgb_ls = _load_images(os.path.join(work_dir, "colors"))

    rm_normals = simple_remove(normal_ls)
    for idx, img in enumerate(rm_normals):
        rgb_ls[idx] = Image.fromarray(
            np.concatenate([
                np.array(rgb_ls[idx])[..., :3],
                np.array(img)[:, :, 3:4],
            ], axis=-1)
        )
    rgb_ls = erode_alpha(rgb_ls)

    _save_images(rgb_ls, os.path.join(work_dir, "rgb_processed"))
    _save_images(rm_normals, os.path.join(work_dir, "rm_normals"))

    distract_mask = None
    if os.path.exists(os.path.join(work_dir, "distract_mask.npy")):
        distract_mask = np.load(os.path.join(work_dir, "distract_mask.npy"))

    distract_bbox = None
    if os.path.exists(os.path.join(work_dir, "distract_bbox.npy")):
        distract_bbox = np.load(os.path.join(work_dir, "distract_bbox.npy"))

    vertices, faces = reconstruct_stage1(
        rm_normals,
        steps=params.get("steps", 200),
        vertices=mesh_v, faces=mesh_f,
        fixed_v=None, fixed_f=None,
        lr=params.get("lr", 0.08),
        remesh_interval=params.get("remesh_interval", 1),
        start_edge_len=params.get("start_edge_len", 0.02),
        end_edge_len=params.get("end_edge_len", 0.005),
        gain=params.get("gain", 0.05),
        loss_expansion_weight=params.get("expansion_weight", 0.1),
        distract_mask=distract_mask,
        distract_bbox=distract_bbox,
    )

    v_np = vertices.detach().cpu().numpy()
    f_np = faces.detach().cpu().numpy()
    np.save(os.path.join(work_dir, "stage1_vertices.npy"), v_np)
    np.save(os.path.join(work_dir, "stage1_faces.npy"), f_np)

    return {"vertex_count": len(v_np), "face_count": len(f_np)}


def _run_mesh_refine(work_dir: str, params: dict):
    import torch
    from refine.mesh_refine import run_mesh_refine

    vertices = torch.tensor(
        np.load(os.path.join(work_dir, "stage1_vertices.npy")),
        device="cuda", dtype=torch.float32,
    )
    faces = torch.tensor(
        np.load(os.path.join(work_dir, "stage1_faces.npy")),
        device="cuda",
    )

    rm_normals = _load_images(os.path.join(work_dir, "rm_normals"))

    vertices, faces = run_mesh_refine(
        vertices, faces, rm_normals,
        fixed_v=None, fixed_f=None,
        steps=params.get("steps", 100),
        start_edge_len=params.get("start_edge_len", 0.005),
        end_edge_len=params.get("end_edge_len", 0.001),
        decay=params.get("decay", 0.99),
        update_normal_interval=params.get("update_normal_interval", 20),
        update_warmup=params.get("update_warmup", 2),
        process_inputs=False,
        process_outputs=False,
        remesh_interval=params.get("remesh_interval", 5),
    )

    v_np = vertices.detach().cpu().numpy()
    f_np = faces.detach().cpu().numpy()
    np.save(os.path.join(work_dir, "refine_vertices.npy"), v_np)
    np.save(os.path.join(work_dir, "refine_faces.npy"), f_np)

    return {"vertex_count": len(v_np), "face_count": len(f_np)}


def _run_color_projection(work_dir: str, params: dict):
    import gc
    import torch
    import trimesh
    import pytorch3d
    from pytorch3d.structures import Meshes
    from refine.mesh_refine import merge_small_faces
    from refine.func import (
        get_cameras_list, multiview_color_projection,
        simple_clean_mesh, to_pyml_mesh,
    )
    from infer_refine import save_py3dmesh_with_trimesh_fast

    vertices = np.load(os.path.join(work_dir, "refine_vertices.npy"))
    faces = np.load(os.path.join(work_dir, "refine_faces.npy"))

    vertices_t = torch.tensor(vertices, device="cuda", dtype=torch.float32)
    faces_t = torch.tensor(faces, device="cuda")

    meshes = simple_clean_mesh(
        to_pyml_mesh(vertices_t, faces_t),
        apply_smooth=True, stepsmoothnum=2,
        apply_sub_divide=False, sub_divide_threshold=0.25,
    ).to("cuda")

    vertices = meshes.verts_packed().detach().cpu().numpy()
    faces = meshes.faces_packed().detach().cpu().numpy()

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh = merge_small_faces(mesh, thres=params.get("thres", 3e-6))
    parts = mesh.split(only_watertight=False)
    parts = [p for p in parts if len(p.vertices) >= 200]
    mesh = trimesh.Scene(parts).dump(concatenate=True)
    vertices, faces = mesh.vertices.astype("float32"), mesh.faces

    vertices, faces = trimesh.remesh.subdivide(vertices, faces)
    vertices_t = torch.tensor(vertices, device="cuda")
    faces_t = torch.tensor(faces, device="cuda")

    gc.collect()
    torch.cuda.empty_cache()

    meshes = Meshes(
        verts=[vertices_t], faces=[faces_t],
        textures=pytorch3d.renderer.mesh.textures.TexturesVertex(
            [torch.zeros_like(vertices_t).float()]
        ),
    )

    rgb_ls = _load_images(os.path.join(work_dir, "rgb_processed"))

    distract_mask = None
    if os.path.exists(os.path.join(work_dir, "distract_mask.npy")):
        distract_mask = np.load(os.path.join(work_dir, "distract_mask.npy"))

    cameras_list = get_cameras_list(
        [180, 225, 270, 0, 90, 135], "cuda", focal=1 / 1.2,
    )
    mvp_weights = (
        [2.0, 0.0, 0.5, 1.0, 0.5, 0.0] if distract_mask is not None
        else [2.0, 0.5, 0.0, 1.0, 0.0, 0.5]
    )

    new_meshes = multiview_color_projection(
        meshes, rgb_ls, resolution=1024, device="cuda",
        complete_unseen=True, confidence_threshold=0.2,
        cameras_list=cameras_list, weights=mvp_weights,
        distract_mask=distract_mask,
    )

    del meshes, cameras_list
    gc.collect()
    torch.cuda.empty_cache()

    output_path = os.path.join(work_dir, "output.glb")
    save_py3dmesh_with_trimesh_fast(
        new_meshes, output_path, apply_sRGB_to_LinearRGB=True,
    )

    return {
        "vertex_count": int(new_meshes.verts_packed().shape[0]),
        "face_count": int(new_meshes.faces_packed().shape[0]),
    }


def _load_multiview_images(mv_dir, level, ref_colors, ref_mask, remove_mask=None):
    import cv2
    from PIL import Image
    from infer_refine import calc_horizontal_offset, calc_horizontal_offset2

    read_level = 0 if remove_mask is not None else level

    colors, normals = [], []
    for i in range(6):
        color = cv2.imread(os.path.join(mv_dir, f"level{read_level}", f"color_{i}.png"))[..., ::-1]
        normal = cv2.imread(os.path.join(mv_dir, f"level{read_level}", f"normal_{i}.png"))[..., ::-1]

        if ref_colors is not None:
            offset = calc_horizontal_offset(np.array(ref_colors[i]), color)
        else:
            offset = calc_horizontal_offset2(ref_mask[i], color)

        if offset != 0:
            color = np.roll(color, offset, axis=1)
            normal = np.roll(normal, offset, axis=1)

        if remove_mask is not None:
            color[remove_mask[i]] = 255
            normal[remove_mask[i]] = 255

        colors.append(Image.fromarray(color))
        normals.append(Image.fromarray(normal))

    return colors, normals


def _generate_ref_mask(mesh_v, mesh_f):
    import torch
    from refine.func import make_star_cameras_orthographic
    from refine.render import NormalsRenderer

    mv, proj = make_star_cameras_orthographic(8, 1, r=1.2)
    mv = mv[[4, 3, 2, 0, 6, 5]]
    renderer = NormalsRenderer(mv, proj, (1024, 1024))
    images = renderer.render(
        torch.tensor(mesh_v, device="cuda").float(),
        torch.ones_like(torch.from_numpy(np.array(mesh_v)), device="cuda").float(),
        torch.tensor(np.array(mesh_f), device="cuda"),
    )
    ref_mask = (images[..., 3] < 0.9).cpu().numpy()
    del renderer, images, mv, proj
    return ref_mask


def _render_silhouette_mask(mesh_path):
    import gc
    import torch
    import trimesh
    from refine.func import make_star_cameras_orthographic
    from refine.render import NormalsRenderer

    mesh = trimesh.load(mesh_path)
    parts = mesh.split(only_watertight=False)
    parts = [p for p in parts if len(p.vertices) >= 300]
    if not parts:
        return np.zeros((6, 1024, 1024), dtype=bool)
    mesh = trimesh.Scene(parts).to_geometry()

    mv, proj = make_star_cameras_orthographic(8, 1, r=1.2)
    mv = mv[[4, 3, 2, 0, 6, 5]]
    renderer = NormalsRenderer(mv, proj, (1024, 1024))
    images = renderer.render(
        torch.tensor(np.array(mesh.vertices), device="cuda").float(),
        torch.ones(len(mesh.vertices), 3, device="cuda").float(),
        torch.tensor(np.array(mesh.faces), device="cuda"),
    )
    mask = (images[..., 3] > 0.5).cpu().numpy()
    del renderer, images, mv, proj
    gc.collect()
    torch.cuda.empty_cache()
    return mask


def _run_generate_distract_mask(work_dir: str, params: dict):
    import cv2
    from PIL import Image
    from infer_refine import get_distract_mask, _unload_sam, calc_horizontal_offset, calc_horizontal_offset2

    tmp_dir = params["tmp_dir"]
    mv_dir = params["mv_dir"]
    level = params["level"]

    last_front_color = np.load(os.path.join(tmp_dir, "last_front_color.npy"))

    ref_mask_path = os.path.join(tmp_dir, "ref_mask.npy")
    ref_colors_path = os.path.join(tmp_dir, "ref_colors")

    ref_colors = None
    if os.path.exists(ref_colors_path):
        ref_colors = []
        for i in range(6):
            ref_colors.append(Image.open(os.path.join(ref_colors_path, f"{i}.png")))

    ref_mask = None
    if os.path.exists(ref_mask_path):
        ref_mask = np.load(ref_mask_path)

    color_0 = cv2.imread(os.path.join(mv_dir, f"level{level}", "color_0.png"))[..., ::-1]
    if ref_colors is not None:
        offset = calc_horizontal_offset(np.array(ref_colors[0]), color_0)
    elif ref_mask is not None:
        offset = calc_horizontal_offset2(ref_mask[0], color_0)
    else:
        offset = 0
    if offset != 0:
        color_0 = np.roll(color_0, offset, axis=1)

    current_front_color = color_0.astype(np.float32) / 255.0

    _, distract_bbox, _, distract_mask = get_distract_mask(
        last_front_color,
        current_front_color,
        outside_ratio=0.20,
    )
    _unload_sam()

    np.save(os.path.join(tmp_dir, "distract_mask.npy"), distract_mask)
    np.save(os.path.join(tmp_dir, "distract_bbox.npy"), distract_bbox)

    return {"status": "ok"}


def _laplacian_smooth(vertices, src_indices, dst_indices, neighbor_count, iterations=2):
    import torch

    v = vertices.clone()
    E2 = src_indices.shape[0]
    for _ in range(iterations):
        neighbor_sum = torch.zeros_like(v)
        neighbor_sum.scatter_add_(
            0,
            dst_indices.unsqueeze(1).expand(E2, 3),
            v[src_indices],
        )
        neighbor_mean = neighbor_sum / neighbor_count
        v = 0.5 * v + 0.5 * neighbor_mean
    return v


def _run_refine_level(work_dir: str, params: dict):
    import gc
    from copy import deepcopy

    import torch
    from PIL import Image
    from tqdm import tqdm

    from refine.mesh_refine import simple_remove, erode_alpha, init_target
    from refine.func import (
        get_cameras_list, multiview_color_projection,
        make_star_cameras_orthographic, to_py3d_mesh, from_py3d_mesh,
    )
    from refine.render import NormalsRenderer, calc_vertex_normals
    from infer_refine import save_py3dmesh_with_trimesh_fast
    import pytorch3d
    from pytorch3d.structures import Meshes
    import trimesh

    slrm_dir = params["slrm_dir"]
    mv_dir = params["mv_dir"]
    level = params["level"]
    mesh_idx = params["mesh_idx"]
    tmp_dir = params["tmp_dir"]

    ref_mask_path = os.path.join(tmp_dir, "ref_mask.npy")
    last_front_color_path = os.path.join(tmp_dir, "last_front_color.npy")

    mesh = trimesh.load(os.path.join(slrm_dir, f"mesh_{mesh_idx}.obj"))
    parts = mesh.split(only_watertight=False)
    parts = [p for p in parts if len(p.vertices) >= 300]
    mesh = trimesh.Scene(parts).to_geometry()
    mesh_v, mesh_f = mesh.vertices, mesh.faces

    if os.path.exists(ref_mask_path):
        ref_mask = np.load(ref_mask_path)
    else:
        ref_mask = _generate_ref_mask(mesh_v, mesh_f)
        np.save(ref_mask_path, ref_mask)
        gc.collect()
        torch.cuda.empty_cache()

    ref_colors = None
    ref_colors_path = os.path.join(tmp_dir, "ref_colors")
    if os.path.exists(ref_colors_path):
        ref_colors = []
        for i in range(6):
            ref_colors.append(Image.open(os.path.join(ref_colors_path, f"{i}.png")))
    else:
        os.makedirs(ref_colors_path, exist_ok=True)

    era3d_mode = not os.path.exists(os.path.join(mv_dir, "level1"))
    remove_mask = None
    if era3d_mode and level != 0:
        hair_mesh_path = os.path.join(slrm_dir, "mesh_2.obj")
        remove_mask = _render_silhouette_mask(hair_mesh_path)

    colors, normals = _load_multiview_images(mv_dir, level, ref_colors, ref_mask, remove_mask)

    if ref_colors is None:
        ref_colors = deepcopy(colors)
        for i, img in enumerate(ref_colors):
            img.save(os.path.join(ref_colors_path, f"{i}.png"))

    current_front_color = np.array(colors[0]).astype(np.float32) / 255.0
    np.save(last_front_color_path, current_front_color)

    distract_mask, distract_bbox = None, None
    if level == 0:
        distract_mask_path = os.path.join(tmp_dir, "distract_mask.npy")
        distract_bbox_path = os.path.join(tmp_dir, "distract_bbox.npy")
        if os.path.exists(distract_mask_path):
            distract_mask = np.load(distract_mask_path)
        if os.path.exists(distract_bbox_path):
            distract_bbox = np.load(distract_bbox_path)

    F = len(mesh_f)
    if F > MAX_INPUT_FACES:
        sys.stderr.write(f"[refine_level] decimating {F} -> {MAX_INPUT_FACES} faces\n")
        sys.stderr.flush()
        mesh_v_dec, mesh_f_dec = _decimate_mesh(
            np.array(mesh_v, dtype="float32"),
            np.array(mesh_f),
            MAX_INPUT_FACES,
        )
    else:
        mesh_v_dec = np.array(mesh_v, dtype="float32")
        mesh_f_dec = np.array(mesh_f)

    vertices = torch.tensor(mesh_v_dec, device="cuda", dtype=torch.float32)
    faces = torch.tensor(mesh_f_dec, device="cuda").long()
    V_fixed = vertices.shape[0]
    F_fixed = faces.shape[0]
    sys.stderr.write(
        f"[refine_level] fixed topology: V={V_fixed}, F={F_fixed}\n"
    )
    sys.stderr.flush()

    src_indices, dst_indices, neighbor_count = _build_fixed_adjacency(
        faces, V_fixed,
    )

    normal_ls = list(normals)
    rgb_ls = list(colors)
    rm_normals = simple_remove(normal_ls)
    for idx, img in enumerate(rm_normals):
        rgb_ls[idx] = Image.fromarray(
            np.concatenate([
                np.array(rgb_ls[idx])[..., :3],
                np.array(img)[:, :, 3:4],
            ], axis=-1)
        )
    rgb_ls = erode_alpha(rgb_ls)

    mv, proj = make_star_cameras_orthographic(8, 1, r=1.2)
    mv = mv[[4, 3, 2, 0, 6, 5]]
    render_size = list(rm_normals[0].size)
    renderer = NormalsRenderer(mv, proj, render_size)
    target_images = init_target(rm_normals, new_bkgd=(0., 0., 0.))
    alpha_mask = target_images[..., -1] < 0.5

    vertices_opt = vertices.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([vertices_opt], lr=0.08)

    PHASE1_STEPS = 200
    PHASE2_STEPS = 100
    TOTAL_STEPS = PHASE1_STEPS + PHASE2_STEPS
    LR_DECAY = 0.995
    LAPLACIAN_WEIGHT = 0.02
    EXPANSION_WEIGHT = 0.1
    REFINE_NORMAL_INTERVAL = 25

    sys.stderr.write(f"[refine_level] level={level} geometry optimization\n")
    sys.stderr.flush()

    debug_images = None
    E2 = src_indices.shape[0]
    dst_expanded = dst_indices.unsqueeze(1).expand(E2, 3).contiguous()

    for i in tqdm(range(TOTAL_STEPS)):
        optimizer.zero_grad()

        for pg in optimizer.param_groups:
            pg["lr"] *= LR_DECAY

        normals_v = calc_vertex_normals(vertices_opt, faces)

        if i < PHASE1_STEPS:
            normals_render = normals_v.clone()
            normals_render[:, 0] *= -1
            normals_render[:, 2] *= -1

            images = renderer.render(vertices_opt, normals_render, faces)

            loss_expand = 0.5 * (
                (vertices_opt + normals_v).detach() - vertices_opt
            ).pow(2).mean()

            t_mask = images[..., -1] > 0.5
            loss_target = (
                images[t_mask] - target_images[t_mask]
            ).abs().pow(2).mean()
            loss_alpha = (
                images[..., -1][alpha_mask] - target_images[..., -1][alpha_mask]
            ).pow(2).mean()

            loss = loss_target + loss_alpha + loss_expand * EXPANSION_WEIGHT

            if distract_mask is not None:
                hair_normals = normals_render.clone()
                _images = renderer.render(vertices_opt, hair_normals, faces)
                loss_distract = (
                    _images[0][distract_mask] - target_images[0][distract_mask]
                ).pow(2).mean()

                target_outside = target_images[0][..., :3].clone()
                target_outside[~distract_mask] = 0.0
                loss_outside = (
                    _images[0][..., :3][~distract_mask]
                    - target_outside[..., :3][~distract_mask]
                ).pow(2).mean()

                loss = loss + loss_distract + loss_outside * 10.0
                del _images, target_outside

            del images, normals_render

        else:
            images = renderer.render(vertices_opt, normals_v, faces)

            should_update = (i == PHASE1_STEPS) or (
                (i - PHASE1_STEPS) % REFINE_NORMAL_INTERVAL == 0
            )
            if should_update:
                with torch.no_grad():
                    py3d_mesh = to_py3d_mesh(vertices_opt, faces, normals_v)
                    cameras = get_cameras_list(
                        azim_list=[180, 225, 270, 0, 90, 135],
                        device=vertices_opt.device, focal=1 / 1.2,
                    )
                    projected = multiview_color_projection(
                        py3d_mesh, rm_normals, cameras_list=cameras,
                        weights=[2, 0.8, 0.8, 2, 0.8, 0.8],
                        confidence_threshold=0.1, complete_unseen=False,
                        below_confidence_strategy="original",
                        reweight_with_cosangle="linear",
                    )
                    _, _, target_normal = from_py3d_mesh(projected)
                    target_normal = target_normal * 2 - 1
                    target_normal = torch.nn.functional.normalize(
                        target_normal, dim=-1,
                    )
                    target_normal[:, 0] *= -1
                    target_normal[:, 2] *= -1

                    del debug_images
                    debug_images = renderer.render(
                        vertices_opt, target_normal, faces,
                    )
                    del projected, py3d_mesh, cameras, target_normal

            d_mask = images[..., -1] > 0.5
            loss = (
                images[..., :3][d_mask] - debug_images[..., :3][d_mask]
            ).pow(2).mean()
            loss = loss + (
                images[..., -1][alpha_mask]
                - target_images[..., -1][alpha_mask]
            ).pow(2).mean()

            del images

        with torch.no_grad():
            neighbor_sum = torch.zeros(V_fixed, 3, device="cuda")
            neighbor_sum.scatter_add_(0, dst_expanded, vertices_opt.detach()[src_indices])
            neighbor_mean = neighbor_sum / neighbor_count
        loss_lap = ((vertices_opt - neighbor_mean.detach()) ** 2).mean() * LAPLACIAN_WEIGHT
        loss = loss + loss_lap

        loss = loss + (vertices_opt.abs() > 0.99).float().mean() * 10

        loss.backward()
        optimizer.step()

        del loss, normals_v, loss_lap, neighbor_sum, neighbor_mean

    vertices_final = vertices_opt.detach()
    del optimizer, vertices_opt, debug_images, dst_expanded

    sys.stderr.write(f"[refine_level] level={level} laplacian smoothing\n")
    sys.stderr.flush()

    vertices_final = _laplacian_smooth(
        vertices_final, src_indices, dst_indices, neighbor_count, iterations=2,
    )

    sys.stderr.write(f"[refine_level] level={level} color_projection\n")
    sys.stderr.flush()

    color_meshes = Meshes(
        verts=[vertices_final], faces=[faces],
        textures=pytorch3d.renderer.mesh.textures.TexturesVertex(
            [torch.zeros_like(vertices_final).float()]
        ),
    )

    cameras_list = get_cameras_list(
        [180, 225, 270, 0, 90, 135], "cuda", focal=1 / 1.2,
    )
    mvp_weights = (
        [2.0, 0.0, 0.5, 1.0, 0.5, 0.0] if distract_mask is not None
        else [2.0, 0.5, 0.0, 1.0, 0.0, 0.5]
    )

    new_meshes = multiview_color_projection(
        color_meshes, rgb_ls, resolution=1024, device="cuda",
        complete_unseen=True, confidence_threshold=0.2,
        cameras_list=cameras_list, weights=mvp_weights,
        distract_mask=distract_mask,
    )

    del color_meshes, cameras_list
    gc.collect()
    torch.cuda.empty_cache()

    refined_dir = os.path.join(tmp_dir, "refined")
    os.makedirs(refined_dir, exist_ok=True)
    output_path = os.path.join(refined_dir, f"out_{level}.glb")
    save_py3dmesh_with_trimesh_fast(new_meshes, output_path, apply_sRGB_to_LinearRGB=True)

    vertex_count = int(new_meshes.verts_packed().shape[0])
    face_count = int(new_meshes.faces_packed().shape[0])
    del new_meshes

    return {"vertex_count": vertex_count, "face_count": face_count}


STAGES = {
    "reconstruct_stage1": _run_reconstruct_stage1,
    "run_mesh_refine": _run_mesh_refine,
    "color_projection": _run_color_projection,
    "refine_level": _run_refine_level,
    "generate_distract_mask": _run_generate_distract_mask,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=STAGES.keys())
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--params-json", default="{}")
    args = parser.parse_args()

    params = json.loads(args.params_json)

    t0 = time.monotonic()
    sys.stderr.write(f"[refine_worker] stage={args.stage} start\n")
    sys.stderr.flush()

    result = STAGES[args.stage](args.work_dir, params)

    elapsed_ms = (time.monotonic() - t0) * 1000
    result["elapsed_ms"] = elapsed_ms
    sys.stderr.write(f"[refine_worker] stage={args.stage} done ({elapsed_ms:.0f}ms)\n")
    sys.stderr.flush()

    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
