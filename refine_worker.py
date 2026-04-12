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
        STDGEN_VIEWS,
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

    views = STDGEN_VIEWS
    cameras_list = get_cameras_list(views.azim_list, "cuda", focal=1 / 1.2)
    mvp_weights = views.get_projection_weights(distract=distract_mask is not None)

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


def _load_multiview_images(mv_dir, level, ref_colors, ref_mask):
    import cv2
    from PIL import Image
    from infer_refine import calc_horizontal_offset, calc_horizontal_offset2

    colors, normals = [], []
    for i in range(6):
        color = cv2.imread(os.path.join(mv_dir, f"level{level}", f"color_{i}.png"))[..., ::-1]
        normal = cv2.imread(os.path.join(mv_dir, f"level{level}", f"normal_{i}.png"))[..., ::-1]

        if ref_colors is not None:
            offset = calc_horizontal_offset(np.array(ref_colors[i]), color)
        else:
            offset = calc_horizontal_offset2(ref_mask[i], color)

        if offset != 0:
            color = np.roll(color, offset, axis=1)
            normal = np.roll(normal, offset, axis=1)

        colors.append(Image.fromarray(color))
        normals.append(Image.fromarray(normal))

    return colors, normals


def _generate_ref_mask(mesh_v, mesh_f, resolution=1024, camera_indices=None):
    import torch
    from refine.func import STDGEN_VIEWS, make_star_cameras_orthographic
    from refine.render import NormalsRenderer

    mv, proj = make_star_cameras_orthographic(8, 1, r=1.2)
    mv = mv[camera_indices or STDGEN_VIEWS.camera_indices]
    renderer = NormalsRenderer(mv, proj, (resolution, resolution))
    images = renderer.render(
        torch.tensor(mesh_v, device="cuda").float(),
        torch.ones_like(torch.from_numpy(np.array(mesh_v)), device="cuda").float(),
        torch.tensor(np.array(mesh_f), device="cuda"),
    )
    ref_mask = (images[..., 3] < 0.9).cpu().numpy()
    del renderer, images, mv, proj
    return ref_mask



def _render_part_mask(mesh_path: str, vertex_mask: np.ndarray):
    import gc
    import torch
    import trimesh
    from refine.func import STDGEN_VIEWS, make_star_cameras_orthographic
    from refine.render import NormalsRenderer

    mesh = trimesh.load(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_geometry()

    vertices = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces)

    colors = np.zeros((len(vertices), 3), dtype=np.float32)
    colors[vertex_mask] = 1.0

    mv, proj = make_star_cameras_orthographic(8, 1, r=1.2)
    mv = mv[STDGEN_VIEWS.camera_indices]
    renderer = NormalsRenderer(mv, proj, (1024, 1024))
    images = renderer.render(
        torch.tensor(vertices, device="cuda").float(),
        torch.tensor(colors, device="cuda").float(),
        torch.tensor(faces, device="cuda"),
    )
    mask = (images[..., 0] > 0.3).cpu().numpy()
    del renderer, images, mv, proj
    gc.collect()
    torch.cuda.empty_cache()
    return mask


def _run_generate_levels(work_dir: str, params: dict):
    import cv2
    import torch
    import trimesh

    mv_dir = params["mv_dir"]
    slrm_dir = params["slrm_dir"]

    level0_dir = os.path.join(mv_dir, "level0")

    if os.path.exists(os.path.join(mv_dir, "level1")):
        return {"status": "ok", "skipped": True}

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    src_dir = os.path.join(project_root, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from anim_ml.utils.mesh_contraction import segment_humanoid, LABEL_HEAD

    mesh_path = os.path.join(slrm_dir, "mesh_0.obj")
    mesh = trimesh.load(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_geometry()

    V = torch.tensor(np.array(mesh.vertices), dtype=torch.float32)
    F = torch.tensor(np.array(mesh.faces), dtype=torch.long)

    sys.stderr.write("[generate_levels] contracting mesh_0 for part labeling\n")
    sys.stderr.flush()
    labels = segment_humanoid(V, F, head_ratio=0.20, device="cuda")
    head_mask = (labels == LABEL_HEAD).numpy()
    body_mask = ~head_mask

    sys.stderr.write(f"[generate_levels] head={head_mask.sum()} body={body_mask.sum()}\n")
    sys.stderr.flush()

    head_silhouette = _render_part_mask(mesh_path, head_mask)
    body_silhouette = _render_part_mask(mesh_path, body_mask)

    for level, silhouette in [(2, head_silhouette), (1, body_silhouette)]:
        level_dir = os.path.join(mv_dir, f"level{level}")
        os.makedirs(level_dir, exist_ok=True)

        for i in range(6):
            for kind in ("color", "normal"):
                src = cv2.imread(os.path.join(level0_dir, f"{kind}_{i}.png"))
                src[~silhouette[i]] = 255
                cv2.imwrite(os.path.join(level_dir, f"{kind}_{i}.png"), src)

        sys.stderr.write(f"[generate_levels] level{level} created\n")
        sys.stderr.flush()

    return {"status": "ok"}


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



def _run_refine_level(work_dir: str, params: dict):
    import gc
    from copy import deepcopy

    import torch
    from PIL import Image

    from refine.mesh_refine import geo_refine
    from infer_refine import save_py3dmesh_with_trimesh_fast
    import trimesh

    slrm_dir = params["slrm_dir"]
    mv_dir = params["mv_dir"]
    level = params["level"]
    mesh_idx = params["mesh_idx"]
    tmp_dir = params["tmp_dir"]
    normal_flip = params.get("normal_flip")

    last_front_color_path = os.path.join(tmp_dir, "last_front_color.npy")

    mesh = trimesh.load(os.path.join(slrm_dir, f"mesh_{mesh_idx}.obj"))
    parts = mesh.split(only_watertight=False)
    parts = [p for p in parts if len(p.vertices) >= 300]
    mesh = trimesh.Scene(parts).to_geometry()
    mesh_v, mesh_f = mesh.vertices, mesh.faces

    import cv2
    sample_img = cv2.imread(os.path.join(mv_dir, f"level{level}", "color_0.png"))
    mv_resolution = sample_img.shape[0]

    level_ref_mask_path = os.path.join(tmp_dir, f"ref_mask_level{level}.npy")
    if os.path.exists(level_ref_mask_path):
        ref_mask = np.load(level_ref_mask_path)
    else:
        from refine.func import ERA3D_VIEWS, STDGEN_VIEWS
        view_cfg = ERA3D_VIEWS if normal_flip is not None else STDGEN_VIEWS
        ref_mask = _generate_ref_mask(mesh_v, mesh_f, resolution=mv_resolution,
                                      camera_indices=view_cfg.camera_indices)
        np.save(level_ref_mask_path, ref_mask)
        gc.collect()
        torch.cuda.empty_cache()

    ref_colors = None
    ref_colors_path = os.path.join(tmp_dir, f"ref_colors_level{level}")
    if os.path.exists(ref_colors_path):
        ref_colors = []
        for i in range(6):
            ref_colors.append(Image.open(os.path.join(ref_colors_path, f"{i}.png")))
    else:
        os.makedirs(ref_colors_path, exist_ok=True)

    colors, normals = _load_multiview_images(mv_dir, level, ref_colors, ref_mask)

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
        mesh_v, mesh_f = _decimate_mesh(
            np.array(mesh_v, dtype="float32"),
            np.array(mesh_f),
            MAX_INPUT_FACES,
        )

    mesh_v_t = torch.tensor(np.array(mesh_v, dtype="float32"), device="cuda")
    mesh_f_t = torch.tensor(np.array(mesh_f), device="cuda")
    sys.stderr.write(
        f"[refine_level] V={mesh_v_t.shape[0]}, F={mesh_f_t.shape[0]}\n"
    )
    sys.stderr.flush()

    fixed_v_path = os.path.join(tmp_dir, "fixed_v.npy")
    fixed_f_path = os.path.join(tmp_dir, "fixed_f.npy")
    fixed_v, fixed_f = None, None
    if os.path.exists(fixed_v_path):
        fixed_v = torch.from_numpy(np.load(fixed_v_path))
        fixed_f = torch.from_numpy(np.load(fixed_f_path)).long()

    no_decompose = params.get("no_decompose", False)

    from refine.mesh_refine import set_debug_dir
    debug_dir = params.get("debug_dir")
    if debug_dir:
        set_debug_dir(debug_dir)

    use_poisson = normal_flip is not None

    if use_poisson:
        sys.stderr.write(f"[refine_level] level={level} geo_refine_poisson (ERA3D mode)\n")
        sys.stderr.flush()

        from refine.func import ERA3D_VIEWS
        from refine.mesh_refine import geo_refine_poisson

        new_meshes, simp_v, simp_f = geo_refine_poisson(
            mesh_v_t, mesh_f_t,
            list(colors), list(normals),
            fixed_v=fixed_v, fixed_f=fixed_f,
            distract_mask=distract_mask,
            camera_indices=ERA3D_VIEWS.camera_indices,
            azim_list=ERA3D_VIEWS.azim_list,
            normal_flip=normal_flip,
        )
    else:
        sys.stderr.write(f"[refine_level] level={level} geo_refine (no_decompose={no_decompose})\n")
        sys.stderr.flush()

        new_meshes, simp_v, simp_f = geo_refine(
            mesh_v_t, mesh_f_t,
            list(colors), list(normals),
            expansion_weight=0.0 if no_decompose else 0.1,
            fixed_v=fixed_v, fixed_f=fixed_f,
            distract_mask=distract_mask, distract_bbox=distract_bbox,
            no_decompose=no_decompose,
            normal_flip=normal_flip,
        )

    if fixed_v is None:
        np.save(fixed_v_path, simp_v.cpu().numpy())
        np.save(fixed_f_path, simp_f.cpu().numpy())
    else:
        combined_v = np.concatenate([fixed_v.numpy(), simp_v.cpu().numpy()], axis=0)
        combined_f = np.concatenate([
            fixed_f.numpy(),
            simp_f.cpu().numpy() + fixed_v.shape[0],
        ], axis=0)
        np.save(fixed_v_path, combined_v)
        np.save(fixed_f_path, combined_f)
    del fixed_v, fixed_f, simp_v, simp_f

    refined_dir = os.path.join(tmp_dir, "refined")
    os.makedirs(refined_dir, exist_ok=True)
    output_path = os.path.join(refined_dir, f"out_{level}.glb")
    save_py3dmesh_with_trimesh_fast(new_meshes, output_path, apply_sRGB_to_LinearRGB=True)

    vertex_count = int(new_meshes.verts_packed().shape[0])
    face_count = int(new_meshes.faces_packed().shape[0])
    del new_meshes
    gc.collect()
    torch.cuda.empty_cache()

    return {"vertex_count": vertex_count, "face_count": face_count}


STAGES = {
    "reconstruct_stage1": _run_reconstruct_stage1,
    "run_mesh_refine": _run_mesh_refine,
    "color_projection": _run_color_projection,
    "refine_level": _run_refine_level,
    "generate_distract_mask": _run_generate_distract_mask,
    "generate_levels": _run_generate_levels,
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
