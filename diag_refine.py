"""
Stage 3 refine の処理を1ステップずつ実行し、
各処理の入出力テンソルのshape・dtype・値の範囲を記録する診断スクリプト。
"""
import os
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.expanduser("~/.cache/torch/inductor"))

import gc
import time
import json
import torch
import trimesh
import numpy as np
from PIL import Image
import cv2

from vram_monitor import init_log, log_vram, set_stage
from refine.mesh_refine import simple_remove, erode_alpha, init_target, reconstruct_stage1
from refine.func import (
    STDGEN_VIEWS,
    make_star_cameras_orthographic, to_py3d_mesh, get_cameras_list,
    multiview_color_projection, from_py3d_mesh, get_visible_faces, project_color,
)
from refine.render import NormalsRenderer, calc_vertex_normals
from refine.opt import MeshOptimizer

init_log()

LOG_PATH = os.path.join(os.path.dirname(__file__), "log", "diag_refine.jsonl")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)


def diag(label, **tensors):
    free, total = torch.cuda.mem_get_info(0)
    alloc = torch.cuda.memory_allocated(0)
    reserved = torch.cuda.memory_reserved(0)

    record = {
        "ts": time.strftime("%H:%M:%S"),
        "label": label,
        "alloc_mb": round(alloc / 1e6, 1),
        "reserved_mb": round(reserved / 1e6, 1),
        "free_mb": round(free / 1e6, 1),
    }

    for name, t in tensors.items():
        if isinstance(t, torch.Tensor):
            record[name] = {
                "shape": list(t.shape),
                "dtype": str(t.dtype),
                "device": str(t.device),
                "min": float(t.min()) if t.numel() > 0 else None,
                "max": float(t.max()) if t.numel() > 0 else None,
                "has_nan": bool(t.isnan().any()),
                "has_inf": bool(t.isinf().any()),
            }
        elif t is not None:
            record[name] = str(type(t))

    line = json.dumps(record, default=str)
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")
        f.flush()


def load_mesh_and_images():
    mesh = trimesh.load("/tmp/stdgen_pipeline/slrm/girl_3.obj")
    new_mesh = mesh.split(only_watertight=False)
    new_mesh = [j for j in new_mesh if len(j.vertices) >= 300]
    mesh = trimesh.Scene(new_mesh).to_geometry()
    mesh_v, mesh_f = mesh.vertices, mesh.faces
    diag("mesh_loaded", verts=torch.tensor(mesh_v), faces=torch.tensor(mesh_f))

    colors, normals_img = [], []
    for i in range(6):
        color = cv2.imread(f"/tmp/stdgen_pipeline/multiview/girl/level2/color_{i}.png")[..., ::-1]
        normal = cv2.imread(f"/tmp/stdgen_pipeline/multiview/girl/level2/normal_{i}.png")[..., ::-1]
        colors.append(Image.fromarray(color))
        normals_img.append(Image.fromarray(normal))

    return mesh_v, mesh_f, colors, normals_img


def test_reconstruct_stage1(mesh_v, mesh_f, rm_normals):
    diag("stage1_start")
    vertices = torch.tensor(mesh_v, device="cuda", dtype=torch.float32)
    faces = torch.tensor(mesh_f, device="cuda")

    mv, proj = make_star_cameras_orthographic(8, 1, r=1.2)
    mv = mv[STDGEN_VIEWS.camera_indices]
    renderer = NormalsRenderer(mv, proj, list(rm_normals[0].size))
    target_images = init_target(rm_normals, new_bkgd=(0., 0., 0.))
    diag("stage1_target_images", target=target_images)

    opt = MeshOptimizer(vertices, faces, local_edgelen=False, gain=0.05,
                        edge_len_lims=(0.005, 0.02), lr=0.08, remesh_interval=1, remesh_start=0)

    for i in range(3):
        set_stage("stage1", step=i)
        opt.zero_grad()
        _v, _f = opt.vertices, opt.faces
        normals = calc_vertex_normals(_v, _f)
        normals[:, 0] *= -1
        normals[:, 2] *= -1
        images = renderer.render(_v, normals, _f)
        diag(f"stage1_step{i}_render", images=images, verts=_v, faces=_f, normals=normals)

        mask = target_images[..., -1] < 0.5
        t_mask = images[..., -1] > 0.5
        loss = (images[t_mask] - target_images[t_mask]).abs().pow(2).mean()
        loss.backward()
        opt.step()
        if i % 1 == 0:
            _v, _f = opt.remesh(poisson=False)
        diag(f"stage1_step{i}_done", loss=torch.tensor(loss.item()))

    vertices, faces = opt._vertices.detach(), opt._faces.detach()
    diag("stage1_done", verts=vertices, faces=faces)
    return vertices, faces


def test_run_mesh_refine(vertices, faces, rm_normals):
    diag("stage2_start")
    gc.collect()
    torch.cuda.empty_cache()
    log_vram("stage2_after_cache_clear")

    mv, proj = make_star_cameras_orthographic(8, 1, r=1.2)
    mv = mv[STDGEN_VIEWS.camera_indices]
    renderer = NormalsRenderer(mv, proj, list(rm_normals[0].size))
    target_images = init_target(rm_normals, new_bkgd=(0., 0., 0.))

    opt = MeshOptimizer(vertices, faces, ramp=5, edge_len_lims=(0.001, 0.005),
                        local_edgelen=False, laplacian_weight=0.02)

    _vertices = opt.vertices
    _faces = opt.faces

    for i in range(3):
        set_stage("stage2", step=i)
        torch.cuda.empty_cache()
        diag(f"stage2_step{i}_begin")

        opt.zero_grad()
        normals = calc_vertex_normals(_vertices, _faces)
        diag(f"stage2_step{i}_normals", normals=normals, verts=_vertices, faces=_faces)

        images = renderer.render(_vertices, normals, _faces)
        diag(f"stage2_step{i}_render", images=images)

        if i < 5:
            diag(f"stage2_step{i}_mvp_start")
            torch.cuda.empty_cache()

            with torch.no_grad():
                py3d_mesh = to_py3d_mesh(_vertices, _faces, normals)
                diag(f"stage2_step{i}_py3d_mesh",
                     py3d_verts=py3d_mesh.verts_packed(),
                     py3d_faces=py3d_mesh.faces_packed())

                cameras = get_cameras_list(
                    azim_list=STDGEN_VIEWS.azim_list,
                    device=_vertices.device, focal=1/1.2,
                )
                diag(f"stage2_step{i}_cameras_ready")

                for view_idx, (cam, pil, w) in enumerate(zip(cameras, rm_normals, [2, 0.8, 0.8, 2, 0.8, 0.8])):
                    diag(f"stage2_step{i}_view{view_idx}_start")

                    visible = get_visible_faces(py3d_mesh, cam, resolution=1024)
                    diag(f"stage2_step{i}_view{view_idx}_visible", visible_faces=visible)

                    ret = project_color(py3d_mesh, cam, pil, eps=0.05, resolution=1024, device="cuda", use_alpha=True)
                    diag(f"stage2_step{i}_view{view_idx}_projected",
                         valid_verts=ret["valid_verts"],
                         valid_colors=ret["valid_colors"],
                         cos_angles=ret["cos_angles"])

                diag(f"stage2_step{i}_mvp_done")

                _, _, target_normal = from_py3d_mesh(
                    multiview_color_projection(
                        py3d_mesh, rm_normals, cameras_list=cameras,
                        weights=[2, 0.8, 0.8, 2, 0.8, 0.8],
                        confidence_threshold=0.1, complete_unseen=False,
                        below_confidence_strategy="original",
                        reweight_with_cosangle="linear",
                    )
                )
                diag(f"stage2_step{i}_target_normal", target_normal=target_normal)

                target_normal = target_normal * 2 - 1
                target_normal = torch.nn.functional.normalize(target_normal, dim=-1)
                target_normal[:, 0] *= -1
                target_normal[:, 2] *= -1

                debug_images = renderer.render(_vertices, target_normal, _faces)
                diag(f"stage2_step{i}_debug_render", debug_images=debug_images)

        d_mask = images[..., -1] > 0.5
        mask = target_images[..., -1] < 0.5
        loss_debug = (images[..., :3][d_mask] - debug_images[..., :3][d_mask]).pow(2).mean()
        loss_alpha = (images[..., -1][mask] - target_images[..., -1][mask]).pow(2).mean()
        loss = loss_debug + loss_alpha
        diag(f"stage2_step{i}_loss", loss=torch.tensor(loss.item()), loss_debug=torch.tensor(loss_debug.item()))

        loss.backward()
        opt.step()

        if i % 5 == 0:
            _vertices, _faces = opt.remesh(poisson=False)

        diag(f"stage2_step{i}_done")

    diag("stage2_complete", final_verts=opt._vertices.detach(), final_faces=opt._faces.detach())


if __name__ == "__main__":
    with open(LOG_PATH, "w") as f:
        f.write("")

    mesh_v, mesh_f, colors, normals_img = load_mesh_and_images()

    rm_normals = simple_remove(normals_img)
    for idx, img in enumerate(rm_normals):
        colors[idx] = Image.fromarray(
            np.concatenate([np.array(colors[idx])[..., :3], np.array(img)[:, :, 3:4]], axis=-1)
        )
    colors = erode_alpha(colors)

    log_vram("diag_start")

    vertices, faces = test_reconstruct_stage1(mesh_v, mesh_f, rm_normals)

    test_run_mesh_refine(vertices, faces, rm_normals)

    diag("all_done")
    log_vram("diag_end")
