import gc
import sys
import time

import torch
import numpy as np
import trimesh
from PIL import Image
from typing import List
from tqdm import tqdm
from sklearn.neighbors import KDTree


def _log(msg: str):
    sys.stderr.write(f"[refine] {msg}\n")
    sys.stderr.flush()


_DEBUG_DIR = None
_JSONL_FILE = None


def set_debug_dir(path: str):
    global _DEBUG_DIR, _JSONL_FILE
    _DEBUG_DIR = path
    if path:
        import os
        os.makedirs(path, exist_ok=True)
        _JSONL_FILE = open(os.path.join(path, "refine_diagnostics.jsonl"), "w")


def _write_jsonl(data: dict):
    if _JSONL_FILE is None:
        return
    import json
    _JSONL_FILE.write(json.dumps(data, ensure_ascii=False, default=str) + "\n")
    _JSONL_FILE.flush()


def _save_normal_comparison(rendered: torch.Tensor, target: torch.Tensor, name: str):
    if _DEBUG_DIR is None:
        return
    import os
    from PIL import Image

    os.makedirs(_DEBUG_DIR, exist_ok=True)
    n_views = min(rendered.shape[0], target.shape[0])

    for v in range(n_views):
        rend_np = (rendered[v, :, :, :3].detach().cpu().clamp(0, 1) * 255).byte().numpy()
        targ_np = (target[v, :, :, :3].detach().cpu().clamp(0, 1) * 255).byte().numpy()

        h, w = rend_np.shape[:2]
        combined = np.concatenate([targ_np, rend_np], axis=1)
        path = os.path.join(_DEBUG_DIR, f"debug_{name}_view{v}.png")
        Image.fromarray(combined).save(path)

    _log(f"debug: saved {name} comparison ({n_views} views) to {_DEBUG_DIR}")


def _log_normal_diagnostics(rendered: torch.Tensor, target: torch.Tensor, name: str):
    n_views = min(rendered.shape[0], target.shape[0])
    for v in range(n_views):
        r_alpha = rendered[v, :, :, 3].detach()
        t_alpha = target[v, :, :, 3].detach()

        r_fg = r_alpha > 0.5
        t_fg = t_alpha > 0.5
        intersection = (r_fg & t_fg).sum().item()
        union = (r_fg | t_fg).sum().item()
        iou = intersection / max(union, 1)

        overlap = r_fg & t_fg
        if overlap.sum() < 10:
            _log(f"  view{v}: iou={iou:.3f} overlap_pixels={overlap.sum().item()} (too few)")
            _write_jsonl({"event": "normal_diag", "name": name, "view": v, "iou": iou, "overlap": overlap.sum().item()})
            continue

        r_rgb = rendered[v, :, :, :3].detach()[overlap]
        t_rgb = target[v, :, :, :3].detach()[overlap]
        r_n = r_rgb * 2 - 1
        t_n = t_rgb * 2 - 1
        cos_sim = (r_n * t_n).sum(dim=-1).mean().item()

        color_diff = (r_rgb - t_rgb).abs().mean(dim=0)
        r_mean = r_rgb.mean(dim=0)
        t_mean = t_rgb.mean(dim=0)

        diag = {
            "event": "normal_diag", "name": name, "view": v,
            "iou": round(iou, 4), "cos_sim": round(cos_sim, 4),
            "color_diff": [round(color_diff[i].item(), 4) for i in range(3)],
            "rend_mean": [round(r_mean[i].item(), 4) for i in range(3)],
            "targ_mean": [round(t_mean[i].item(), 4) for i in range(3)],
        }
        _write_jsonl(diag)

        _log(f"  view{v}: iou={iou:.3f} cos_sim={cos_sim:.3f} "
             f"color_diff=({color_diff[0]:.3f},{color_diff[1]:.3f},{color_diff[2]:.3f}) "
             f"rend_mean=({r_mean[0]:.3f},{r_mean[1]:.3f},{r_mean[2]:.3f}) "
             f"targ_mean=({t_mean[0]:.3f},{t_mean[1]:.3f},{t_mean[2]:.3f})")

    _log(f"  diagnostics done for {name}")


def _save_debug_glb(vertices, faces, name: str):
    if _DEBUG_DIR is None:
        return
    import os
    os.makedirs(_DEBUG_DIR, exist_ok=True)
    v = vertices.detach().cpu().numpy() if torch.is_tensor(vertices) else vertices
    f = faces.detach().cpu().numpy() if torch.is_tensor(faces) else faces
    mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
    path = os.path.join(_DEBUG_DIR, f"debug_{name}.glb")
    mesh.export(path, file_type="glb")
    _log(f"debug: saved {path} ({os.path.getsize(path)/1024:.0f} KB)")

from refine.func import STDGEN_VIEWS, _find_view_config_by_azim, from_py3d_mesh, get_cameras_list, make_star_cameras_orthographic, multiview_color_projection, simple_clean_mesh, to_py3d_mesh, to_pyml_mesh
from refine.opt import MeshOptimizer
from refine.render import NormalsRenderer, calc_vertex_normals, get_shared_glctx

import pytorch3d
from pytorch3d.structures import Meshes


def remove_color(arr):
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    # calc diffs
    base = arr[0, 0]
    diffs = np.abs(arr.astype(np.int32) - base.astype(np.int32)).sum(axis=-1)
    alpha = (diffs <= 80)
    
    arr[alpha] = 255
    alpha = ~alpha
    arr = np.concatenate([arr, alpha[..., None].astype(np.int32) * 255], axis=-1)
    return arr


def simple_remove(imgs):
    """Only works for normal"""
    if not isinstance(imgs, list):
        imgs = [imgs]
        single_input = True
    else:
        single_input = False
    rets = []
    for img in imgs:
        arr = np.array(img)
        arr = remove_color(arr)
        rets.append(Image.fromarray(arr.astype(np.uint8)))
    if single_input:
        return rets[0]
    return rets


def erode_alpha(img_list):
    out_img_list = []
    for idx, img in enumerate(img_list):
        arr = np.array(img)
        alpha = (arr[:, :, 3] > 127).astype(np.uint8)
        # erode 1px
        import cv2
        alpha = cv2.erode(alpha, np.ones((3, 3), np.uint8), iterations=1)
        alpha = (alpha * 255).astype(np.uint8)
        img = Image.fromarray(np.concatenate([arr[:, :, :3], alpha[:, :, None]], axis=-1))
        out_img_list.append(img)
    return out_img_list


def merge_small_faces(mesh, thres=1e-5):
    area_faces = mesh.area_faces
    small_faces = area_faces < thres

    vertices = mesh.vertices
    faces = mesh.faces

    new_vertices = vertices.tolist()
    vertex_mapping = {}
    
    for face_idx in np.where(small_faces)[0]:
        face = faces[face_idx]
        v1, v2, v3 = face
        center = np.mean(vertices[face], axis=0)

        new_vertex_idx = len(new_vertices)
        new_vertices.append(center)

        vertex_mapping[v1] = new_vertex_idx
        vertex_mapping[v2] = new_vertex_idx
        vertex_mapping[v3] = new_vertex_idx

    for k,v in vertex_mapping.items():
        faces[faces == k] = v

    faces = faces[~small_faces]

    new_mesh = trimesh.Trimesh(vertices=new_vertices, faces=faces, postprocess=False)
    new_mesh.remove_unreferenced_vertices()
    new_mesh.update_faces(new_mesh.nondegenerate_faces())
    new_mesh.update_faces(new_mesh.unique_faces())
    
    return new_mesh


def init_target(img_pils, new_bkgd=(0., 0., 0.), device="cuda"):
    # Convert the background color to a PyTorch tensor
    new_bkgd = torch.tensor(new_bkgd, dtype=torch.float32).view(1, 1, 3).to(device)
    
    # Convert all images to PyTorch tensors and process them
    imgs = torch.stack([torch.from_numpy(np.array(img, dtype=np.float32)) for img in img_pils]).to(device) / 255
    img_nps = imgs[..., :3]
    alpha_nps = imgs[..., 3]
    ori_bkgds = img_nps[:, :1, :1]
    
    # Avoid divide by zero and calculate the original image
    alpha_nps_clamp = torch.clamp(alpha_nps, 1e-6, 1)
    ori_img_nps = (img_nps - ori_bkgds * (1 - alpha_nps.unsqueeze(-1))) / alpha_nps_clamp.unsqueeze(-1)
    ori_img_nps = torch.clamp(ori_img_nps, 0, 1)
    img_nps = torch.where(alpha_nps.unsqueeze(-1) > 0.05, ori_img_nps * alpha_nps.unsqueeze(-1) + new_bkgd * (1 - alpha_nps.unsqueeze(-1)), new_bkgd)

    rgba_img_np = torch.cat([img_nps, alpha_nps.unsqueeze(-1)], dim=-1)
    return rgba_img_np


def reconstruct_stage1(pils: List[Image.Image], steps=100, vertices=None, faces=None, fixed_v=None, fixed_f=None, lr=0.03, start_edge_len=0.15, end_edge_len=0.005,
                       decay=0.995, loss_expansion_weight=0.1, gain=0.1, remesh_interval=1, remesh_start=0, distract_mask=None, distract_bbox=None,
                       camera_indices=None, normal_flip=None):
    vertices, faces = vertices.cuda(), faces.cuda()
    assert len(pils) == 6
    mv, proj = make_star_cameras_orthographic(8, 1, r=1.2)
    mv = mv[camera_indices or STDGEN_VIEWS.camera_indices]

    render_size = list(pils[0].size)
    renderer = NormalsRenderer(mv, proj, render_size)

    target_images = init_target(pils, new_bkgd=(0., 0., 0.))

    opt = MeshOptimizer(vertices, faces, local_edgelen=False, gain=gain, edge_len_lims=(end_edge_len, start_edge_len), lr=lr,
                        remesh_interval=remesh_interval, remesh_start=remesh_start)

    _vertices = opt.vertices
    _faces = opt.faces

    has_fixed = fixed_v is not None and fixed_f is not None
    fixed_v_cpu = fixed_v.cpu() if has_fixed else None
    fixed_f_cpu = fixed_f.cpu() if has_fixed else None
    if has_fixed:
        del fixed_v, fixed_f
        kdtree = KDTree(fixed_v_cpu.numpy())

    mask = target_images[..., -1] < 0.5

    _log(f"stage1: start steps={steps} lr={lr:.4f} remesh_interval={remesh_interval} "
         f"edge_len=[{end_edge_len},{start_edge_len}] verts={len(vertices)} faces={len(faces)}")
    t0 = time.monotonic()

    for i in tqdm(range(steps)):
        if has_fixed:
            with torch.no_grad():
                fv = fixed_v_cpu.cuda()
                ff = fixed_f_cpu.cuda()
            faces = torch.cat([_faces, ff + len(_vertices)], dim=0)
            vertices = torch.cat([_vertices, fv.detach()], dim=0)
        else:
            faces, vertices = _faces, _vertices

        opt.zero_grad()
        opt._lr *= decay
        normals = calc_vertex_normals(vertices, faces)

        nf = normal_flip if normal_flip is not None else [-1, 1, -1]
        normals[:, 0] *= nf[0]
        normals[:, 1] *= nf[1]
        normals[:, 2] *= nf[2]

        images = renderer.render(vertices, normals, faces)
        loss_expand = 0.5 * ((vertices+normals).detach() - vertices).pow(2).mean()

        if has_fixed:
            del fv, ff

        t_mask = images[..., -1] > 0.5
        loss_target_l2 = (images[t_mask] - target_images[t_mask]).abs().pow(2).mean()
        loss_alpha_target_mask_l2 = (images[..., -1][mask] - target_images[..., -1][mask]).pow(2).mean()

        loss = loss_target_l2 + loss_alpha_target_mask_l2 + loss_expand * loss_expansion_weight

        if i % 50 == 0 or i == steps - 1:
            _log(f"stage1: step={i}/{steps} loss={loss.item():.4f} "
                 f"rgb_l2={loss_target_l2.item():.4f} alpha_l2={loss_alpha_target_mask_l2.item():.4f} "
                 f"expand={loss_expand.item():.4f} verts={len(_vertices)} faces={len(_faces)} "
                 f"lr={opt._lr:.6f}")
            _write_jsonl({
                "event": "stage1_loss", "step": i, "steps": steps,
                "loss": round(loss.item(), 6),
                "rgb_l2": round(loss_target_l2.item(), 6),
                "alpha_l2": round(loss_alpha_target_mask_l2.item(), 6),
                "expand": round(loss_expand.item(), 6),
                "verts": len(_vertices), "faces": len(_faces),
                "lr": round(opt._lr, 8),
            })

        if i == 0:
            _log(f"stage1: step0 diagnostics (per-view normal match):")
            _log_normal_diagnostics(images, target_images, "stage1_step0")
            if _DEBUG_DIR is not None:
                _save_normal_comparison(images, target_images, "stage1_step0")

        if distract_mask is not None:
            hair_visible_normals = normals
            hair_visible_normals[len(_vertices):] = -1.
            _images = renderer.render(vertices, hair_visible_normals, faces)
            loss_distract = (_images[0][distract_mask] - target_images[0][distract_mask]).pow(2).mean()

            target_outside = target_images[0][..., :3].clone()
            target_outside[~distract_mask] = 0.

            loss_outside_distract = (_images[0][..., :3][~distract_mask] - target_outside[..., :3][~distract_mask]).pow(2).mean()

            loss = loss + loss_distract * 1. + loss_outside_distract * 10.
            del _images, target_outside

        if has_fixed:
            _, idx = kdtree.query(_vertices.detach().cpu().numpy(), k=1)
            idx = idx.squeeze()

            with torch.no_grad():
                fv_anchor = fixed_v_cpu.cuda()
                ff_anchor = fixed_f_cpu.cuda()
            anchors = fv_anchor[idx].detach()
            normals_fixed = calc_vertex_normals(fv_anchor, ff_anchor)
            loss_anchor = (torch.clamp(((anchors - _vertices) * normals_fixed[idx]).sum(-1), min=-0)+0).pow(3)
            loss_anchor_dist_mask = (anchors - _vertices).norm(dim=-1) < 0.05
            loss_anchor = loss_anchor[loss_anchor_dist_mask].mean()
            del fv_anchor, ff_anchor, anchors, normals_fixed

            loss = loss + loss_anchor * 100.

        loss_oob = (vertices.abs() > 0.99).float().mean() * 10
        loss = loss + loss_oob

        loss.backward()
        opt.step()

        del loss, loss_expand, loss_target_l2, loss_alpha_target_mask_l2, loss_oob
        del images, normals, t_mask

        if i % remesh_interval == 0 and i >= remesh_start:
            _vertices,_faces = opt.remesh(poisson=False)
            gc.collect()
            torch.cuda.empty_cache()

    vertices, faces = opt._vertices.detach(), opt._faces.detach()

    with torch.no_grad():
        final_normals = calc_vertex_normals(vertices, faces)
        nf = normal_flip if normal_flip is not None else [-1, 1, -1]
        final_normals[:, 0] *= nf[0]
        final_normals[:, 1] *= nf[1]
        final_normals[:, 2] *= nf[2]
        final_images = renderer.render(vertices, final_normals, faces)
        _log(f"stage1: final diagnostics:")
        _log_normal_diagnostics(final_images, target_images, "stage1_final")
        if _DEBUG_DIR is not None:
            _save_normal_comparison(final_images, target_images, "stage1_final")
        del final_normals, final_images

    _log(f"stage1: done in {time.monotonic()-t0:.1f}s verts={len(vertices)} faces={len(faces)}")

    return vertices, faces


def run_mesh_refine(vertices, faces, pils: List[Image.Image], fixed_v=None, fixed_f=None, steps=100, start_edge_len=0.02, end_edge_len=0.005,
                    decay=0.99, update_normal_interval=10, update_warmup=10, return_mesh=True, process_inputs=True, process_outputs=True, remesh_interval=20,
                    camera_indices=None, azim_list=None, normal_flip=None):
    poission_steps = []

    assert len(pils) == 6
    mv, proj = make_star_cameras_orthographic(8, 1, r=1.2)
    mv = mv[camera_indices or STDGEN_VIEWS.camera_indices]

    render_size = list(pils[0].size)
    renderer = NormalsRenderer(mv, proj, render_size)

    target_images = init_target(pils, new_bkgd=(0., 0., 0.))

    opt = MeshOptimizer(vertices, faces, ramp=5, edge_len_lims=(end_edge_len, start_edge_len), local_edgelen=False, laplacian_weight=0.02)

    _vertices = opt.vertices
    _faces = opt.faces
    alpha_init = None

    has_fixed = fixed_v is not None and fixed_f is not None
    fixed_v_cpu = fixed_v.cpu() if has_fixed else None
    fixed_f_cpu = fixed_f.cpu() if has_fixed else None
    if has_fixed:
        del fixed_v, fixed_f

    mask = target_images[..., -1] < 0.5
    debug_images = None

    _log(f"stage2: start steps={steps} laplacian_weight={opt._laplacian_weight} "
         f"remesh_interval={remesh_interval} edge_len=[{end_edge_len},{start_edge_len}] "
         f"verts={len(vertices)} faces={len(faces)}")
    t0 = time.monotonic()

    for i in tqdm(range(steps)):
        if has_fixed:
            with torch.no_grad():
                fv = fixed_v_cpu.cuda()
                ff = fixed_f_cpu.cuda()
            faces = torch.cat([_faces, ff + len(_vertices)], dim=0)
            vertices = torch.cat([_vertices, fv.detach()], dim=0)
        else:
            faces, vertices = _faces, _vertices

        opt.zero_grad()
        opt._lr *= decay
        normals = calc_vertex_normals(vertices, faces)
        images = renderer.render(vertices, normals, faces)
        if alpha_init is None:
            alpha_init = images.detach()

        if has_fixed:
            del fv, ff

        if i < update_warmup or i % update_normal_interval == 0:
            with torch.no_grad():
                py3d_mesh = to_py3d_mesh(vertices, faces, normals)
                _azim = azim_list or STDGEN_VIEWS.azim_list
                cameras = get_cameras_list(azim_list=_azim, device=vertices.device, focal=1/1.2)
                projected = multiview_color_projection(py3d_mesh, pils, cameras_list=cameras, weights=[2,0.8,0.8,2,0.8,0.8], confidence_threshold=0.1, complete_unseen=False, below_confidence_strategy='original', reweight_with_cosangle='linear')
                _, _, target_normal = from_py3d_mesh(projected)
                del projected, py3d_mesh, cameras
                gc.collect()
                torch.cuda.empty_cache()

                target_normal = target_normal * 2 - 1
                target_normal = torch.nn.functional.normalize(target_normal, dim=-1)

                nf = normal_flip if normal_flip is not None else [-1, 1, -1]
                target_normal[:, 0] *= nf[0]
                target_normal[:, 1] *= nf[1]
                target_normal[:, 2] *= nf[2]

                del debug_images
                debug_images = renderer.render(vertices, target_normal, faces)
                del target_normal

        d_mask = images[..., -1] > 0.5
        loss_debug_l2 = (images[..., :3][d_mask] - debug_images[..., :3][d_mask]).pow(2).mean()

        loss_alpha_target_mask_l2 = (images[..., -1][mask] - target_images[..., -1][mask]).pow(2).mean()

        loss = loss_debug_l2 + loss_alpha_target_mask_l2

        loss_oob = (vertices.abs() > 0.99).float().mean() * 10
        loss = loss + loss_oob

        if i % 20 == 0 or i == steps - 1:
            _log(f"stage2: step={i}/{steps} loss={loss.item():.4f} "
                 f"normal_l2={loss_debug_l2.item():.4f} alpha_l2={loss_alpha_target_mask_l2.item():.4f} "
                 f"oob={loss_oob.item():.4f} verts={len(_vertices)} faces={len(_faces)} "
                 f"lr={opt._lr:.6f}")
            _write_jsonl({
                "event": "stage2_loss", "step": i, "steps": steps,
                "loss": round(loss.item(), 6),
                "normal_l2": round(loss_debug_l2.item(), 6),
                "alpha_l2": round(loss_alpha_target_mask_l2.item(), 6),
                "oob": round(loss_oob.item(), 6),
                "verts": len(_vertices), "faces": len(_faces),
                "lr": round(opt._lr, 8),
            })

        loss.backward()
        opt.step()

        del loss, loss_debug_l2, loss_alpha_target_mask_l2, loss_oob
        del images, normals, d_mask

        if i % remesh_interval == 0:
            _vertices,_faces = opt.remesh(poisson=(i in poission_steps))
            gc.collect()
            torch.cuda.empty_cache()

    vertices, faces = opt._vertices.detach(), opt._faces.detach()
    _log(f"stage2: done in {time.monotonic()-t0:.1f}s verts={len(vertices)} faces={len(faces)}")

    if process_outputs:
        vertices = vertices / 2 * 1.35
        vertices[..., [0, 2]] = - vertices[..., [0, 2]]

    return vertices, faces


def _sample_vertex_normals_from_views(vertices, faces, normal_maps, camera_indices, normal_flip):
    import nvdiffrast.torch as dr
    import torch.nn.functional as F

    nf = torch.tensor(normal_flip or [1, 1, -1], dtype=torch.float32, device="cuda")
    V = vertices.shape[0]

    mv, proj = make_star_cameras_orthographic(8, 1, r=1.2)
    from refine.func import ERA3D_VIEWS
    mv = mv[camera_indices or ERA3D_VIEWS.camera_indices]
    mvp = proj @ mv

    glctx = get_shared_glctx("cuda")
    vert_hom = torch.cat([vertices, torch.ones(V, 1, device="cuda")], dim=-1)
    clips = torch.stack([vert_hom @ mvp[c].T for c in range(6)])
    rast_out, _ = dr.rasterize(glctx, clips, faces.int(), resolution=[512, 512], grad_db=False)

    target_sum = torch.zeros(V, 3, device="cuda")
    weight_sum = torch.zeros(V, device="cuda")

    for vi in range(6):
        face_ids = rast_out[vi, :, :, 3].long()
        alpha = normal_maps[vi, :, :, 3]
        valid = (face_ids > 0) & (alpha > 0.5)
        if not valid.any():
            continue

        valid_face_ids = face_ids[valid] - 1
        bary = rast_out[vi, :, :, :2][valid]
        world_normal = (normal_maps[vi, :, :, :3][valid] * 2 - 1) * nf

        u, bv = bary[:, 0], bary[:, 1]
        w0 = 1 - u - bv
        fv = faces[valid_face_ids].long()

        for vert_ids, bary_w in [(fv[:, 0], w0), (fv[:, 1], u), (fv[:, 2], bv)]:
            target_sum.scatter_add_(0, vert_ids.unsqueeze(-1).expand(-1, 3),
                                    world_normal * bary_w.unsqueeze(-1))
            weight_sum.scatter_add_(0, vert_ids, bary_w.abs())

    has_data = weight_sum > 1e-6
    target = torch.zeros(V, 3, device="cuda")
    target[has_data] = target_sum[has_data] / weight_sum[has_data].unsqueeze(-1)
    target = F.normalize(target, dim=-1, eps=1e-6)

    _log(f"poisson: vertex coverage {has_data.sum().item()}/{V} ({100*has_data.float().mean():.1f}%)")
    return target, has_data


def _poisson_vertex_update(vertices, faces, target_vertex_normals, has_target,
                           outer_iters=1, inner_iters=40, sigma=0.5):
    import torch.nn.functional as F

    v = vertices.clone()
    fl = faces.long()
    V = v.shape[0]

    target_fn = F.normalize(
        (target_vertex_normals[fl[:, 0]] + target_vertex_normals[fl[:, 1]] + target_vertex_normals[fl[:, 2]]) / 3,
        dim=-1, eps=1e-6,
    )

    has_f = (has_target[fl[:, 0]].float() + has_target[fl[:, 1]].float() + has_target[fl[:, 2]].float()) / 3.0

    for outer in range(outer_iters):
        for _ in range(inner_iters):
            v0, v1, v2 = v[fl[:, 0]], v[fl[:, 1]], v[fl[:, 2]]
            current_fn = F.normalize(torch.cross(v1 - v0, v2 - v0, dim=1), dim=-1, eps=1e-6)

            per_face_sigma = sigma * has_f
            filtered_fn = F.normalize(
                (1 - per_face_sigma.unsqueeze(-1)) * current_fn + per_face_sigma.unsqueeze(-1) * target_fn,
                dim=-1, eps=1e-6,
            )

            centroids = (v0 + v1 + v2) / 3.0
            delta = torch.zeros_like(v)
            count = torch.zeros(V, 1, device=v.device)
            F_count = fl.shape[0]

            for corner in range(3):
                vert_ids = fl[:, corner]
                diff = centroids - v[vert_ids]
                proj = (diff * filtered_fn).sum(dim=-1, keepdim=True) * filtered_fn
                delta.scatter_add_(0, vert_ids.unsqueeze(-1).expand(-1, 3), proj)
                count.scatter_add_(0, vert_ids.unsqueeze(-1),
                                   torch.ones(F_count, 1, device=v.device))

            count = count.clamp(min=1)
            v = v + delta / count

        with torch.no_grad():
            vn = calc_vertex_normals(v, faces)
            cos = (vn[has_target] * target_vertex_normals[has_target]).sum(dim=-1).mean()
            disp = (v - vertices).norm(dim=-1).mean()
        _log(f"poisson: outer {outer} cos_sim={cos.item():.4f} mean_disp={disp.item():.6f}")

    return v


def geo_refine_poisson(mesh_v, mesh_f, rgb_ls, normal_ls, fixed_v=None, fixed_f=None,
                       distract_mask=None, thres=3e-6,
                       camera_indices=None, azim_list=None, normal_flip=None):
    rm_normals = simple_remove(normal_ls)

    for idx, img in enumerate(rm_normals):
        rgb_ls[idx] = Image.fromarray(np.concatenate([
            np.array(rgb_ls[idx])[..., :3],
            np.array(img)[:, :, 3:4],
        ], axis=-1))
    rgb_ls = erode_alpha(rgb_ls)

    has_fixed = fixed_v is not None and fixed_f is not None
    fixed_v_cpu = fixed_v.cpu() if has_fixed else None
    fixed_f_cpu = fixed_f.cpu() if has_fixed else None
    if has_fixed:
        del fixed_v, fixed_f

    normal_imgs = torch.stack([
        torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).to("cuda")
        for img in rm_normals
    ])

    _log(f"poisson: sampling target normals from {len(rm_normals)} views")
    target_normals, has_target = _sample_vertex_normals_from_views(
        mesh_v, mesh_f, normal_imgs, camera_indices, normal_flip,
    )

    with torch.no_grad():
        vn = calc_vertex_normals(mesh_v, mesh_f)
        cos_before = (vn[has_target] * target_normals[has_target]).sum(dim=-1).mean()
    _log(f"poisson: initial cos_sim={cos_before.item():.4f}")

    vertices = _poisson_vertex_update(
        mesh_v, mesh_f, target_normals, has_target,
        outer_iters=1, inner_iters=40, sigma=0.5,
    )
    faces = mesh_f

    _save_debug_glb(vertices, faces, "after_poisson")

    _log(f"post: before clean verts={len(vertices)} faces={len(faces)}")
    meshes = simple_clean_mesh(
        to_pyml_mesh(vertices, faces),
        apply_smooth=True, stepsmoothnum=2,
        apply_sub_divide=False, sub_divide_threshold=0.25,
    ).to("cuda")
    simp_vertices, simp_faces = meshes.verts_packed(), meshes.faces_packed()
    _log(f"post: after clean+smooth verts={len(simp_vertices)} faces={len(simp_faces)}")
    vertices, faces = simp_vertices.detach().cpu().numpy(), simp_faces.detach().cpu().numpy()
    _save_debug_glb(simp_vertices, simp_faces, "after_smooth")

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh = merge_small_faces(mesh, thres=thres)
    _log(f"post: after merge_small_faces verts={len(mesh.vertices)} faces={len(mesh.faces)}")
    new_mesh = mesh.split(only_watertight=False)
    new_mesh = [j for j in new_mesh if len(j.vertices) >= 200]
    _log(f"post: {len(new_mesh)} components after split (>= 200 verts)")
    mesh = trimesh.Scene(new_mesh).dump(concatenate=True)
    vertices, faces = mesh.vertices.astype('float32'), mesh.faces

    _log(f"post: before subdivide verts={len(vertices)} faces={len(faces)}")
    vertices, faces = trimesh.remesh.subdivide(vertices, faces)
    _log(f"post: after subdivide verts={len(vertices)} faces={len(faces)}")
    origin_len_v, origin_len_f = len(vertices), len(faces)

    if has_fixed:
        fv_np = fixed_v_cpu.numpy()
        ff_np = fixed_f_cpu.numpy()
        vertices = np.concatenate([vertices, fv_np], axis=0)
        faces = np.concatenate([faces, ff_np + origin_len_v], axis=0)
    vertices, faces = torch.tensor(vertices, device='cuda'), torch.tensor(faces, device='cuda')

    gc.collect()
    torch.cuda.empty_cache()

    _log(f"color_projection: start verts={len(vertices)} faces={len(faces)}")
    t_cp = time.monotonic()
    meshes = Meshes(verts=[vertices], faces=[faces],
                    textures=pytorch3d.renderer.mesh.textures.TexturesVertex(
                        [torch.zeros_like(vertices).float()]))
    _azim = azim_list or STDGEN_VIEWS.azim_list
    cameras_list = get_cameras_list(_azim, "cuda", focal=1/1.2)
    _view_cfg = _find_view_config_by_azim(_azim)
    mvp_weights = _view_cfg.get_projection_weights(distract=distract_mask is not None)
    new_meshes = multiview_color_projection(
        meshes, rgb_ls, resolution=1024, device="cuda",
        complete_unseen=True, confidence_threshold=0.2,
        cameras_list=cameras_list, weights=mvp_weights,
        distract_mask=distract_mask,
    )
    _log(f"color_projection: done in {time.monotonic()-t_cp:.1f}s")
    del meshes, cameras_list
    gc.collect()
    torch.cuda.empty_cache()

    if has_fixed:
        new_meshes = Meshes(
            verts=[new_meshes.verts_packed()[:origin_len_v]],
            faces=[new_meshes.faces_packed()[:origin_len_f]],
            textures=pytorch3d.renderer.mesh.textures.TexturesVertex(
                [new_meshes.textures.verts_features_packed()[:origin_len_v]]),
        )
    return new_meshes, simp_vertices, simp_faces


def geo_refine(mesh_v, mesh_f, rgb_ls, normal_ls, expansion_weight=0.1, fixed_v=None, fixed_f=None,
               distract_mask=None, distract_bbox=None, thres=3e-6, no_decompose=False,
               camera_indices=None, azim_list=None, normal_flip=None):
    rm_normals = simple_remove(normal_ls)

    for idx, img in enumerate(rm_normals):
        rgb_ls[idx] = Image.fromarray(np.concatenate([np.array(rgb_ls[idx])[..., :3], np.array(img)[:, :, 3:4]], axis=-1))
    assert np.mean(np.array(rgb_ls[0])[..., 3]) < 250

    rgb_ls = erode_alpha(rgb_ls)

    has_fixed = fixed_v is not None and fixed_f is not None
    fixed_v_cpu = fixed_v.cpu() if has_fixed else None
    fixed_f_cpu = fixed_f.cpu() if has_fixed else None
    if has_fixed:
        del fixed_v, fixed_f

    stage1_lr = 0.08 if not has_fixed else 0.01
    stage1_remesh_interval = 1 if not has_fixed else 30

    if no_decompose:
        stage1_lr = 0.03
        stage1_remesh_interval = 30

    vertices, faces = reconstruct_stage1(rm_normals, steps=200, vertices=mesh_v, faces=mesh_f,
                                         fixed_v=fixed_v_cpu, fixed_f=fixed_f_cpu,
                                         lr=stage1_lr, remesh_interval=stage1_remesh_interval, start_edge_len=0.02,
                                         end_edge_len=0.005, gain=0.05, loss_expansion_weight=expansion_weight,
                                         distract_mask=distract_mask, distract_bbox=distract_bbox,
                                         camera_indices=camera_indices, normal_flip=normal_flip)

    _log(f"stage1 output: verts={len(vertices)} faces={len(faces)}")
    _save_debug_glb(vertices, faces, "after_stage1")
    vertices = vertices.detach().clone()
    faces = faces.detach().clone()
    gc.collect()
    torch.cuda.empty_cache()

    vertices, faces = run_mesh_refine(vertices, faces, rm_normals,
                                      fixed_v=fixed_v_cpu, fixed_f=fixed_f_cpu,
                                      steps=100, start_edge_len=0.005, end_edge_len=0.0002,
                                      decay=0.99, update_normal_interval=20, update_warmup=5,
                                      process_inputs=False, process_outputs=False, remesh_interval=1,
                                      camera_indices=camera_indices, azim_list=azim_list,
                                      normal_flip=normal_flip)

    _save_debug_glb(vertices, faces, "after_stage2")
    _log(f"post: before clean verts={len(vertices)} faces={len(faces)}")
    meshes = simple_clean_mesh(to_pyml_mesh(vertices, faces), apply_smooth=True, stepsmoothnum=2, apply_sub_divide=False, sub_divide_threshold=0.25).to("cuda")
    simp_vertices, simp_faces = meshes.verts_packed(), meshes.faces_packed()
    _log(f"post: after clean+smooth verts={len(simp_vertices)} faces={len(simp_faces)}")
    vertices, faces = simp_vertices.detach().cpu().numpy(), simp_faces.detach().cpu().numpy()
    _save_debug_glb(simp_vertices, simp_faces, "after_smooth")

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh = merge_small_faces(mesh, thres=thres)
    _log(f"post: after merge_small_faces verts={len(mesh.vertices)} faces={len(mesh.faces)}")
    new_mesh = mesh.split(only_watertight=False)

    new_mesh = [ j for j in new_mesh if len(j.vertices) >= 200 ]
    _log(f"post: {len(new_mesh)} components after split (>= 200 verts)")
    mesh = trimesh.Scene(new_mesh).dump(concatenate=True)
    vertices, faces = mesh.vertices.astype('float32'), mesh.faces

    _log(f"post: before subdivide verts={len(vertices)} faces={len(faces)}")
    vertices, faces = trimesh.remesh.subdivide(vertices, faces)
    _log(f"post: after subdivide verts={len(vertices)} faces={len(faces)}")
    origin_len_v, origin_len_f = len(vertices), len(faces)

    if has_fixed:
        fv_np = fixed_v_cpu.numpy()
        ff_np = fixed_f_cpu.numpy()
        vertices = np.concatenate([vertices, fv_np], axis=0)
        faces = np.concatenate([faces, ff_np + origin_len_v], axis=0)
    vertices, faces = torch.tensor(vertices, device='cuda'), torch.tensor(faces, device='cuda')

    gc.collect()
    torch.cuda.empty_cache()

    _log(f"color_projection: start verts={len(vertices)} faces={len(faces)}")
    t_cp = time.monotonic()
    meshes = Meshes(verts=[vertices], faces=[faces], textures=pytorch3d.renderer.mesh.textures.TexturesVertex([torch.zeros_like(vertices).float()]))
    _azim = azim_list or STDGEN_VIEWS.azim_list
    cameras_list = get_cameras_list(_azim, "cuda", focal=1/1.2)
    _view_cfg = _find_view_config_by_azim(_azim)
    mvp_weights = _view_cfg.get_projection_weights(distract=distract_mask is not None)
    new_meshes = multiview_color_projection(meshes, rgb_ls, resolution=1024, device="cuda", complete_unseen=True, confidence_threshold=0.2, cameras_list=cameras_list, weights=mvp_weights, distract_mask=distract_mask)
    _log(f"color_projection: done in {time.monotonic()-t_cp:.1f}s")
    del meshes, cameras_list
    gc.collect()
    torch.cuda.empty_cache()

    if has_fixed:
        new_meshes = Meshes(verts=[new_meshes.verts_packed()[:origin_len_v]], faces=[new_meshes.faces_packed()[:origin_len_f]],
                            textures=pytorch3d.renderer.mesh.textures.TexturesVertex([new_meshes.textures.verts_features_packed()[:origin_len_v]]))
    return new_meshes, simp_vertices, simp_faces
