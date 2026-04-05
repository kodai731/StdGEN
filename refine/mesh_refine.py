import torch
import numpy as np
import trimesh
from PIL import Image
from typing import List
from tqdm import tqdm
from sklearn.neighbors import KDTree

from refine.func import from_py3d_mesh, get_cameras_list, make_star_cameras_orthographic, multiview_color_projection, simple_clean_mesh, to_py3d_mesh, to_pyml_mesh
from refine.opt import MeshOptimizer
from refine.render import NormalsRenderer, calc_vertex_normals

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
                       decay=0.995, loss_expansion_weight=0.1, gain=0.1, remesh_interval=1, remesh_start=0, distract_mask=None, distract_bbox=None):
    vertices, faces = vertices.cuda(), faces.cuda()
    assert len(pils) == 6
    mv, proj = make_star_cameras_orthographic(8, 1, r=1.2)
    mv = mv[[4, 3, 2, 0, 6, 5]]

    render_size = list(pils[0].size)
    if fixed_v is not None:
        render_size = [s // 2 for s in render_size]
        pils = [p.resize(render_size, Image.BILINEAR) for p in pils]
        if distract_mask is not None:
            distract_mask = distract_mask[::2, ::2]

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
        torch.cuda.empty_cache()
        kdtree = KDTree(fixed_v_cpu.numpy())

    mask = target_images[..., -1] < 0.5

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

        normals[:, 0] *= -1
        normals[:, 2] *= -1

        images = renderer.render(vertices, normals, faces)
        loss_expand = 0.5 * ((vertices+normals).detach() - vertices).pow(2).mean()

        if has_fixed:
            del fv, ff

        t_mask = images[..., -1] > 0.5
        loss_target_l2 = (images[t_mask] - target_images[t_mask]).abs().pow(2).mean()
        loss_alpha_target_mask_l2 = (images[..., -1][mask] - target_images[..., -1][mask]).pow(2).mean()

        loss = loss_target_l2 + loss_alpha_target_mask_l2 + loss_expand * loss_expansion_weight

        if distract_mask is not None:
            hair_visible_normals = normals
            hair_visible_normals[len(_vertices):] = -1.
            _images = renderer.render(vertices, hair_visible_normals, faces)
            loss_distract = (_images[0][distract_mask] - target_images[0][distract_mask]).pow(2).mean()

            target_outside = target_images[0][..., :3].clone()
            target_outside[~distract_mask] = 0.

            loss_outside_distract = (_images[0][..., :3][~distract_mask] - target_outside[..., :3][~distract_mask]).pow(2).mean()

            loss = loss + loss_distract * 1. + loss_outside_distract * 10.

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
        torch.cuda.empty_cache()

        if i % remesh_interval == 0 and i >= remesh_start:
            _vertices,_faces = opt.remesh(poisson=False)

    vertices, faces = opt._vertices.detach(), opt._faces.detach()

    return vertices, faces


def run_mesh_refine(vertices, faces, pils: List[Image.Image], fixed_v=None, fixed_f=None, steps=100, start_edge_len=0.02, end_edge_len=0.005,
                    decay=0.99, update_normal_interval=10, update_warmup=10, return_mesh=True, process_inputs=True, process_outputs=True, remesh_interval=20):
    poission_steps = []

    assert len(pils) == 6
    mv, proj = make_star_cameras_orthographic(8, 1, r=1.2)
    mv = mv[[4, 3, 2, 0, 6, 5]]

    render_size = list(pils[0].size)
    if fixed_v is not None:
        render_size = [s // 2 for s in render_size]
        pils = [p.resize(render_size, Image.BILINEAR) for p in pils]

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
        torch.cuda.empty_cache()

    mask = target_images[..., -1] < 0.5

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
                torch.cuda.empty_cache()
                py3d_mesh = to_py3d_mesh(vertices, faces, normals)
                cameras = get_cameras_list(azim_list=[180, 225, 270, 0, 90, 135], device=vertices.device, focal=1/1.2)
                projected = multiview_color_projection(py3d_mesh, pils, cameras_list=cameras, weights=[2,0.8,0.8,2,0.8,0.8], confidence_threshold=0.1, complete_unseen=False, below_confidence_strategy='original', reweight_with_cosangle='linear')
                _, _, target_normal = from_py3d_mesh(projected)
                del projected, py3d_mesh, cameras
                torch.cuda.empty_cache()

                target_normal = target_normal * 2 - 1
                target_normal = torch.nn.functional.normalize(target_normal, dim=-1)

                target_normal[:, 0] *= -1
                target_normal[:, 2] *= -1

                debug_images = renderer.render(vertices, target_normal, faces)
        
        d_mask = images[..., -1] > 0.5
        loss_debug_l2 = (images[..., :3][d_mask] - debug_images[..., :3][d_mask]).pow(2).mean()
        
        loss_alpha_target_mask_l2 = (images[..., -1][mask] - target_images[..., -1][mask]).pow(2).mean()
        
        loss = loss_debug_l2 + loss_alpha_target_mask_l2
        
        # out of box
        loss_oob = (vertices.abs() > 0.99).float().mean() * 10
        loss = loss + loss_oob
        
        loss.backward()
        opt.step()
        torch.cuda.empty_cache()

        if i % remesh_interval == 0:
            _vertices,_faces = opt.remesh(poisson=(i in poission_steps))

    vertices, faces = opt._vertices.detach(), opt._faces.detach()
    
    if process_outputs:
        vertices = vertices / 2 * 1.35
        vertices[..., [0, 2]] = - vertices[..., [0, 2]]

    return vertices, faces


def geo_refine(mesh_v, mesh_f, rgb_ls, normal_ls, expansion_weight=0.1, fixed_v=None, fixed_f=None,
               distract_mask=None, distract_bbox=None, thres=3e-6, no_decompose=False):
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
        torch.cuda.empty_cache()

    stage1_lr = 0.08 if not has_fixed else 0.01
    stage1_remesh_interval = 1 if not has_fixed else 30

    if no_decompose:
        stage1_lr = 0.03
        stage1_remesh_interval = 30

    vertices, faces = reconstruct_stage1(rm_normals, steps=200, vertices=mesh_v, faces=mesh_f,
                                         fixed_v=fixed_v_cpu, fixed_f=fixed_f_cpu,
                                         lr=stage1_lr, remesh_interval=stage1_remesh_interval, start_edge_len=0.02,
                                         end_edge_len=0.005, gain=0.05, loss_expansion_weight=expansion_weight,
                                         distract_mask=distract_mask, distract_bbox=distract_bbox)

    import gc
    vertices = vertices.detach().clone()
    faces = faces.detach().clone()
    gc.collect()
    torch.cuda.empty_cache()

    vertices, faces = run_mesh_refine(vertices, faces, rm_normals,
                                      fixed_v=fixed_v_cpu, fixed_f=fixed_f_cpu,
                                      steps=100, start_edge_len=0.005, end_edge_len=0.001,
                                      decay=0.99, update_normal_interval=20, update_warmup=2,
                                      process_inputs=False, process_outputs=False, remesh_interval=5)
    meshes = simple_clean_mesh(to_pyml_mesh(vertices, faces), apply_smooth=True, stepsmoothnum=2, apply_sub_divide=False, sub_divide_threshold=0.25).to("cuda")
    simp_vertices, simp_faces = meshes.verts_packed(), meshes.faces_packed()
    vertices, faces = simp_vertices.detach().cpu().numpy(), simp_faces.detach().cpu().numpy()

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh = merge_small_faces(mesh, thres=thres)
    new_mesh = mesh.split(only_watertight=False)

    new_mesh = [ j for j in new_mesh if len(j.vertices) >= 200 ]
    mesh = trimesh.Scene(new_mesh).dump(concatenate=True)
    vertices, faces = mesh.vertices.astype('float32'), mesh.faces

    vertices, faces = trimesh.remesh.subdivide(vertices, faces)
    origin_len_v, origin_len_f = len(vertices), len(faces)

    if has_fixed:
        fv_np = fixed_v_cpu.numpy()
        ff_np = fixed_f_cpu.numpy()
        vertices = np.concatenate([vertices, fv_np], axis=0)
        faces = np.concatenate([faces, ff_np + origin_len_v], axis=0)
    vertices, faces = torch.tensor(vertices, device='cuda'), torch.tensor(faces, device='cuda')

    gc.collect()
    torch.cuda.empty_cache()

    meshes = Meshes(verts=[vertices], faces=[faces], textures=pytorch3d.renderer.mesh.textures.TexturesVertex([torch.zeros_like(vertices).float()]))
    cameras_list = get_cameras_list([180, 225, 270, 0, 90, 135], "cuda", focal=1/1.2)
    mvp_weights = [2.0, 0.5, 0.0, 1.0, 0.0, 0.5] if distract_mask is None else [2.0, 0.0, 0.5, 1.0, 0.5, 0.0]
    new_meshes = multiview_color_projection(meshes, rgb_ls, resolution=1024, device="cuda", complete_unseen=True, confidence_threshold=0.2, cameras_list=cameras_list, weights=mvp_weights, distract_mask=distract_mask)
    del meshes, cameras_list
    gc.collect()
    torch.cuda.empty_cache()

    if has_fixed:
        new_meshes = Meshes(verts=[new_meshes.verts_packed()[:origin_len_v]], faces=[new_meshes.faces_packed()[:origin_len_f]],
                            textures=pytorch3d.renderer.mesh.textures.TexturesVertex([new_meshes.textures.verts_features_packed()[:origin_len_v]]))
    return new_meshes, simp_vertices, simp_faces
