import os

os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.expanduser("~/.cache/torch/inductor"))

import gc
import cv2
import numpy as np
import trimesh
import argparse
import torch
import scipy
from PIL import Image

from refine.mesh_refine import geo_refine
from refine.func import STDGEN_VIEWS, make_star_cameras_orthographic
from refine.render import NormalsRenderer, calc_vertex_normals

from pytorch3d.structures import Meshes
from sklearn.neighbors import KDTree

from vram_monitor import init_log, log_vram, set_stage

init_log()

_sam_generator = None


def _get_sam_generator():
    global _sam_generator
    if _sam_generator is not None:
        return _sam_generator

    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    log_vram("before SAM load")
    sam = sam_model_registry["vit_h"](checkpoint="./ckpt/sam_vit_h_4b8939.pth").float().cuda()
    log_vram("after SAM load")
    _sam_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=64,
        pred_iou_thresh=0.80,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
    )
    return _sam_generator


def _unload_sam():
    global _sam_generator
    if _sam_generator is not None:
        del _sam_generator
        _sam_generator = None
        gc.collect()
        torch.cuda.empty_cache()


def filter_fixed_mesh_by_proximity(fixed_v, fixed_f, target_v, margin=0.1):
    target_np = target_v if isinstance(target_v, np.ndarray) else target_v.numpy()
    fixed_np = fixed_v.numpy()

    kdtree = KDTree(target_np)
    dists, _ = kdtree.query(fixed_np, k=1)
    near_mask = dists.squeeze() < margin

    old_to_new = np.full(len(fixed_np), -1, dtype=np.int64)
    new_indices = np.where(near_mask)[0]
    old_to_new[new_indices] = np.arange(len(new_indices))

    fixed_f_np = fixed_f.numpy()
    v0_near = near_mask[fixed_f_np[:, 0]]
    v1_near = near_mask[fixed_f_np[:, 1]]
    v2_near = near_mask[fixed_f_np[:, 2]]
    face_mask = v0_near & v1_near & v2_near

    new_fixed_v = torch.from_numpy(fixed_np[near_mask])
    new_fixed_f = torch.from_numpy(old_to_new[fixed_f_np[face_mask]]).long()

    return new_fixed_v, new_fixed_f


def fix_vert_color_glb(mesh_path):
    from pygltflib import GLTF2, Material, PbrMetallicRoughness
    obj1 = GLTF2().load(mesh_path)
    obj1.meshes[0].primitives[0].material = 0
    obj1.materials.append(Material(
        pbrMetallicRoughness = PbrMetallicRoughness(
            baseColorFactor = [1.0, 1.0, 1.0, 1.0],
            metallicFactor = 0.,
            roughnessFactor = 1.0,
        ),
        emissiveFactor = [0.0, 0.0, 0.0],
        doubleSided = True,
    ))
    obj1.save(mesh_path)


def srgb_to_linear(c_srgb):
    c_linear = np.where(c_srgb <= 0.04045, c_srgb / 12.92, ((c_srgb + 0.055) / 1.055) ** 2.4)
    return c_linear.clip(0, 1.)


def save_py3dmesh_with_trimesh_fast(meshes: Meshes, save_glb_path, apply_sRGB_to_LinearRGB=True):
    # convert from pytorch3d meshes to trimesh mesh
    vertices = meshes.verts_packed().cpu().float().numpy()
    triangles = meshes.faces_packed().cpu().long().numpy()
    np_color = meshes.textures.verts_features_packed().cpu().float().numpy()
    if save_glb_path.endswith(".glb"):
        # rotate 180 along +Y
        vertices[:, [0, 2]] = -vertices[:, [0, 2]]

    if apply_sRGB_to_LinearRGB:
        np_color = srgb_to_linear(np_color)
    assert vertices.shape[0] == np_color.shape[0]
    assert np_color.shape[1] == 3
    assert 0 <= np_color.min() and np_color.max() <= 1.001, f"min={np_color.min()}, max={np_color.max()}"
    np_color = np.clip(np_color, 0, 1)
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, vertex_colors=np_color)
    mesh.remove_unreferenced_vertices()
    # save mesh
    mesh.export(save_glb_path)
    if save_glb_path.endswith(".glb"):
        fix_vert_color_glb(save_glb_path)
    print(f"saving to {save_glb_path}")


def calc_horizontal_offset(target_img, source_img):
    target_mask = target_img.astype(np.float32).sum(axis=-1) > 750
    source_mask = source_img.astype(np.float32).sum(axis=-1) > 750
    best_offset = -114514
    for offset in range(-200, 200):
        offset_mask = np.roll(source_mask, offset, axis=1)
        overlap = (target_mask & offset_mask).sum()
        if overlap > best_offset:
            best_offset = overlap
            best_offset_value = offset
    return best_offset_value


def calc_horizontal_offset2(target_mask, source_img):
    source_mask = source_img.astype(np.float32).sum(axis=-1) > 750
    best_offset = -114514
    for offset in range(-200, 200):
        offset_mask = np.roll(source_mask, offset, axis=1)
        overlap = (target_mask & offset_mask).sum()
        if overlap > best_offset:
            best_offset = overlap
            best_offset_value = offset
    return best_offset_value


def get_distract_mask(color_0, color_1, normal_0=None, normal_1=None, thres=0.25, ratio=0.50, outside_thres=0.10, outside_ratio=0.20):
    distract_area = np.abs(color_0 - color_1).sum(axis=-1) > thres
    if normal_0 is not None and normal_1 is not None:
        distract_area |= np.abs(normal_0 - normal_1).sum(axis=-1) > thres
    labeled_array, num_features = scipy.ndimage.label(distract_area)
    results = []

    random_sampled_points = []

    for i in range(num_features + 1):
        if np.sum(labeled_array == i) > 1000 and np.sum(labeled_array == i) < 100000:
            results.append((i, np.sum(labeled_array == i)))
            # random sample a point in the area
            points = np.argwhere(labeled_array == i)
            random_sampled_points.append(points[np.random.randint(0, points.shape[0])])

    results = sorted(results, key=lambda x: x[1], reverse=True)  # [1:]
    distract_mask = np.zeros_like(distract_area)
    distract_bbox = np.zeros_like(distract_area)
    for i, _ in results:
        distract_mask |= labeled_array == i
        bbox = np.argwhere(labeled_array == i)
        min_x, min_y = bbox.min(axis=0)
        max_x, max_y = bbox.max(axis=0)
        distract_bbox[min_x:max_x, min_y:max_y] = 1

    points = np.array(random_sampled_points)[:, ::-1]
    labels = np.ones(len(points), dtype=np.int32)

    masks = _get_sam_generator().generate((color_1 * 255).astype(np.uint8))

    outside_area = np.abs(color_0 - color_1).sum(axis=-1) < outside_thres

    final_mask = np.zeros_like(distract_mask)
    for iii, mask in enumerate(masks):
        mask['segmentation'] = cv2.resize(mask['segmentation'].astype(np.float32), (1024, 1024)) > 0.5
        intersection = np.logical_and(mask['segmentation'], distract_mask).sum()
        total = mask['segmentation'].sum()
        iou = intersection / total
        outside_intersection = np.logical_and(mask['segmentation'], outside_area).sum()
        outside_total = mask['segmentation'].sum()
        outside_iou = outside_intersection / outside_total
        if iou > ratio and outside_iou < outside_ratio:
            final_mask |= mask['segmentation']

    # calculate coverage
    intersection = np.logical_and(final_mask, distract_mask).sum()
    total = distract_mask.sum()
    coverage = intersection / total

    if coverage < 0.8:
        # use original distract mask
        final_mask = (distract_mask.copy() * 255).astype(np.uint8)
        final_mask = cv2.dilate(final_mask, np.ones((3, 3), np.uint8), iterations=3)
        labeled_array_dilate, num_features_dilate = scipy.ndimage.label(final_mask)
        for i in range(num_features_dilate + 1):
            if np.sum(labeled_array_dilate == i) < 200:
                final_mask[labeled_array_dilate == i] = 255

        final_mask = cv2.erode(final_mask, np.ones((3, 3), np.uint8), iterations=3)
        final_mask = final_mask > 127

    return distract_mask, distract_bbox, random_sampled_points, final_mask


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_mv_dir', type=str, default='result/multiview')
    parser.add_argument('--input_obj_dir', type=str, default='result/slrm')
    parser.add_argument('--output_dir', type=str, default='result/refined')
    parser.add_argument('--outside_ratio', type=float, default=0.20)
    parser.add_argument('--no_decompose', action='store_true')
    args = parser.parse_args()

    for test_idx in os.listdir(args.input_mv_dir):
        mv_root_dir = os.path.join(args.input_mv_dir, test_idx)
        obj_dir = os.path.join(args.input_obj_dir, test_idx)

        fixed_v, fixed_f = None, None
        last_colors, last_normals = None, None
        last_front_color, last_front_normal = None, None
        distract_mask = None

        mv, proj = make_star_cameras_orthographic(8, 1, r=1.2)
        mv = mv[STDGEN_VIEWS.camera_indices]
        renderer = NormalsRenderer(mv,proj,(1024,1024))

        log_vram("before refine loop")

        if not args.no_decompose:
            for name_idx, level in zip([3, 1, 2], [2, 1, 0]):
                gc.collect()
                torch.cuda.empty_cache()
                set_stage(f"decompose_level{level}", step=0)
                log_vram(f"start level{level}")
                mesh = trimesh.load(obj_dir + f'_{name_idx}.obj')
                new_mesh = mesh.split(only_watertight=False)
                new_mesh = [ j for j in new_mesh if len(j.vertices) >= 300 ]
                mesh = trimesh.Scene(new_mesh).dump(concatenate=True)
                mesh_v, mesh_f = mesh.vertices, mesh.faces

                if last_colors is None:
                    images = renderer.render(
                        torch.tensor(mesh_v, device='cuda').float(),
                        torch.ones_like(torch.from_numpy(mesh_v), device='cuda').float(),
                        torch.tensor(mesh_f, device='cuda'),
                    )
                    mask = (images[..., 3] < 0.9).cpu().numpy()

                colors, normals = [], []
                for i in range(6):
                    color_path = os.path.join(mv_root_dir, f'level{level}', f'color_{i}.png')
                    normal_path = os.path.join(mv_root_dir, f'level{level}', f'normal_{i}.png')
                    color = cv2.imread(color_path)
                    normal = cv2.imread(normal_path)
                    color = color[..., ::-1]
                    normal = normal[..., ::-1]

                    if last_colors is not None:
                        offset = calc_horizontal_offset(np.array(last_colors[i]), color)
                        # print('offset', i, offset)
                    else:
                        offset = calc_horizontal_offset2(mask[i], color)
                        # print('init offset', i, offset)

                    if offset != 0:
                        color = np.roll(color, offset, axis=1)
                        normal = np.roll(normal, offset, axis=1)

                    color = Image.fromarray(color)
                    normal = Image.fromarray(normal)
                    colors.append(color)
                    normals.append(normal)

                if last_front_color is not None and level == 0:
                    original_mask, distract_bbox, _, distract_mask = get_distract_mask(last_front_color, np.array(colors[0]).astype(np.float32) / 255.0, outside_ratio=args.outside_ratio)
                    cv2.imwrite(f'{args.output_dir}/{test_idx}/distract_mask.png', distract_mask.astype(np.uint8) * 255)

                    _unload_sam()
                    log_vram("after SAM unload")
                else:
                    distract_mask = None
                    distract_bbox = None

                last_front_color = np.array(colors[0]).astype(np.float32) / 255.0
                last_front_normal = np.array(normals[0]).astype(np.float32) / 255.0

                if last_colors is None:
                    from copy import deepcopy
                    last_colors, last_normals = deepcopy(colors), deepcopy(normals)

                # my mesh flow weight by nearest vertexs
                if fixed_v is not None and fixed_f is not None and level == 1:
                    kdtree_anchor = KDTree(fixed_v.numpy())
                    kdtree_mesh_v = KDTree(mesh_v)
                    _, idx_anchor = kdtree_anchor.query(mesh_v, k=1)
                    _, idx_mesh_v = kdtree_mesh_v.query(mesh_v, k=25)
                    idx_anchor = idx_anchor.squeeze()
                    neighbors = torch.tensor(mesh_v).cuda()[idx_mesh_v]
                    neighbor_dists = torch.norm(neighbors - torch.tensor(mesh_v).cuda()[:, None], dim=-1)
                    neighbor_dists[neighbor_dists > 0.06] = 114514.
                    neighbor_weights = torch.exp(-neighbor_dists * 1.)
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

                mesh_v = torch.tensor(mesh_v, device='cuda', dtype=torch.float32)
                mesh_f = torch.tensor(mesh_f, device='cuda')

                level_fixed_v, level_fixed_f = fixed_v, fixed_f
                if level == 0 and fixed_v is not None:
                    level_fixed_v, level_fixed_f = filter_fixed_mesh_by_proximity(
                        fixed_v, fixed_f, mesh_v.cpu(), margin=0.15,
                    )
                    log_vram(f"fixed_v filtered: {fixed_v.shape[0]} -> {level_fixed_v.shape[0]}")

                set_stage(f"geo_refine_level{level}")
                log_vram(f"before geo_refine level{level}")
                new_mesh, simp_v, simp_f = geo_refine(mesh_v, mesh_f, colors, normals, fixed_v=level_fixed_v, fixed_f=level_fixed_f, distract_mask=distract_mask, distract_bbox=distract_bbox)
                log_vram(f"after geo_refine level{level}")

                # my mesh flow weight by nearest vertexs
                try:
                    if fixed_v is not None and fixed_f is not None and level != 0:
                        new_mesh_v = new_mesh.verts_packed().cpu().numpy()

                        kdtree_anchor = KDTree(fixed_v.numpy())
                        kdtree_mesh_v = KDTree(new_mesh_v)
                        _, idx_anchor = kdtree_anchor.query(new_mesh_v, k=1)
                        _, idx_mesh_v = kdtree_mesh_v.query(new_mesh_v, k=25)
                        idx_anchor = idx_anchor.squeeze()
                        neighbors = torch.tensor(new_mesh_v).cuda()[idx_mesh_v]
                        neighbor_dists = torch.norm(neighbors - torch.tensor(new_mesh_v).cuda()[:, None], dim=-1)
                        neighbor_dists[neighbor_dists > 0.06] = 114514.
                        neighbor_weights = torch.exp(-neighbor_dists * 1.)
                        neighbor_weights = neighbor_weights / neighbor_weights.sum(dim=1, keepdim=True)
                        fv_gpu = fixed_v.cuda()
                        ff_gpu = fixed_f.cuda()
                        anchors = fv_gpu[idx_anchor]
                        anchor_normals = calc_vertex_normals(fv_gpu, ff_gpu)[idx_anchor]
                        dis_anchor = torch.clamp(((anchors - torch.tensor(new_mesh_v).cuda()) * anchor_normals).sum(-1), min=0) + 0.01
                        vec_anchor = dis_anchor[:, None] * anchor_normals
                        vec_anchor = vec_anchor[idx_mesh_v]
                        weighted_vec_anchor = (vec_anchor * neighbor_weights[:, :, None]).sum(1)
                        new_mesh_v += weighted_vec_anchor.cpu().numpy()
                        del fv_gpu, ff_gpu, anchors, anchor_normals, neighbors, neighbor_dists, neighbor_weights
                        torch.cuda.empty_cache()

                        new_mesh = Meshes(verts=[torch.tensor(new_mesh_v, device='cuda')], faces=new_mesh.faces_list(), textures=new_mesh.textures)

                except Exception:
                    pass

                os.makedirs(f'{args.output_dir}/{test_idx}', exist_ok=True)
                save_py3dmesh_with_trimesh_fast(new_mesh, f'{args.output_dir}/{test_idx}/out_{level}.glb', apply_sRGB_to_LinearRGB=True)

                if fixed_v is None:
                    fixed_v, fixed_f = simp_v.cpu(), simp_f.cpu()
                else:
                    fixed_f = torch.cat([fixed_f, simp_f.cpu() + fixed_v.shape[0]], dim=0)
                    fixed_v = torch.cat([fixed_v, simp_v.cpu()], dim=0)

                del new_mesh, simp_v, simp_f, colors, normals, mesh_v, mesh_f, level_fixed_v, level_fixed_f
                gc.collect()
                torch.cuda.empty_cache()


        else:
            mesh = trimesh.load(obj_dir + f'_0.obj')
            mesh_v, mesh_f = mesh.vertices, mesh.faces

            images = renderer.render(
                torch.tensor(mesh_v, device='cuda').float(),
                torch.ones_like(torch.from_numpy(mesh_v), device='cuda').float(),
                torch.tensor(mesh_f, device='cuda'),
            )
            mask = (images[..., 3] < 0.9).cpu().numpy()

            colors, normals = [], []
            for i in range(6):
                color_path = os.path.join(mv_root_dir, f'level0', f'color_{i}.png')
                normal_path = os.path.join(mv_root_dir, f'level0', f'normal_{i}.png')
                color = cv2.imread(color_path)
                normal = cv2.imread(normal_path)
                color = color[..., ::-1]
                normal = normal[..., ::-1]

                offset = calc_horizontal_offset2(mask[i], color)

                if offset != 0:
                    color = np.roll(color, offset, axis=1)
                    normal = np.roll(normal, offset, axis=1)

                color = Image.fromarray(color)
                normal = Image.fromarray(normal)
                colors.append(color)
                normals.append(normal)

            mesh_v = torch.tensor(mesh_v, device='cuda', dtype=torch.float32)
            mesh_f = torch.tensor(mesh_f, device='cuda')

            new_mesh, _, _ = geo_refine(mesh_v, mesh_f, colors, normals, no_decompose=True, expansion_weight=0.)

            os.makedirs(f'{args.output_dir}/{test_idx}', exist_ok=True)
            save_py3dmesh_with_trimesh_fast(new_mesh, f'{args.output_dir}/{test_idx}/out_nodecomp.glb', apply_sRGB_to_LinearRGB=True)
