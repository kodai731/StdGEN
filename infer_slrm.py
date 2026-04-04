import os
import imageio
import numpy as np
import torch
import cv2
import glob
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import v2
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from tqdm import tqdm

from slrm.utils.train_util import instantiate_from_config
from slrm.utils.camera_util import (
    FOV_to_intrinsics, 
    get_circular_camera_poses,
)
from slrm.utils.mesh_util import save_obj, save_glb
from slrm.utils.infer_util import images_to_video

from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_render_cameras(batch_size=1, M=120, radius=2.5, elevation=10.0, is_flexicubes=False):
    """
    Get the rendering camera parameters.
    """
    c2ws = get_circular_camera_poses(M=M, radius=radius, elevation=elevation)
    if is_flexicubes:
        cameras = torch.linalg.inv(c2ws)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    else:
        extrinsics = c2ws.flatten(-2)
        intrinsics = FOV_to_intrinsics(30.0).unsqueeze(0).repeat(M, 1, 1).float().flatten(-2)
        cameras = torch.cat([extrinsics, intrinsics], dim=-1)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1)
    return cameras


def images_to_video(images, output_dir, fps=30):
    # images: (N, C, H, W)
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    frames = []
    for i in range(images.shape[0]):
        frame = (images[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8).clip(0, 255)
        assert frame.shape[0] == images.shape[2] and frame.shape[1] == images.shape[3], \
            f"Frame shape mismatch: {frame.shape} vs {images.shape}"
        assert frame.min() >= 0 and frame.max() <= 255, \
            f"Frame value out of range: {frame.min()} ~ {frame.max()}"
        frames.append(frame)
    imageio.mimwrite(output_dir, np.stack(frames), fps=fps, codec='h264')


###############################################################################
# Configuration.
###############################################################################

seed_everything(0)

config_path = 'configs/mesh-slrm-infer.yaml'
config = OmegaConf.load(config_path)
config_name = os.path.basename(config_path).replace('.yaml', '')
model_config = config.model_config
infer_config = config.infer_config

IS_FLEXICUBES = True if config_name.startswith('mesh') else False

device = torch.device('cuda')

# load reconstruction model
print('Loading reconstruction model ...')
model = instantiate_from_config(model_config)
state_dict = torch.load(infer_config.model_path, map_location='cpu')
model.load_state_dict(state_dict, strict=False)

model = model.to(device)
if IS_FLEXICUBES:
    model.init_flexicubes_geometry(device, fovy=30.0, is_ortho=model.is_ortho)
model = model.eval()

print('Loading Finished!')

def make_mesh(mesh_fpath, planes, level=None):

    mesh_basename = os.path.basename(mesh_fpath).split('.')[0]
    mesh_dirname = os.path.dirname(mesh_fpath)
    mesh_glb_fpath = os.path.join(mesh_dirname, f"{mesh_basename}.glb")
        
    with torch.no_grad():
        # get mesh
        mesh_out = model.extract_mesh(
            planes,
            use_texture_map=False,
            levels=torch.tensor([level]).to(device),
            **infer_config,
        )

        vertices, faces, vertex_colors = mesh_out
        vertices = vertices[:, [1, 2, 0]]
        
        save_glb(vertices, faces, vertex_colors, mesh_glb_fpath)
        save_obj(vertices, faces, vertex_colors, mesh_fpath)

    return mesh_fpath, mesh_glb_fpath


def make3d(images, name, output_dir):
    input_cameras = torch.tensor(np.load('slrm/cameras.npy')).to(device)

    render_cameras = get_render_cameras(
        batch_size=1, radius=4.5, elevation=20.0, is_flexicubes=IS_FLEXICUBES).to(device)

    images = images.unsqueeze(0).to(device)
    images = v2.functional.resize(images, (320, 320), interpolation=3, antialias=True).clamp(0, 1)

    mesh_fpath = os.path.join(output_dir, f"{name}.obj")

    mesh_basename = os.path.basename(mesh_fpath).split('.')[0]
    mesh_dirname = os.path.dirname(mesh_fpath)
    video_fpath = os.path.join(mesh_dirname, f"{mesh_basename}.mp4")

    with torch.no_grad():
        torch.cuda.empty_cache()
        planes = model.forward_planes(images, input_cameras.float())

        # get video
        chunk_size = 20 if IS_FLEXICUBES else 1
        render_size = 512
        
        frames = [ [] for _ in range(4) ]
        for i in tqdm(range(0, render_cameras.shape[1], chunk_size)):
            if IS_FLEXICUBES:
                frame = model.forward_geometry_separate(
                    planes,
                    render_cameras[:, i:i+chunk_size],
                    render_size=render_size,
                    levels=torch.tensor([0]).to(device),
                )['imgs']
                for j in range(4):
                    frames[j].append(frame[j])
            else:
                frame = model.synthesizer(
                    planes,
                    cameras=render_cameras[:, i:i+chunk_size],
                    render_size=render_size,
                )['images_rgb']
                frames.append(frame)

        for j in range(4):
            frames[j] = torch.cat(frames[j], dim=1)
            video_fpath_j = video_fpath.replace('.mp4', f'_{j}.mp4')
            images_to_video(
                frames[j][0],
                video_fpath_j,
                fps=30,
            )

            _, mesh_glb_fpath = make_mesh(mesh_fpath.replace(mesh_fpath[-4:], f'_{j}{mesh_fpath[-4:]}'), planes, level=[0, 3, 4, 2][j])

    return video_fpath, mesh_fpath, mesh_glb_fpath


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="result/multiview")
    parser.add_argument('--output_dir', type=str, default="result/slrm")
    args = parser.parse_args()

    paths = glob.glob(args.input_dir + '/*')
    os.makedirs(args.output_dir, exist_ok=True)

    def load_rgb(path):
        img = plt.imread(path)
        img = Image.fromarray(np.uint8(img * 255.))
        return img
    
    for path in tqdm(paths):
        name = path.split('/')[-1]
        index_targets = [
            'level0/color_0.png',
            'level0/color_1.png',
            'level0/color_2.png',
            'level0/color_3.png',
            'level0/color_4.png',
            'level0/color_5.png',
        ] 
        imgs = []
        for index_target in index_targets:
            img = load_rgb(os.path.join(path, index_target))
            imgs.append(img)

        imgs = np.stack(imgs, axis=0).astype(np.float32) / 255.0
        imgs = torch.from_numpy(np.array(imgs)).permute(0, 3, 1, 2).contiguous().float()   # (6, 3, 1024, 1024)

        video_fpath, mesh_fpath, mesh_glb_fpath = make3d(imgs, name, args.output_dir)

