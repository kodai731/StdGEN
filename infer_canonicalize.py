import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from PIL import Image
import glob

import io
import argparse
import inspect
import os
import random
from typing import Dict, Optional, Tuple
from omegaconf import OmegaConf
import numpy as np

import torch

from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils import check_min_version
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor, CLIPVisionModelWithProjection
from torchvision import transforms

from canonicalize.models.unet_mv2d_condition import UNetMV2DConditionModel
from canonicalize.models.unet_mv2d_ref import UNetMV2DRefModel
from canonicalize.pipeline_canonicalize import CanonicalizationPipeline
from einops import rearrange
from torchvision.utils import save_image
import json
import cv2

import onnxruntime as rt
from huggingface_hub.file_download import hf_hub_download
from rm_anime_bg.cli import get_mask, SCALE

check_min_version("0.24.0")
weight_dtype = torch.float32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BkgRemover:
    def __init__(self, force_cpu: Optional[bool] = True):
        session_infer_path = hf_hub_download(
            repo_id="skytnt/anime-seg", filename="isnetis.onnx",
        )
        providers: list[str] = ["CPUExecutionProvider"]
        if not force_cpu and "CUDAExecutionProvider" in rt.get_available_providers():
            providers = ["CUDAExecutionProvider"]

        self.session_infer = rt.InferenceSession(
            session_infer_path, providers=providers,
        )

    def remove_background(
        self,
        img: np.ndarray,
        alpha_min: float,
        alpha_max: float,
    ) -> list:
        img = np.array(img)
        mask = get_mask(self.session_infer, img)
        mask[mask < alpha_min] = 0.0
        mask[mask > alpha_max] = 1.0
        img_after = (mask * img).astype(np.uint8)
        mask = (mask * SCALE).astype(np.uint8)
        img_after = np.concatenate([img_after, mask], axis=2, dtype=np.uint8)
        return Image.fromarray(img_after)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def process_image(image, totensor, width, height):
    assert image.mode == "RGBA"

    # Find non-transparent pixels
    non_transparent = np.nonzero(np.array(image)[..., 3])
    min_x, max_x = non_transparent[1].min(), non_transparent[1].max()
    min_y, max_y = non_transparent[0].min(), non_transparent[0].max()    
    image = image.crop((min_x, min_y, max_x, max_y))

    # paste to center
    max_dim = max(image.width, image.height)
    max_height = int(max_dim * 1.2)
    max_width = int(max_dim / (height/width) * 1.2)
    new_image = Image.new("RGBA", (max_width, max_height))
    left = (max_width - image.width) // 2
    top = (max_height - image.height) // 2
    new_image.paste(image, (left, top))

    image = new_image.resize((width, height), resample=Image.BICUBIC)
    image = np.array(image)
    image = image.astype(np.float32) / 255.
    assert image.shape[-1] == 4  # RGBA
    alpha = image[..., 3:4]
    bg_color = np.array([1., 1., 1.], dtype=np.float32)
    image = image[..., :3] * alpha + bg_color * (1 - alpha)
    return totensor(image)


@torch.no_grad()
def inference(validation_pipeline, bkg_remover, input_image, vae, feature_extractor, image_encoder, unet, ref_unet, tokenizer,
              text_encoder, pretrained_model_path, generator, validation, val_width, val_height, unet_condition_type,
              use_noise=True, noise_d=256, crop=False, seed=100, timestep=20):
    set_seed(seed)

    totensor = transforms.ToTensor()

    prompts = "high quality, best quality"
    prompt_ids = tokenizer(
        prompts, max_length=tokenizer.model_max_length, padding="max_length", truncation=True,
        return_tensors="pt"
    ).input_ids[0]

    # (B*Nv, 3, H, W)
    B = 1
    if input_image.mode != "RGBA":
        # remove background
        input_image = bkg_remover.remove_background(input_image, 0.1, 0.9)
    imgs_in = process_image(input_image, totensor, val_width, val_height)
    imgs_in = rearrange(imgs_in.unsqueeze(0).unsqueeze(0), "B Nv C H W -> (B Nv) C H W")

    imgs_in = imgs_in.to(device=device, dtype=torch.float32)
    out = validation_pipeline(prompt=prompts, image=imgs_in, generator=generator,
                              num_inference_steps=timestep, prompt_ids=prompt_ids,
                              height=val_height, width=val_width, unet_condition_type=unet_condition_type,
                              use_noise=use_noise, **validation,)
    out = rearrange(out, "B C f H W -> (B f) C H W", f=1)

    img_buf = io.BytesIO()
    save_image(out[0], img_buf, format='PNG')
    img_buf.seek(0)
    img = Image.open(img_buf)

    torch.cuda.empty_cache()
    return img


@torch.no_grad()
def main(
    input_dir: str,
    output_dir: str,
    pretrained_model_path: str,
    validation: Dict,
    local_crossattn: bool = True,
    unet_from_pretrained_kwargs=None,
    unet_condition_type=None,
    use_noise=True,
    noise_d=256,
    seed: int = 42,
    timestep: int = 40,
    width_input: int = 640,
    height_input: int = 1024,
):
    *_, config = inspect.getargvalues(inspect.currentframe())

    from vram_monitor import init_log, log_vram, check_vram_before_load, report_all_models

    init_log()
    log_vram("before model load")

    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(pretrained_model_path, subfolder="image_encoder")
    feature_extractor = CLIPImageProcessor()
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNetMV2DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder="unet", local_crossattn=local_crossattn, **unet_from_pretrained_kwargs)
    ref_unet = UNetMV2DRefModel.from_pretrained_2d(pretrained_model_path, subfolder="ref_unet", local_crossattn=local_crossattn, **unet_from_pretrained_kwargs)

    report_all_models(
        text_encoder=text_encoder,
        image_encoder=image_encoder,
        vae=vae,
        unet=unet,
        ref_unet=ref_unet,
    )

    for name, model in [("text_encoder", text_encoder), ("image_encoder", image_encoder), ("vae", vae), ("unet", unet), ("ref_unet", ref_unet)]:
        result = check_vram_before_load(name, model)
        if result < 0:
            raise RuntimeError(f"Cannot load {name} to GPU: would exceed VRAM limit")
        model.to(device)
        log_vram(f"after {name} to GPU")

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    ref_unet.requires_grad_(False)

    noise_scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler-zerosnr")
    validation_pipeline = CanonicalizationPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, ref_unet=ref_unet,feature_extractor=feature_extractor,image_encoder=image_encoder,
        scheduler=noise_scheduler,
    )
    validation_pipeline.enable_vae_slicing()
    validation_pipeline.set_progress_bar_config(disable=True)

    bkg_remover = BkgRemover()

    def canonicalize(image, width, height, seed, timestep):
        generator = torch.Generator(device=device).manual_seed(seed)
        return inference(
            validation_pipeline, bkg_remover, image, vae, feature_extractor, image_encoder, unet, ref_unet, tokenizer, text_encoder,
            pretrained_model_path, generator, validation, width, height, unet_condition_type,
            use_noise=use_noise, noise_d=noise_d, crop=True, seed=seed, timestep=timestep
        )

    img_paths = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    os.makedirs(output_dir, exist_ok=True)

    for path in tqdm(img_paths):
        img_input = Image.open(path)
        if np.array(img_input).shape[-1] == 4 and np.array(img_input)[..., 3].min() == 255:
            # convert to RGB
            img_input = img_input.convert("RGB")
        img_output = canonicalize(img_input, width_input, height_input, seed, timestep)
        img_output.save(os.path.join(output_dir, f"{os.path.basename(path).split('.')[0]}.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/canonicalization-infer.yaml")
    parser.add_argument("--input_dir", type=str, default="./input_cases")
    parser.add_argument("--output_dir", type=str, default="./result/apose")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(**OmegaConf.load(args.config), seed=args.seed, input_dir=args.input_dir, output_dir=args.output_dir)
