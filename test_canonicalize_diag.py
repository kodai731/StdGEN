"""Stage 1.1 diagnostic: traces device/dtype/VRAM at every step.
Writes all output to diag_canonicalize.txt.
Runs only 1 denoising step then exits.
"""

import torch
import sys
import os
import gc
from datetime import datetime

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

LOG_PATH = "diag_canonicalize.txt"

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def gpu():
    a = torch.cuda.memory_allocated() / 1024**3
    r = torch.cuda.memory_reserved() / 1024**3
    return f"alloc={a:.2f}GB reserved={r:.2f}GB"


def tensor_info(name, t):
    if t is None:
        return f"{name}: None"
    return f"{name}: shape={list(t.shape)} dtype={t.dtype} device={t.device}"


def model_info(name, m):
    params = list(m.parameters())
    if not params:
        return f"{name}: no params"
    devices = set(str(p.device) for p in params)
    dtypes = set(str(p.dtype) for p in params)
    size = sum(p.numel() * p.element_size() for p in params) / 1024**3
    return f"{name}: {size:.2f}GB devices={devices} dtypes={dtypes}"


def main():
    with open(LOG_PATH, "w") as f:
        f.write(f"=== Canonicalize Diagnostic {datetime.now()} ===\n\n")

    device = torch.device("cuda")
    weight_dtype = torch.float16

    log(f"device={device} weight_dtype={weight_dtype}")
    log(f"GPU: {gpu()}")

    from omegaconf import OmegaConf
    config = OmegaConf.load("./configs/canonicalization-infer.yaml")
    path = config.pretrained_model_path
    kwargs = dict(config.unet_from_pretrained_kwargs)

    from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor, CLIPVisionModelWithProjection
    from diffusers import AutoencoderKL, DDIMScheduler
    from canonicalize.models.unet_mv2d_condition import UNetMV2DConditionModel
    from canonicalize.models.unet_mv2d_ref import UNetMV2DRefModel
    from canonicalize.pipeline_canonicalize import CanonicalizationPipeline

    log("--- Loading models ---")
    tokenizer = CLIPTokenizer.from_pretrained(path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(path, subfolder="text_encoder").to(dtype=weight_dtype, device=device)
    log(model_info("text_encoder", text_encoder))
    log(f"GPU: {gpu()}")

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(path, subfolder="image_encoder").to(dtype=weight_dtype, device=device)
    log(model_info("image_encoder", image_encoder))
    log(f"GPU: {gpu()}")

    vae = AutoencoderKL.from_pretrained(path, subfolder="vae").to(dtype=weight_dtype, device=device)
    log(model_info("vae", vae))
    log(f"GPU: {gpu()}")

    unet = UNetMV2DConditionModel.from_pretrained_2d(path, subfolder="unet", local_crossattn=True, **kwargs).to(dtype=weight_dtype, device=device)
    log(model_info("unet", unet))
    log(f"GPU: {gpu()}")

    ref_unet = UNetMV2DRefModel.from_pretrained_2d(path, subfolder="ref_unet", local_crossattn=True, **kwargs).to(dtype=weight_dtype, device=device)
    log(model_info("ref_unet", ref_unet))
    log(f"GPU: {gpu()}")

    gc.collect()
    torch.cuda.empty_cache()
    log(f"GPU after gc: {gpu()}")

    log("--- Building pipeline ---")
    feature_extractor = CLIPImageProcessor()
    noise_scheduler = DDIMScheduler.from_pretrained(path, subfolder="scheduler-zerosnr")

    pipeline = CanonicalizationPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
        unet=unet, ref_unet=ref_unet,
        feature_extractor=feature_extractor, image_encoder=image_encoder,
        scheduler=noise_scheduler,
    )
    pipeline.enable_vae_slicing()
    log("Pipeline built")

    log("--- Testing encode_prompt ---")
    prompt = "high quality, best quality"
    prompt_ids = tokenizer(
        prompt, max_length=tokenizer.model_max_length, padding="max_length",
        truncation=True, return_tensors="pt"
    ).input_ids[0]
    log(f"{tensor_info('prompt_ids', prompt_ids)}")

    text_input = tokenizer(
        prompt, padding="max_length", max_length=tokenizer.model_max_length,
        truncation=True, return_tensors="pt"
    )
    log(f"text_input.input_ids device={text_input.input_ids.device}")
    text_emb = text_encoder(text_input.input_ids.to(device))[0]
    log(f"{tensor_info('text_embeddings', text_emb)}")
    log(f"GPU after text encode: {gpu()}")

    log("--- Testing encode_image ---")
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode
    import torchvision.transforms.functional as TF

    dummy_img = torch.randn(1, 3, 1024, 640, device=device, dtype=weight_dtype)
    log(f"{tensor_info('dummy_img', dummy_img)}")

    clip_mean = torch.as_tensor(feature_extractor.image_mean)[:,None,None].to(device, dtype=torch.float32)
    clip_std = torch.as_tensor(feature_extractor.image_std)[:,None,None].to(device, dtype=torch.float32)
    log(f"{tensor_info('clip_mean', clip_mean)}")

    crop_h = feature_extractor.crop_size['height']
    crop_w = feature_extractor.crop_size['width']
    imgs_proc = TF.resize(dummy_img, (crop_h, crop_w), interpolation=InterpolationMode.BICUBIC)
    imgs_proc = ((imgs_proc.float() - clip_mean) / clip_std).to(weight_dtype)
    log(f"{tensor_info('imgs_proc (for image_encoder)', imgs_proc)}")

    img_emb = image_encoder(imgs_proc).image_embeds.unsqueeze(1)
    log(f"{tensor_info('image_embeddings', img_emb)}")
    log(f"GPU after image encode: {gpu()}")

    log("--- Testing VAE encode ---")
    vae_input = dummy_img[:, :, :, :]
    vae_latent = vae.encode(vae_input).latent_dist.sample()
    log(f"{tensor_info('vae_latent', vae_latent)}")
    log(f"GPU after vae encode: {gpu()}")

    del vae_latent, vae_input
    gc.collect()
    torch.cuda.empty_cache()
    log(f"GPU after vae cleanup: {gpu()}")

    log("--- Testing 1 denoising step ---")
    noise_scheduler.set_timesteps(40, device=device)
    t = noise_scheduler.timesteps[0]
    log(f"timestep t={t} device={t.device}")

    latent = torch.randn(2, 4, 1, 128, 80, device=device, dtype=weight_dtype)
    cond_latent = torch.randn(2, 4, 1, 128, 80, device=device, dtype=weight_dtype)
    text_emb_2 = text_emb.repeat(2, 1, 1)
    log(f"{tensor_info('latent', latent)}")
    log(f"{tensor_info('cond_latent', cond_latent)}")
    log(f"{tensor_info('text_emb_2', text_emb_2)}")
    log(f"GPU before ref_unet: {gpu()}")

    from einops import rearrange

    ref_dict = {}
    log("Calling ref_unet...")
    ref_out = ref_unet(
        cond_latent, t,
        encoder_hidden_states=text_emb_2,
        cross_attention_kwargs=dict(mode="w", ref_dict=ref_dict)
    ).sample
    log(f"{tensor_info('ref_unet output', ref_out)}")
    log(f"GPU after ref_unet: {gpu()}")

    img_emb_unet = img_emb.repeat(2, 1, 1)
    log(f"{tensor_info('img_emb_unet (encoder_hidden_states)', img_emb_unet)}")

    log("Calling unet...")
    unet_out = unet(
        latent, t,
        encoder_hidden_states=img_emb_unet,
        cross_attention_kwargs=dict(mode="r", ref_dict=ref_dict, is_cfg_guidance=True)
    ).sample
    log(f"{tensor_info('unet output', unet_out)}")
    log(f"GPU after unet: {gpu()}")

    log("")
    log("=== ALL STEPS PASSED ===")


if __name__ == "__main__":
    main()
