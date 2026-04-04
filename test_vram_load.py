"""VRAM loading diagnostic - runs for ~30s then exits with report."""

import torch
import sys
import os
import gc
from datetime import datetime

LOG_PATH = os.path.join(os.path.dirname(__file__), "vram_diag.txt")

def log(msg):
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def gpu_status():
    if not torch.cuda.is_available():
        return "CUDA not available"
    alloc = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return f"alloc={alloc:.2f}GB reserved={reserved:.2f}GB total={total:.1f}GB"


def verify_device(name, model):
    params = list(model.parameters())
    if not params:
        return f"{name}: no parameters"

    devices = set(str(p.device) for p in params)
    dtypes = set(str(p.dtype) for p in params)
    total_bytes = sum(p.numel() * p.element_size() for p in params)
    total_gb = total_bytes / 1024**3

    on_gpu = all("cuda" in str(p.device) for p in params)
    status = "GPU OK" if on_gpu else f"FAIL (devices: {devices})"

    return f"{name}: {total_gb:.2f}GB dtype={dtypes} {status}"


def main():
    with open(LOG_PATH, "w") as f:
        f.write(f"VRAM Diagnostic - {datetime.now()}\n\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weight_dtype = torch.float16

    log(f"Device: {device}")
    log(f"GPU: {gpu_status()}")
    log(f"System RAM: {os.popen('free -g').read().strip().split(chr(10))[1]}")
    log("")

    sys.path.insert(0, '.')
    from omegaconf import OmegaConf

    config = OmegaConf.load('./configs/canonicalization-infer.yaml')
    path = config.pretrained_model_path
    kwargs = dict(config.unet_from_pretrained_kwargs)

    from transformers import CLIPTextModel, CLIPVisionModelWithProjection
    from diffusers import AutoencoderKL
    from canonicalize.models.unet_mv2d_condition import UNetMV2DConditionModel
    from canonicalize.models.unet_mv2d_ref import UNetMV2DRefModel

    models = {}

    log("--- Loading models (fp32 -> fp16 on CPU) ---")

    log("Loading text_encoder...")
    models['text_encoder'] = CLIPTextModel.from_pretrained(path, subfolder='text_encoder')
    models['text_encoder'].to(dtype=weight_dtype)
    log(f"  {verify_device('text_encoder', models['text_encoder'])}")

    log("Loading image_encoder...")
    models['image_encoder'] = CLIPVisionModelWithProjection.from_pretrained(path, subfolder='image_encoder')
    models['image_encoder'].to(dtype=weight_dtype)
    log(f"  {verify_device('image_encoder', models['image_encoder'])}")

    log("Loading vae...")
    models['vae'] = AutoencoderKL.from_pretrained(path, subfolder='vae')
    models['vae'].to(dtype=weight_dtype)
    log(f"  {verify_device('vae', models['vae'])}")

    log("Loading unet...")
    models['unet'] = UNetMV2DConditionModel.from_pretrained_2d(path, subfolder='unet', local_crossattn=True, **kwargs)
    models['unet'].to(dtype=weight_dtype)
    log(f"  {verify_device('unet', models['unet'])}")

    log("Loading ref_unet...")
    models['ref_unet'] = UNetMV2DRefModel.from_pretrained_2d(path, subfolder='ref_unet', local_crossattn=True, **kwargs)
    models['ref_unet'].to(dtype=weight_dtype)
    log(f"  {verify_device('ref_unet', models['ref_unet'])}")

    gc.collect()
    log("")
    log(f"CPU RAM after load: {os.popen('free -g').read().strip().split(chr(10))[1]}")
    log(f"GPU before transfers: {gpu_status()}")
    log("")

    log("--- Moving models to GPU one by one ---")
    for name in ['text_encoder', 'image_encoder', 'vae', 'unet', 'ref_unet']:
        model = models[name]
        log(f"Moving {name} to GPU...")
        model.to(device)
        result = verify_device(name, model)
        log(f"  {result}")
        log(f"  GPU: {gpu_status()}")
        log("")

    log("--- Final verification ---")
    all_ok = True
    for name, model in models.items():
        result = verify_device(name, model)
        on_gpu = "GPU OK" in result
        if not on_gpu:
            all_ok = False
        log(f"  {result}")

    log(f"")
    log(f"GPU: {gpu_status()}")
    log(f"ALL ON GPU: {'YES' if all_ok else 'NO - PROBLEM DETECTED'}")

    log("")
    log("--- Quick inference test (1 dummy forward) ---")
    unet = models['unet']
    try:
        dummy_input = torch.randn(1, 4, 1, 128, 80, device=device, dtype=weight_dtype)
        dummy_t = torch.tensor([999], device=device)
        dummy_emb = torch.randn(1, 77, 768, device=device, dtype=torch.float32)
        log(f"  dummy input device: {dummy_input.device}")
        log(f"  GPU before forward: {gpu_status()}")

        with torch.no_grad():
            out = unet(dummy_input.to(torch.float32), dummy_t, encoder_hidden_states=dummy_emb).sample
        log(f"  output device: {out.device}, shape: {out.shape}")
        log(f"  GPU after forward: {gpu_status()}")
        log("  INFERENCE TEST: PASS")
    except Exception as e:
        log(f"  INFERENCE TEST: FAIL - {e}")

    log("")
    log("=== DIAGNOSTIC COMPLETE ===")


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__) or '.')
    main()
