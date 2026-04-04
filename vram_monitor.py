import torch
import sys
import os
from datetime import datetime


VRAM_LIMIT_GB = 24.0
VRAM_SAFETY_MARGIN_GB = 3.0
VRAM_MAX_ALLOWED_GB = VRAM_LIMIT_GB - VRAM_SAFETY_MARGIN_GB

LOG_PATH = os.path.join(os.path.dirname(__file__), "vram_log.txt")


def _write_log(msg: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def init_log():
    with open(LOG_PATH, "w") as f:
        f.write(f"VRAM Monitor Log - {datetime.now()}\n")
        f.write(f"Limit: {VRAM_MAX_ALLOWED_GB:.1f}GB (total {VRAM_LIMIT_GB:.0f}GB - {VRAM_SAFETY_MARGIN_GB:.0f}GB margin)\n\n")
        f.flush()
        os.fsync(f.fileno())


def log_vram(label: str):
    if not torch.cuda.is_available():
        _write_log(f"[VRAM] {label}: CUDA not available")
        return 0.0

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3

    _write_log(f"[VRAM] {label}: allocated={allocated:.2f}GB reserved={reserved:.2f}GB total={total:.1f}GB")
    return allocated


def check_vram_before_load(model_name: str, model) -> float:
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    param_gb = param_bytes / 1024**3

    current_allocated = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
    projected = current_allocated + param_gb

    _write_log(f"[VRAM] {model_name}: size={param_gb:.2f}GB current={current_allocated:.2f}GB projected={projected:.2f}GB limit={VRAM_MAX_ALLOWED_GB:.1f}GB")

    if projected > VRAM_MAX_ALLOWED_GB:
        _write_log(f"[VRAM] ERROR: Loading {model_name} would exceed VRAM limit ({projected:.2f}GB > {VRAM_MAX_ALLOWED_GB:.1f}GB)")
        return -1.0

    return param_gb


def model_size_gb(model) -> float:
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3


def report_all_models(**models):
    lines = []
    lines.append(f"{'='*60}")
    lines.append(f"  VRAM Budget Report (limit={VRAM_MAX_ALLOWED_GB:.1f}GB / total={VRAM_LIMIT_GB:.0f}GB)")
    lines.append(f"{'='*60}")

    total = 0.0
    for name, model in models.items():
        if model is None or not hasattr(model, 'parameters'):
            continue
        size = model_size_gb(model)
        params = list(model.parameters())
        dev = params[0].device if params else "N/A"
        total += size
        lines.append(f"  {name:20s}: {size:.2f} GB  ({dev})")

    lines.append(f"  {'TOTAL':20s}: {total:.2f} GB")
    lines.append(f"  {'Available':20s}: {VRAM_MAX_ALLOWED_GB:.1f} GB")

    if total > VRAM_MAX_ALLOWED_GB:
        lines.append(f"  OVER BUDGET by {total - VRAM_MAX_ALLOWED_GB:.2f} GB")
    else:
        lines.append(f"  Remaining: {VRAM_MAX_ALLOWED_GB - total:.2f} GB (for activations)")

    lines.append(f"{'='*60}")

    for line in lines:
        _write_log(line)

    return total
