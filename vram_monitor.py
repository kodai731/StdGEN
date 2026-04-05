import json
import os
import sys
import threading
import time
from datetime import UTC, datetime

import torch

LOG_DIR = os.path.join(os.path.dirname(__file__), "log", "gpu_health")
HEARTBEAT_PATH = os.path.join(os.path.dirname(__file__), "log", ".heartbeat")
INTERVAL_SEC = 1.0
HANG_TIMEOUT_SEC = 30.0

_monitor: "_VRAMMonitor | None" = None


class _VRAMMonitor:
    def __init__(self, interval: float = INTERVAL_SEC) -> None:
        self._interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._file = None
        self._lock = threading.Lock()
        self._stage = ""
        self._step = -1

    def start(self) -> None:
        if not torch.cuda.is_available():
            return
        os.makedirs(LOG_DIR, exist_ok=True)
        date_str = time.strftime("%Y-%m-%d", time.gmtime())
        path = os.path.join(LOG_DIR, f"{date_str}.jsonl")
        self._file = open(path, "a", encoding="utf-8")
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="vram-monitor",
        )
        self._thread.start()
        self._write_event("monitor_started", {
            "pid": os.getpid(),
            "script": os.path.basename(sys.argv[0]) if sys.argv else "unknown",
        })

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        if self._file:
            self._file.close()
            self._file = None

    def set_stage(self, stage: str, step: int = -1) -> None:
        self._stage = stage
        self._step = step

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._update_heartbeat()
            try:
                self._record_snapshot()
            except Exception:
                pass
            self._stop.wait(self._interval)

    def _update_heartbeat(self) -> None:
        try:
            free, total = torch.cuda.mem_get_info(0)
            os.makedirs(os.path.dirname(HEARTBEAT_PATH), exist_ok=True)
            with open(HEARTBEAT_PATH, "w") as f:
                f.write(json.dumps({
                    "ts": time.time(),
                    "pid": os.getpid(),
                    "stage": self._stage,
                    "step": self._step,
                    "vram_ratio": round(1.0 - free / total, 3) if total > 0 else 0.0,
                    "vram_used_mb": round((total - free) / (1024 * 1024), 1),
                    "vram_total_mb": round(total / (1024 * 1024), 1),
                }))
        except Exception:
            pass

    def _record_snapshot(self) -> None:
        free, total = torch.cuda.mem_get_info(0)
        allocated = torch.cuda.memory_allocated(0)
        reserved = torch.cuda.memory_reserved(0)

        record = {
            "vram_used_mb": round((total - free) / (1024 * 1024), 1),
            "vram_total_mb": round(total / (1024 * 1024), 1),
            "vram_ratio": round(1.0 - free / total, 3) if total > 0 else 0.0,
            "allocated_mb": round(allocated / (1024 * 1024), 1),
            "reserved_mb": round(reserved / (1024 * 1024), 1),
        }

        if self._stage:
            record["stage"] = self._stage
        if self._step >= 0:
            record["step"] = self._step

        self._write_event("gpu_health", record)

    def _write_event(self, event: str, extra: dict | None = None) -> None:
        if self._file is None:
            return
        record = {"ts": datetime.now(tz=UTC).isoformat(), "event": event}
        if extra:
            record.update(extra)
        with self._lock:
            self._file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            self._file.flush()
            os.fsync(self._file.fileno())


def init_log() -> None:
    global _monitor
    if _monitor is not None:
        return
    _monitor = _VRAMMonitor()
    _monitor.start()
    import atexit
    atexit.register(_stop_monitor)


def _stop_monitor() -> None:
    global _monitor
    if _monitor:
        _monitor.stop()
        _monitor = None
    try:
        os.remove(HEARTBEAT_PATH)
    except FileNotFoundError:
        pass


def set_stage(stage: str, step: int = -1) -> None:
    if _monitor:
        _monitor.set_stage(stage, step)


def log_vram(label: str) -> float:
    if not torch.cuda.is_available():
        return 0.0

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3

    msg = f"[VRAM] {label}: allocated={allocated:.2f}GB reserved={reserved:.2f}GB total={total:.1f}GB"
    print(msg, flush=True)

    if _monitor:
        _monitor._write_event("vram_snapshot", {
            "label": label,
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
            "total_gb": round(total, 1),
        })

    return allocated


def check_vram_before_load(model_name: str, model) -> float:
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    param_gb = param_bytes / 1024**3

    current = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
    projected = current + param_gb

    limit = 21.0
    msg = f"[VRAM] {model_name}: size={param_gb:.2f}GB current={current:.2f}GB projected={projected:.2f}GB limit={limit:.1f}GB"
    print(msg, flush=True)

    if _monitor:
        _monitor._write_event("vram_load_check", {
            "model": model_name,
            "size_gb": round(param_gb, 2),
            "current_gb": round(current, 2),
            "projected_gb": round(projected, 2),
        })

    if projected > limit:
        print(f"[VRAM] ERROR: Loading {model_name} would exceed limit ({projected:.2f}GB > {limit:.1f}GB)", flush=True)
        return -1.0

    return param_gb


def report_all_models(**models) -> float:
    total_size = 0.0
    details = {}
    for name, model in models.items():
        if model is None or not hasattr(model, "parameters"):
            continue
        size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3
        params = list(model.parameters())
        dev = str(params[0].device) if params else "N/A"
        total_size += size
        details[name] = {"size_gb": round(size, 2), "device": dev}
        print(f"  {name:20s}: {size:.2f} GB  ({dev})", flush=True)

    print(f"  {'TOTAL':20s}: {total_size:.2f} GB", flush=True)

    if _monitor:
        _monitor._write_event("model_report", {
            "models": details,
            "total_gb": round(total_size, 2),
        })

    return total_size
