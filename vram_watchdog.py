"""
GPU hang watchdog — separate process that monitors heartbeat file.

Launched automatically by vram_monitor.init_log().
Detects hangs by checking if the heartbeat timestamp stops updating.
Writes hang event to JSONL log and kills the target process.

Usage: python vram_watchdog.py <target_pid>
"""
import json
import os
import signal
import sys
import time
from datetime import UTC, datetime

LOG_DIR = os.path.join(os.path.dirname(__file__), "log", "gpu_health")
HEARTBEAT_PATH = os.path.join(os.path.dirname(__file__), "log", ".heartbeat")
HANG_TIMEOUT_SEC = 30.0
CHECK_INTERVAL_SEC = 2.0


def write_log(record: dict) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    date_str = time.strftime("%Y-%m-%d", time.gmtime())
    path = os.path.join(LOG_DIR, f"{date_str}.jsonl")
    record["ts"] = datetime.now(tz=UTC).isoformat()
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_heartbeat() -> dict | None:
    try:
        with open(HEARTBEAT_PATH) as f:
            return json.loads(f.read())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def is_process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def main(target_pid: int) -> None:
    write_log({
        "event": "watchdog_started",
        "target_pid": target_pid,
        "hang_timeout_sec": HANG_TIMEOUT_SEC,
    })

    while is_process_alive(target_pid):
        hb = read_heartbeat()
        if hb is None:
            time.sleep(CHECK_INTERVAL_SEC)
            continue

        hb_age = time.time() - hb["ts"]

        if hb_age > HANG_TIMEOUT_SEC:
            write_log({
                "event": "gpu_hang_detected",
                "target_pid": target_pid,
                "last_heartbeat_age_sec": round(hb_age, 1),
                "last_stage": hb.get("stage", ""),
                "last_step": hb.get("step", -1),
                "action": "SIGKILL",
            })

            print(
                f"[WATCHDOG] GPU HANG DETECTED: pid={target_pid} "
                f"stage={hb.get('stage','')} step={hb.get('step','')} "
                f"no heartbeat for {hb_age:.0f}s — sending SIGKILL",
                file=sys.stderr, flush=True,
            )

            try:
                os.kill(target_pid, signal.SIGKILL)
            except OSError:
                pass

            time.sleep(1)
            break

        time.sleep(CHECK_INTERVAL_SEC)

    write_log({
        "event": "watchdog_stopped",
        "target_pid": target_pid,
        "reason": "target_exited",
    })


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <target_pid>", file=sys.stderr)
        sys.exit(1)
    main(int(sys.argv[1]))
