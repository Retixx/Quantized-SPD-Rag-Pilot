"""CUDA verification, GPU-offload detection and peak memory sampling."""
from __future__ import annotations

import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


def nvidia_smi_available() -> bool:
    return shutil.which("nvidia-smi") is not None


def cuda_report() -> Dict[str, object]:
    """Structured `nvidia-smi` snapshot. Never raises."""
    rep: Dict[str, object] = {"nvidia_smi": nvidia_smi_available(), "gpus": [], "error": None}
    if not rep["nvidia_smi"]:
        rep["error"] = "nvidia-smi not found on PATH"
        return rep
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.total,memory.used,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=60, check=True,
        ).stdout
    except Exception as exc:  # pragma: no cover - environment dependent
        rep["error"] = f"{type(exc).__name__}: {exc}"
        return rep
    gpus: List[Dict[str, object]] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        gpus.append({
            "index": int(parts[0]), "name": parts[1],
            "memory_total_mib": float(parts[2]), "memory_used_mib": float(parts[3]),
            "driver_version": parts[4],
        })
    rep["gpus"] = gpus
    return rep


def require_cuda() -> Dict[str, object]:
    rep = cuda_report()
    if not rep["gpus"]:
        raise RuntimeError(
            "No CUDA GPU visible. On Kaggle set Settings -> Accelerator to a GPU. "
            f"nvidia-smi report: {rep}"
        )
    return rep


_OFFLOAD_PATTERNS = [
    re.compile(r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers to GPU", re.I),
    re.compile(r"offloading\s+(\d+)\s+repeating layers to GPU", re.I),
    re.compile(r"n_gpu_layers\s*[:=]\s*(\d+)", re.I),
]


def detect_gpu_offload(server_log: str) -> Dict[str, object]:
    """Parse a llama.cpp load log for evidence that weights went to the GPU."""
    info: Dict[str, object] = {
        "offload_detected": False, "layers_offloaded": None,
        "layers_total": None, "evidence": None,
    }
    m = _OFFLOAD_PATTERNS[0].search(server_log)
    if m:
        info.update(offload_detected=int(m.group(1)) > 0,
                    layers_offloaded=int(m.group(1)),
                    layers_total=int(m.group(2)), evidence=m.group(0))
        return info
    m = _OFFLOAD_PATTERNS[1].search(server_log)
    if m:
        info.update(offload_detected=int(m.group(1)) > 0,
                    layers_offloaded=int(m.group(1)), evidence=m.group(0))
        return info
    m = _OFFLOAD_PATTERNS[2].search(server_log)
    if m:
        info.update(offload_detected=int(m.group(1)) > 0,
                    layers_offloaded=int(m.group(1)), evidence=m.group(0))
    if "CUDA" in server_log and info["evidence"] is None:
        info["evidence"] = "CUDA mentioned in log but no offload line parsed"
    return info


@dataclass
class MemorySampler:
    """Background sampler for peak CPU RAM and peak GPU VRAM (MiB)."""

    interval_s: float = 1.0
    peak_cpu_rss_mib: float = 0.0
    peak_gpu_used_mib: float = 0.0
    samples: int = 0
    _stop: threading.Event = field(default_factory=threading.Event, repr=False)
    _thread: Optional[threading.Thread] = field(default=None, repr=False)

    def _read_cpu(self) -> float:
        try:
            import psutil  # noqa: WPS433
            return psutil.Process().memory_info().rss / (1024 ** 2)
        except Exception:
            try:
                with open("/proc/self/status", "r", encoding="utf-8") as fh:
                    for line in fh:
                        if line.startswith("VmRSS:"):
                            return float(line.split()[1]) / 1024.0
            except Exception:
                pass
        return 0.0

    def _read_gpu(self) -> float:
        if not nvidia_smi_available():
            return 0.0
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=20, check=True).stdout
            return max((float(x.strip()) for x in out.strip().splitlines() if x.strip()),
                       default=0.0)
        except Exception:
            return 0.0

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.peak_cpu_rss_mib = max(self.peak_cpu_rss_mib, self._read_cpu())
            self.peak_gpu_used_mib = max(self.peak_gpu_used_mib, self._read_gpu())
            self.samples += 1
            self._stop.wait(self.interval_s)

    def start(self) -> "MemorySampler":
        self.peak_cpu_rss_mib = max(self.peak_cpu_rss_mib, self._read_cpu())
        self.peak_gpu_used_mib = max(self.peak_gpu_used_mib, self._read_gpu())
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> Dict[str, float]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        return self.report()

    def report(self) -> Dict[str, float]:
        return {"peak_cpu_rss_mib": round(self.peak_cpu_rss_mib, 2),
                "peak_gpu_used_mib": round(self.peak_gpu_used_mib, 2),
                "samples": self.samples}

    def __enter__(self) -> "MemorySampler":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


def free_gpu_memory() -> None:
    """Best-effort VRAM release between precision blocks."""
    try:
        import gc
        gc.collect()
    except Exception:
        pass
    try:
        import torch  # noqa: WPS433
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    time.sleep(1.0)
