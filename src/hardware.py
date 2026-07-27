"""CUDA verification, GPU-offload detection and peak memory sampling.

Measurement invariants, because the systems numbers are part of the result:
  * `offload_detected` is set only by an OUTCOME line in the llama.cpp log --
    the echo of the requested `n_gpu_layers` is recorded separately, since a
    CPU-only build prints it too;
  * CPU RSS and VRAM are attributed to the model process where possible, and
    the harness-wide / device-wide numbers are kept as clearly named extras;
  * the sampler thread is always fully joined before a peak is reported, so a
    report is never torn.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


def nvidia_smi_available() -> bool:
    return shutil.which("nvidia-smi") is not None


def _nvidia_smi(query: str, timeout_s: float = 10.0) -> Optional[str]:
    """Run one `nvidia-smi --query-*` call. Returns None if it is unusable."""
    if not nvidia_smi_available():
        return None
    try:
        return subprocess.run(
            ["nvidia-smi", query, "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=timeout_s, check=True).stdout
    except Exception:
        return None


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


# -- GPU offload detection --------------------------------------------------
#
# Strong patterns describe an OUTCOME ("this many layers ended up on the GPU").
# Weak patterns are llama.cpp echoing back the parameter it was ASKED for; a
# CPU-only build prints `n_gpu_layers = 99` just as happily as a CUDA build, so
# matching one proves nothing about where the weights live.

_STRONG_OFFLOAD_PATTERNS = [
    re.compile(r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers to GPU", re.I),
    re.compile(r"offloading\s+(\d+)\s+repeating layers to GPU", re.I),
]
_WEAK_OFFLOAD_PATTERNS = [
    re.compile(r"n_gpu_layers\s*[:=]\s*(-?\d+)", re.I),
]


def detect_gpu_offload(server_log: str) -> Dict[str, object]:
    """Parse a llama.cpp load log for evidence that weights went to the GPU.

    `offload_detected` is an outcome claim and is only ever set from a strong
    pattern. A requested-but-unconfirmed `-ngl` is reported as
    `n_gpu_layers_requested` with `offload_evidence_strength="requested_only"`.
    """
    info: Dict[str, object] = {
        "offload_detected": False, "layers_offloaded": None,
        "layers_total": None, "n_gpu_layers_requested": None,
        "offload_evidence_strength": "none", "evidence": None,
    }

    weak = _WEAK_OFFLOAD_PATTERNS[0].search(server_log)
    if weak:
        info["n_gpu_layers_requested"] = int(weak.group(1))

    m = _STRONG_OFFLOAD_PATTERNS[0].search(server_log)
    if m:
        info.update(offload_detected=int(m.group(1)) > 0,
                    layers_offloaded=int(m.group(1)),
                    layers_total=int(m.group(2)),
                    offload_evidence_strength="offloaded_layer_count",
                    evidence=m.group(0))
        return info
    m = _STRONG_OFFLOAD_PATTERNS[1].search(server_log)
    if m:
        info.update(offload_detected=int(m.group(1)) > 0,
                    layers_offloaded=int(m.group(1)),
                    offload_evidence_strength="repeating_layer_count",
                    evidence=m.group(0))
        return info

    if weak:
        # Requested, not confirmed. Deliberately leaves offload_detected False.
        info.update(offload_evidence_strength="requested_only",
                    evidence=f"{weak.group(0)} (requested parameter echoed by "
                             "llama.cpp; printed even on a CPU-only build, so "
                             "not evidence that any layer was offloaded)")
        return info

    if "CUDA" in server_log:
        info["evidence"] = "CUDA mentioned in log but no offload line parsed"
    return info


# -- process attribution helpers -------------------------------------------


def _tree_pids(pid: Optional[int]) -> List[int]:
    """`pid` plus its descendants, so a forked worker is not missed."""
    if not pid:
        return []
    pids = [int(pid)]
    try:
        import psutil  # noqa: WPS433
        pids += [c.pid for c in psutil.Process(int(pid)).children(recursive=True)]
    except Exception:
        pass
    return pids


def _rss_mib(pid: Optional[int] = None) -> float:
    """RSS of one process in MiB (the current process when `pid` is None)."""
    try:
        import psutil  # noqa: WPS433
        return psutil.Process(pid).memory_info().rss / (1024 ** 2)
    except Exception:
        pass
    status = "/proc/self/status" if pid is None else f"/proc/{int(pid)}/status"
    try:
        with open(status, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0


def _tree_rss_mib(pid: Optional[int]) -> float:
    """Summed RSS of a process tree in MiB. 0.0 if the tree is gone."""
    return sum(_rss_mib(p) for p in _tree_pids(pid))


def gpu_device_used_mib(timeout_s: float = 10.0) -> Optional[float]:
    """Busiest device's total VRAM in use -- ALL processes, not just ours.

    Includes the embedder's CUDA context, co-tenants and residue that the
    driver has not reclaimed yet, so it is an upper bound on our own usage.
    """
    out = _nvidia_smi("--query-gpu=memory.used", timeout_s=timeout_s)
    if out is None:
        return None
    values = [float(x.strip()) for x in out.strip().splitlines() if x.strip()]
    return max(values, default=0.0)


def gpu_process_used_mib(pids: Sequence[int],
                         timeout_s: float = 10.0) -> Optional[float]:
    """VRAM attributed to `pids` by the driver, summed over devices.

    None means the driver could not tell us (no nvidia-smi, or the query is
    unsupported -- notably inside some containers and on WSL).
    """
    if not pids:
        return None
    out = _nvidia_smi("--query-compute-apps=pid,used_memory", timeout_s=timeout_s)
    if out is None:
        return None
    wanted = {int(p) for p in pids}
    total = 0.0
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            if int(parts[0]) in wanted:
                total += float(parts[1])
        except ValueError:
            continue
    return total


@dataclass
class MemorySampler:
    """Background sampler for peak CPU RAM and peak GPU VRAM (MiB).

    Set `pid` (or call `track_pid`) to the llama.cpp server process to get the
    numbers that actually matter: the model lives in that child, not in this
    python process, and its VRAM share is not the device total.
    """

    interval_s: float = 1.0
    pid: Optional[int] = None
    nvidia_smi_timeout_s: float = 10.0
    peak_harness_rss_mib: float = 0.0
    peak_model_rss_mib: float = 0.0
    peak_gpu_process_mib: float = 0.0
    peak_gpu_device_total_mib: float = 0.0
    gpu_process_query_ok: bool = False
    samples: int = 0
    complete: bool = True
    _stop: threading.Event = field(default_factory=threading.Event, repr=False)
    _thread: Optional[threading.Thread] = field(default=None, repr=False)

    # -- one sample ------------------------------------------------------
    def _sample(self) -> None:
        self.peak_harness_rss_mib = max(self.peak_harness_rss_mib, _rss_mib(None))
        pids = _tree_pids(self.pid)
        if pids:
            self.peak_model_rss_mib = max(self.peak_model_rss_mib,
                                          sum(_rss_mib(p) for p in pids))
            proc_vram = gpu_process_used_mib(pids, self.nvidia_smi_timeout_s)
            if proc_vram is not None:
                self.gpu_process_query_ok = True
                self.peak_gpu_process_mib = max(self.peak_gpu_process_mib, proc_vram)
        device = gpu_device_used_mib(self.nvidia_smi_timeout_s)
        if device is not None:
            self.peak_gpu_device_total_mib = max(self.peak_gpu_device_total_mib, device)
        self.samples += 1

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval_s)

    # -- lifecycle -------------------------------------------------------
    def start(self) -> "MemorySampler":
        self._sample()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def track_pid(self, pid: Optional[int]) -> "MemorySampler":
        """Attribute subsequent samples to `pid` and its children."""
        self.pid = int(pid) if pid else None
        return self

    def stop(self, join_timeout_s: Optional[float] = None) -> Dict[str, Any]:
        """Stop sampling and return a report that is never torn.

        The worker can be sitting inside an `nvidia-smi` call when the stop flag
        is set, so the join budget must exceed one worst-case iteration (two
        nvidia-smi calls plus the sleep). Joining for less than that used to let
        `report()` read a peak the thread was still writing.
        """
        self._stop.set()
        if self._thread is not None:
            budget = (join_timeout_s if join_timeout_s is not None
                      else 2.0 * self.nvidia_smi_timeout_s + self.interval_s + 5.0)
            self._thread.join(timeout=budget)
            self.complete = not self._thread.is_alive()
            if self.complete:
                self._thread = None
        return self.report()

    def report(self) -> Dict[str, Any]:
        model_rss = self.peak_model_rss_mib
        gpu_proc = self.peak_gpu_process_mib if self.gpu_process_query_ok else 0.0
        return {
            # primary numbers: the model process where we could attribute it
            "peak_cpu_rss_mib": round(model_rss or self.peak_harness_rss_mib, 2),
            "peak_gpu_used_mib": round(gpu_proc or self.peak_gpu_device_total_mib, 2),
            # attributed components, so the primaries are auditable
            "peak_model_rss_mib": round(model_rss, 2),
            "peak_harness_rss_mib": round(self.peak_harness_rss_mib, 2),
            "peak_gpu_process_mib": round(self.peak_gpu_process_mib, 2),
            "peak_gpu_device_total_mib": round(self.peak_gpu_device_total_mib, 2),
            "cpu_rss_source": "model_process" if model_rss else "harness_process",
            "gpu_used_source": "per_process" if gpu_proc else "device_total",
            "sampled_pid": self.pid,
            "samples": self.samples,
            "sampler_complete": self.complete,
        }

    def __enter__(self) -> "MemorySampler":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


def free_gpu_memory(threshold_mib: float = 1024.0, timeout_s: float = 60.0,
                    poll_s: float = 1.0, plateau_polls: int = 5,
                    plateau_delta_mib: float = 32.0) -> Dict[str, Any]:
    """Wait for VRAM to actually come back between precision blocks.

    `torch.cuda.empty_cache()` frees nothing held by a separate process, and a
    fixed sleep is not long enough for a CUDA context teardown, so this polls
    `nvidia-smi` until the device drops below `threshold_mib`, or stops dropping
    (a co-tenant can sit above any threshold forever), or the timeout elapses --
    and RETURNS what happened so the caller can record it per block.

    Note: importing torch and touching `torch.cuda` CREATES a CUDA context in
    this process, which itself makes VRAM asymmetric across precision blocks --
    so the cache drop is only attempted when torch is already imported.
    """
    result: Dict[str, Any] = {
        "nvidia_smi": nvidia_smi_available(), "threshold_mib": threshold_mib,
        "timeout_s": timeout_s, "start_mib": None, "final_mib": None,
        "waited_s": 0.0, "polls": 0, "released": False, "reason": "not_polled",
        "timed_out": False, "torch_cache_cleared": False, "note": None,
    }
    try:
        import gc
        gc.collect()
    except Exception:
        pass

    torch = sys.modules.get("torch")
    if torch is not None:  # already imported elsewhere: no new context created
        try:
            if torch.cuda.is_available() and torch.cuda.is_initialized():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                result["torch_cache_cleared"] = True
        except Exception:
            pass
    else:
        result["note"] = "torch not imported; no CUDA context created here"

    start = gpu_device_used_mib()
    result["start_mib"] = start
    if start is None:
        result["reason"] = "nvidia_smi_unavailable"
        result["note"] = "nvidia-smi unavailable; no VRAM release could be verified"
        return result

    t0 = time.time()
    used = start
    lowest = start
    stable = 0
    while True:
        sample = gpu_device_used_mib()
        result["polls"] += 1
        if sample is None:
            result["reason"] = "nvidia_smi_stopped_responding"
            result["note"] = "nvidia-smi stopped responding during the wait"
            break
        used = sample
        if used <= threshold_mib:
            result["released"] = True
            result["reason"] = "below_threshold"
            break
        # A CUDA context can take many seconds to tear down, but a co-tenant's
        # allocation never will: stop once the number stops falling.
        if lowest - used < plateau_delta_mib:
            stable += 1
        else:
            stable = 0
        lowest = min(lowest, used)
        if stable >= plateau_polls:
            freed = start - used
            result["released"] = freed >= plateau_delta_mib
            result["reason"] = ("plateau_after_release" if result["released"]
                                else "plateau_no_release")
            break
        if time.time() - t0 >= timeout_s:
            result["timed_out"] = True
            result["reason"] = "timeout"
            break
        time.sleep(poll_s)
    result["final_mib"] = used
    result["waited_s"] = round(time.time() - t0, 2)
    return result
