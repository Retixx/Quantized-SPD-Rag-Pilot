"""llama.cpp backends: llama-server (preferred), llama-cli (fallback), fixture.

Invariants enforced here, because the science depends on them:
  * one physical model per PRECISION BLOCK, never one per logical agent;
  * identical sampling params, context size, chat template and max tokens
    across precisions -- the sampling config is hashed and recorded;
  * every generation returns provenance (model sha256, backend, is_fixture);
  * a backend never returns its own prompt as if it were a generation --
    the evidence passages carry the gold facts, so an echoed prompt would
    score like a near-perfect answer;
  * the fixture backend is refused unless explicitly unlocked, and everything
    it produces is watermarked so the analysis layer can quarantine it.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import signal
import socket
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import urllib.error
import urllib.request

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hardware import MemorySampler, detect_gpu_offload  # noqa: E402
from logging_utils import sha256_file, sha256_text  # noqa: E402

FIXTURE_ENV = "SPDQ_ALLOW_FIXTURE"
FIXTURE_WATERMARK = "FIXTURE_NOT_REAL_MODEL_OUTPUT"

#: Default per-request timeout. Deliberately far below the old 1800s: a dead
#: server should fail a call in minutes, not stall a Kaggle session for half an
#: hour per question.
DEFAULT_REQUEST_TIMEOUT_S = 300


# --------------------------------------------------------------------------


@dataclass
class SamplingConfig:
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 1
    min_p: float = 0.0
    repeat_penalty: float = 1.0
    seed: int = 1234
    max_tokens: int = 512
    n_ctx: int = 8192

    def hash(self) -> str:
        return sha256_text(json.dumps(asdict(self), sort_keys=True))


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int = 0
    generated_tokens: int = 0
    latency_s: float = 0.0
    tokens_per_second: float = 0.0
    finish_reason: str = ""
    is_fixture: bool = False
    backend: str = ""
    model_sha256: str = ""
    precision: str = ""
    error: Optional[str] = None
    token_accounting: str = ""   # how the token counts above were obtained

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelSpec:
    precision: str            # "F16" | "Q8_0" | "Q4_K_M"
    path: str
    sha256: str = ""
    size_bytes: int = 0

    @classmethod
    def from_path(cls, precision: str, path: str) -> "ModelSpec":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"GGUF for {precision} not found: {path}")
        return cls(precision=precision, path=str(p),
                   sha256=sha256_file(p), size_bytes=p.stat().st_size)


# -- process-group helpers --------------------------------------------------
#
# Every llama.cpp child is launched with start_new_session=True, so it leads its
# own process group and can be signalled as a unit. `preexec_fn=os.setsid` would
# do the same thing, but it runs arbitrary python between fork() and exec() in a
# process that already has a live MemorySampler thread -- the classic
# fork-with-threads deadlock. start_new_session performs the setsid() inside
# CPython's async-signal-safe C helper instead.

_SIGKILL = getattr(signal, "SIGKILL", signal.SIGTERM)


def _signal_process_group(proc: subprocess.Popen, sig: int) -> None:
    """Signal the child's whole process group, falling back to the leader."""
    try:
        if hasattr(os, "killpg"):
            os.killpg(os.getpgid(proc.pid), sig)
            return
    except Exception:
        pass
    try:
        if sig == _SIGKILL:
            proc.kill()
        else:
            proc.terminate()
    except Exception:
        pass


def _reap_process_group(proc: subprocess.Popen, term_timeout_s: float = 60.0,
                        kill_timeout_s: float = 30.0) -> str:
    """SIGTERM then SIGKILL the whole group, waiting after EACH signal.

    Returns a status string. `UNREAPED_*` means a survivor is still holding the
    port and the VRAM, and the caller must keep tracking it.
    """
    if proc.poll() is not None:
        return "already_exited"
    _signal_process_group(proc, signal.SIGTERM)
    try:
        proc.wait(timeout=term_timeout_s)
        return "reaped_sigterm"
    except Exception:
        pass
    _signal_process_group(proc, _SIGKILL)
    try:
        proc.wait(timeout=kill_timeout_s)
        return "reaped_sigkill"
    except Exception:
        return f"UNREAPED_pid_{proc.pid}"


_ADDRESS_IN_USE = re.compile(
    r"address already in use|addrinuse|failed to bind|couldn't bind|"
    r"error while binding|bind\(\) failed", re.I)


def _is_address_in_use(text: str) -> bool:
    return bool(_ADDRESS_IN_USE.search(text or ""))


# -- echoed-prompt handling -------------------------------------------------

_ECHO_PROBE_CHARS = 64      # prefix used to decide "did it echo at all?"
_LEAK_WINDOW_CHARS = 200    # contiguous prompt span that must never survive
#: ChatML turn marker. Only the prompt contains it, so finding one in stdout is
#: proof that prompt text (or a runaway extra turn) is still in the output.
CHATML_TURN_MARKER = "<|im_start|>"


def _normalise_with_map(text: str) -> Tuple[str, List[int]]:
    """Whitespace-collapsed copy plus a per-character index back into `text`."""
    out: List[str] = []
    idx: List[int] = []
    prev_space = True
    for i, ch in enumerate(text):
        if ch.isspace():
            if prev_space:
                continue
            out.append(" ")
            idx.append(i)
            prev_space = True
        else:
            out.append(ch)
            idx.append(i)
            prev_space = False
    if out and out[-1] == " ":
        out.pop()
        idx.pop()
    return "".join(out), idx


@dataclass
class _EchoStrip:
    """Outcome of removing an echoed prompt from a subprocess's stdout."""

    text: str = ""
    method: str = ""
    error: Optional[str] = None


def _strip_echoed_prompt(raw: str, rendered: str,
                         echo_markers: Sequence[str] = ()) -> _EchoStrip:
    """Remove the echoed prompt from llama-cli stdout, or fail loudly.

    Returning the prompt as a generation is the worst failure mode in this file:
    the prompt contains the evidence passages, so it would be scored as an
    almost perfect answer. Anything short of a confident strip is an error.

    `echo_markers` are chat-template control strings that only the prompt can
    contain (e.g. ``<|im_start|>``); one surviving in stdout means the echo is
    present but could not be located, which is a hard failure.
    """
    if not rendered:
        return _EchoStrip(text=raw, method="no_prompt")

    if rendered in raw:
        return _EchoStrip(text=raw.split(rendered, 1)[1], method="exact")

    # llama-cli re-wraps and re-spaces what it echoes, so try again on a
    # whitespace-normalised view and map the split point back to the raw text.
    norm_raw, idx_raw = _normalise_with_map(raw)
    norm_prompt, _ = _normalise_with_map(rendered)
    if norm_prompt and norm_prompt in norm_raw:
        end = norm_raw.find(norm_prompt) + len(norm_prompt)
        start = idx_raw[end] if end < len(idx_raw) else len(raw)
        return _EchoStrip(text=raw[start:], method="whitespace_normalised")

    marker = next((m for m in echo_markers if m and m in rendered and m in raw), None)
    if marker:
        return _EchoStrip(
            error=f"prompt_echo_strip_failed: chat-template marker {marker!r} "
                  "survives in stdout, so the prompt was echoed but could not be "
                  "located; refusing to return prompt text as a generation")

    probe = norm_prompt[:_ECHO_PROBE_CHARS]
    if probe and norm_raw.startswith(probe):
        return _EchoStrip(
            error="prompt_echo_strip_failed: stdout begins with the prompt but the "
                  "echo could not be located exactly or after whitespace "
                  "normalisation; refusing to return the prompt as a generation")

    # Last resort: probe a few fixed offsets for a long contiguous slab of the
    # prompt. A quoted fact is short; 200 unbroken characters is an echo.
    if len(norm_prompt) >= _LEAK_WINDOW_CHARS:
        span = len(norm_prompt) - _LEAK_WINDOW_CHARS
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            offset = int(span * fraction)
            if norm_prompt[offset:offset + _LEAK_WINDOW_CHARS] in norm_raw:
                return _EchoStrip(
                    error="prompt_leak_detected: a long contiguous span of the "
                          "prompt survives in stdout; refusing to score prompt "
                          "text as output")

    # No echo at all (llama-cli can be quiet about the prompt) -- stdout is the
    # generation, and it demonstrably does not carry the prompt.
    return _EchoStrip(text=raw, method="no_echo")


# -- llama-cli timing parsing ----------------------------------------------

_CLI_PROMPT_TIMING = re.compile(
    r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens", re.I)
_CLI_EVAL_TIMING = re.compile(
    r"(?<!prompt )eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*(?:runs|tokens)", re.I)


def _parse_cli_timings(stderr: str) -> Dict[str, float]:
    """Token counts and times from llama-cli's `llama_perf_*` timing block."""
    out: Dict[str, float] = {}
    m = _CLI_PROMPT_TIMING.search(stderr or "")
    if m:
        out["prompt_ms"] = float(m.group(1))
        out["prompt_tokens"] = float(int(m.group(2)))
    m = _CLI_EVAL_TIMING.search(stderr or "")
    if m:
        out["eval_ms"] = float(m.group(1))
        out["generated_tokens"] = float(int(m.group(2)))
    return out


def _estimate_tokens(text: str) -> int:
    """Crude ~4-chars-per-token estimate, used only when timings are absent."""
    if not text:
        return 0
    return max(1, len(text) // 4)


# --------------------------------------------------------------------------


class Backend:
    name = "base"
    is_fixture = False

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def warmup(self, max_tokens: int = 8) -> Dict[str, Any]:
        """One short throwaway generation, discarded.

        First-touch page faults, kernel selection and CUDA graph capture are
        real costs that otherwise land entirely on the first question of the
        block and pollute its latency and tokens/s. Never raises.
        """
        return {"performed": False, "backend": self.name,
                "reason": "backend does not support warmup"}

    def chat(self, messages: List[Dict[str, str]],
             max_tokens: Optional[int] = None) -> GenerationResult:
        raise NotImplementedError

    def provenance(self) -> Dict[str, Any]:
        return {"backend": self.name, "is_fixture": self.is_fixture}

    def __enter__(self) -> "Backend":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


def _free_port(preferred: int = 0) -> int:
    """Pick a port that was free a moment ago.

    Inherently TOCTOU: the socket is closed before llama-server binds, so a
    racing process can take the port in between. `LlamaServerBackend.start()`
    therefore retries the whole bind+launch on an address-in-use exit.
    """
    if preferred:
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", preferred))
                return preferred
            except OSError:
                pass
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class LlamaServerBackend(Backend):
    """Persistent `llama-server` OpenAI-compatible endpoint. One per precision."""

    name = "llama-server"

    def __init__(self, model: ModelSpec, sampling: SamplingConfig,
                 binary: str = "llama-server", n_gpu_layers: int = 99,
                 host: str = "127.0.0.1", port: int = 0,
                 log_dir: str = "results/server_logs", startup_timeout_s: int = 900,
                 extra_args: Optional[List[str]] = None, threads: Optional[int] = None,
                 request_timeout_s: int = DEFAULT_REQUEST_TIMEOUT_S,
                 port_retries: int = 3, term_timeout_s: float = 60.0,
                 kill_timeout_s: float = 30.0, warmup_on_start: bool = True,
                 warmup_max_tokens: int = 8):
        self.model = model
        self.sampling = sampling
        self.binary = binary
        self.n_gpu_layers = n_gpu_layers
        self.host = host
        self.port = port
        self.startup_timeout_s = startup_timeout_s
        self.extra_args = list(extra_args or [])
        self.threads = threads
        self.request_timeout_s = int(request_timeout_s)
        self.port_retries = max(1, int(port_retries))
        self.term_timeout_s = term_timeout_s
        self.kill_timeout_s = kill_timeout_s
        self.warmup_on_start = warmup_on_start
        self.warmup_max_tokens = warmup_max_tokens
        self.log_path = Path(log_dir) / f"llama-server.{model.precision}.log"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._proc: Optional[subprocess.Popen] = None
        self._log_fh = None
        self.load_time_s: float = 0.0
        self.offload: Dict[str, Any] = {}
        # Populated by stop() and never cleared afterwards, so provenance()
        # still carries the peaks whether it is called before or after stop().
        self.memory: Dict[str, Any] = {}
        self.memory_is_final: bool = False
        self.warmup_info: Dict[str, Any] = {}
        self.stop_status: str = "not_started"
        self.start_attempts: int = 0
        self._sampler: Optional[MemorySampler] = None
        self.command: List[str] = []

    # -- lifecycle -------------------------------------------------------
    def start(self) -> None:
        if shutil.which(self.binary) is None and not Path(self.binary).exists():
            raise FileNotFoundError(
                f"llama-server binary not found: {self.binary!r}. Build llama.cpp "
                "first (scripts/prepare_models.py --build-llama-cpp).")
        requested_port = self.port
        for attempt in range(1, self.port_retries + 1):
            self.start_attempts = attempt
            # Only the first attempt honours an explicitly requested port; a
            # retry always takes a fresh ephemeral one.
            self.port = _free_port(requested_port if attempt == 1 else 0)
            try:
                self._launch(log_mode="w" if attempt == 1 else "a")
                return
            except Exception as exc:
                tail = self.read_log()[-4000:]
                self._abandon_attempt()
                if attempt < self.port_retries and _is_address_in_use(f"{exc}\n{tail}"):
                    print(f"[backend] port {self.port} was taken between probe and "
                          f"bind; retrying ({attempt}/{self.port_retries})")
                    time.sleep(1.0)
                    continue
                raise

    def _launch(self, log_mode: str = "w") -> None:
        self.command = [
            self.binary,
            "-m", self.model.path,
            "-c", str(self.sampling.n_ctx),
            "-ngl", str(self.n_gpu_layers),
            "--host", self.host,
            "--port", str(self.port),
            "--parallel", "1",
            "--seed", str(self.sampling.seed),
            "--no-webui",
        ]
        if self.threads:
            self.command += ["-t", str(self.threads)]
        self.command += self.extra_args
        self._sampler = MemorySampler().start()
        self._log_fh = open(self.log_path, log_mode, encoding="utf-8", errors="replace")
        t0 = time.time()
        self._proc = subprocess.Popen(
            self.command, stdout=self._log_fh, stderr=subprocess.STDOUT,
            start_new_session=True)
        # Attribute RSS and VRAM to the process that actually holds the model.
        self._sampler.track_pid(self._proc.pid)
        self._wait_healthy(t0)
        self.load_time_s = time.time() - t0
        self.offload = detect_gpu_offload(self.read_log())
        self.stop_status = "running"
        if self.warmup_on_start:
            self.warmup(self.warmup_max_tokens)

    def _abandon_attempt(self) -> None:
        """Tear a failed launch down completely before retrying or re-raising."""
        if self._proc is not None:
            status = _reap_process_group(self._proc, self.term_timeout_s,
                                         self.kill_timeout_s)
            self.stop_status = status
            if self._proc.poll() is not None:
                self._proc = None
        if self._sampler is not None:
            try:
                self._sampler.stop()
            except Exception:
                pass
            self._sampler = None
        self._close_log()

    def _close_log(self) -> None:
        if self._log_fh is not None:
            try:
                self._log_fh.flush()
                self._log_fh.close()
            except Exception:
                pass
            self._log_fh = None

    def _wait_healthy(self, t0: float) -> None:
        url = f"http://{self.host}:{self.port}/health"
        deadline = t0 + self.startup_timeout_s
        while time.time() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited with code {self._proc.returncode} during "
                    f"startup. Tail of log:\n{self.read_log()[-4000:]}")
            try:
                with urllib.request.urlopen(url, timeout=5) as resp:
                    if resp.status == 200:
                        return
            except Exception:
                time.sleep(1.0)
        raise TimeoutError(
            f"llama-server did not become healthy within {self.startup_timeout_s}s. "
            f"Tail of log:\n{self.read_log()[-4000:]}")

    def warmup(self, max_tokens: int = 8) -> Dict[str, Any]:
        """One throwaway generation against the healthy server; text discarded."""
        t0 = time.perf_counter()
        res = self.chat([{"role": "user", "content": "ping"}], max_tokens=max_tokens)
        self.warmup_info = {
            "performed": True, "backend": self.name,
            "latency_s": round(time.perf_counter() - t0, 3),
            "generated_tokens": res.generated_tokens,
            "discarded_chars": len(res.text or ""),
            "error": res.error,
        }
        return self.warmup_info

    def stop(self) -> None:
        if self._sampler is not None:
            report = self._sampler.stop()
            if report:
                self.memory = report
                self.memory_is_final = True
            self._sampler = None
        proc = self._proc
        if proc is not None:
            self.stop_status = _reap_process_group(proc, self.term_timeout_s,
                                                   self.kill_timeout_s)
            # Only forget the handle once the process is genuinely reaped; an
            # untracked survivor still holds the port and the VRAM.
            if proc.poll() is not None:
                self._proc = None
        self._close_log()

    def read_log(self) -> str:
        try:
            return self.log_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""

    # -- inference -------------------------------------------------------
    def chat(self, messages: List[Dict[str, str]],
             max_tokens: Optional[int] = None) -> GenerationResult:
        payload = {
            "messages": messages,
            "temperature": self.sampling.temperature,
            "top_p": self.sampling.top_p,
            "top_k": self.sampling.top_k,
            "min_p": self.sampling.min_p,
            "repeat_penalty": self.sampling.repeat_penalty,
            "seed": self.sampling.seed,
            "max_tokens": int(max_tokens or self.sampling.max_tokens),
            "stream": False,
            "cache_prompt": False,
        }
        req = urllib.request.Request(
            f"http://{self.host}:{self.port}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.request_timeout_s) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            return GenerationResult(
                text="", latency_s=time.perf_counter() - t0, backend=self.name,
                model_sha256=self.model.sha256, precision=self.model.precision,
                error=f"{type(exc).__name__}: {exc}")
        dt = time.perf_counter() - t0
        choice = (body.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""
        usage = body.get("usage") or {}
        gen = int(usage.get("completion_tokens") or 0)
        return GenerationResult(
            text=text, prompt_tokens=int(usage.get("prompt_tokens") or 0),
            generated_tokens=gen, latency_s=dt,
            tokens_per_second=(gen / dt if dt > 0 else 0.0),
            finish_reason=str(choice.get("finish_reason") or ""),
            backend=self.name, model_sha256=self.model.sha256,
            precision=self.model.precision,
            token_accounting="server_usage")

    def provenance(self) -> Dict[str, Any]:
        # Survives either call order: final peaks once stop() has run, a live
        # snapshot while the block is still in flight.
        memory = self.memory
        if not memory and self._sampler is not None:
            memory = self._sampler.report()
        return {
            "backend": self.name, "is_fixture": False,
            "command": self.command, "model": asdict(self.model),
            "sampling": asdict(self.sampling),
            "sampling_hash": self.sampling.hash(),
            "load_time_s": round(self.load_time_s, 3),
            "gpu_offload": self.offload, "memory": memory,
            "memory_is_final": self.memory_is_final,
            "warmup": self.warmup_info,
            "chat_template": {"source": "gguf_embedded",
                              "applied_by": "llama-server /v1/chat/completions"},
            "request_timeout_s": self.request_timeout_s,
            "port": self.port, "start_attempts": self.start_attempts,
            "stop_status": self.stop_status,
            "server_log": str(self.log_path),
        }


class LlamaCliBackend(Backend):
    """`llama-cli` subprocess fallback. Same params, one process per call."""

    name = "llama-cli"

    def __init__(self, model: ModelSpec, sampling: SamplingConfig,
                 binary: str = "llama-cli", n_gpu_layers: int = 99,
                 log_dir: str = "results/server_logs",
                 extra_args: Optional[List[str]] = None,
                 request_timeout_s: int = DEFAULT_REQUEST_TIMEOUT_S,
                 chat_template: str = "chatml",
                 warmup_on_start: bool = True, warmup_max_tokens: int = 8):
        self.model = model
        self.sampling = sampling
        self.binary = binary
        self.n_gpu_layers = n_gpu_layers
        self.extra_args = list(extra_args or [])
        self.request_timeout_s = int(request_timeout_s)
        self.chat_template = chat_template
        self.warmup_on_start = warmup_on_start
        self.warmup_max_tokens = warmup_max_tokens
        self.log_path = Path(log_dir) / f"llama-cli.{model.precision}.log"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.offload: Dict[str, Any] = {}
        self.load_time_s = 0.0
        self.memory: Dict[str, Any] = {}
        self.memory_is_final: bool = False
        self.warmup_info: Dict[str, Any] = {}
        self.last_prompt_sha256: str = ""
        self._sampler: Optional[MemorySampler] = None
        self.command: List[str] = []

    def start(self) -> None:
        if shutil.which(self.binary) is None and not Path(self.binary).exists():
            raise FileNotFoundError(f"llama-cli binary not found: {self.binary!r}")
        # Same accounting as the server backend: one sampler for the whole
        # precision block, re-pointed at each llama-cli child as it runs.
        self._sampler = MemorySampler().start()
        if self.warmup_on_start:
            self.warmup(self.warmup_max_tokens)

    def stop(self) -> None:
        if self._sampler is not None:
            report = self._sampler.stop()
            if report:
                self.memory = report
                self.memory_is_final = True
            self._sampler = None

    def warmup(self, max_tokens: int = 8) -> Dict[str, Any]:
        """One throwaway llama-cli generation; result discarded.

        Each call is a fresh process, so this cannot warm a resident model --
        it warms the page cache for the GGUF and the CUDA kernel/JIT caches,
        which are otherwise charged to the first real question.
        """
        t0 = time.perf_counter()
        res = self.chat([{"role": "user", "content": "ping"}], max_tokens=max_tokens)
        self.warmup_info = {
            "performed": True, "backend": self.name,
            "latency_s": round(time.perf_counter() - t0, 3),
            "generated_tokens": res.generated_tokens,
            "discarded_chars": len(res.text or ""),
            "error": res.error,
        }
        return self.warmup_info

    # -- prompt rendering ------------------------------------------------
    def _render(self, messages: List[Dict[str, str]]) -> str:
        """Hand-rolled ChatML, because llama-cli is fed a flat prompt via -f.

        This is NOT the GGUF's embedded template: llama-server applies that one
        itself, so the two backends can send different token sequences for the
        same messages. llama-cli's `--jinja` / `--chat-template` only take
        effect in conversation mode, which cannot accept a pre-built multi-turn
        transcript, so the divergence is recorded in provenance() instead of
        being papered over.
        """
        return "\n".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>"
                         for m in messages) + "\n<|im_start|>assistant\n"

    def _chat_template_provenance(self) -> Dict[str, Any]:
        return {
            "source": "hand_rolled",
            "template": self.chat_template,
            "applied_by": "LlamaCliBackend._render",
            "matches_server_template": False,
            "last_prompt_sha256": self.last_prompt_sha256,
            "note": ("llama-cli receives a pre-rendered prompt via -f, so the "
                     "GGUF's embedded chat template is bypassed. Do not mix "
                     "llama-cli and llama-server results within one comparison."),
        }

    # -- inference -------------------------------------------------------
    def _run(self, command: List[str]) -> subprocess.CompletedProcess:
        """Run one llama-cli call, attributing memory to it and killing its group."""
        proc = subprocess.Popen(command, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True,
                                start_new_session=True)
        if self._sampler is not None:
            self._sampler.track_pid(proc.pid)
        try:
            try:
                out, err = proc.communicate(timeout=self.request_timeout_s)
            except subprocess.TimeoutExpired:
                _reap_process_group(proc, term_timeout_s=10.0, kill_timeout_s=10.0)
                try:
                    out, err = proc.communicate(timeout=10)
                except Exception:
                    out, err = "", ""
                raise subprocess.TimeoutExpired(
                    command, self.request_timeout_s, output=out, stderr=err)
        finally:
            if self._sampler is not None:
                self._sampler.track_pid(None)
        return subprocess.CompletedProcess(command, proc.returncode, out, err)

    def chat(self, messages: List[Dict[str, str]],
             max_tokens: Optional[int] = None) -> GenerationResult:
        rendered = self._render(messages)
        self.last_prompt_sha256 = sha256_text(rendered)
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                         encoding="utf-8") as fh:
            fh.write(rendered)
            msg_file = fh.name
        # Every sampling parameter the server path sends must appear here too,
        # or sampling_hash would claim a parity the two backends do not have.
        self.command = [
            self.binary, "-m", self.model.path, "-c", str(self.sampling.n_ctx),
            "-ngl", str(self.n_gpu_layers), "--temp", str(self.sampling.temperature),
            "--top-p", str(self.sampling.top_p), "--top-k", str(self.sampling.top_k),
            "--min-p", str(self.sampling.min_p),
            "--repeat-penalty", str(self.sampling.repeat_penalty),
            "--seed", str(self.sampling.seed),
            "-n", str(int(max_tokens or self.sampling.max_tokens)),
            "--no-warmup", "-no-cnv", "--simple-io",
            "-f", msg_file,
        ] + self.extra_args
        t0 = time.perf_counter()
        try:
            proc = self._run(self.command)
        except Exception as exc:
            return GenerationResult(text="", latency_s=time.perf_counter() - t0,
                                    backend=self.name, model_sha256=self.model.sha256,
                                    precision=self.model.precision,
                                    error=f"{type(exc).__name__}: {exc}")
        finally:
            try:
                os.unlink(msg_file)
            except OSError:
                pass
        dt = time.perf_counter() - t0
        stderr = proc.stderr or ""
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(stderr)
        self.offload = detect_gpu_offload(stderr)
        exit_error = None if proc.returncode == 0 else f"exit_code={proc.returncode}"

        strip = _strip_echoed_prompt(proc.stdout or "", rendered,
                                     echo_markers=(CHATML_TURN_MARKER,))
        if strip.error is not None:
            # Hard failure: an unstripped prompt carries the gold facts and would
            # be scored as an answer. Return nothing rather than the prompt.
            return GenerationResult(
                text="", latency_s=dt, backend=self.name,
                model_sha256=self.model.sha256, precision=self.model.precision,
                token_accounting="none_generation_rejected",
                error="; ".join(x for x in (strip.error, exit_error) if x))

        text = strip.text.strip()
        timings = _parse_cli_timings(stderr)
        notes = [f"prompt_strip={strip.method}"]
        if "prompt_tokens" in timings:
            prompt_tokens = int(timings["prompt_tokens"])
            notes.append("prompt_tokens=llama_cli_timings")
        else:
            prompt_tokens = _estimate_tokens(rendered)
            notes.append("prompt_tokens=estimated_chars_div_4")
        if "generated_tokens" in timings:
            gen_tokens = int(timings["generated_tokens"])
            notes.append("generated_tokens=llama_cli_timings")
        else:
            gen_tokens = _estimate_tokens(text)
            notes.append("generated_tokens=estimated_chars_div_4")
        eval_s = float(timings.get("eval_ms", 0.0)) / 1000.0
        if eval_s > 0:
            tps = gen_tokens / eval_s
            notes.append("tokens_per_second=eval_time")
        else:
            tps = gen_tokens / dt if dt > 0 else 0.0
            notes.append("tokens_per_second=wall_time")
        return GenerationResult(text=text, prompt_tokens=prompt_tokens,
                                generated_tokens=gen_tokens, latency_s=dt,
                                tokens_per_second=tps,
                                backend=self.name,
                                model_sha256=self.model.sha256,
                                precision=self.model.precision,
                                token_accounting=";".join(notes),
                                error=exit_error)

    def provenance(self) -> Dict[str, Any]:
        memory = self.memory
        if not memory and self._sampler is not None:
            memory = self._sampler.report()
        return {"backend": self.name, "is_fixture": False,
                "command": self.command, "model": asdict(self.model),
                "sampling": asdict(self.sampling),
                "sampling_hash": self.sampling.hash(),
                "gpu_offload": self.offload,
                "memory": memory, "memory_is_final": self.memory_is_final,
                "warmup": self.warmup_info,
                "chat_template": self._chat_template_provenance(),
                "request_timeout_s": self.request_timeout_s}


class FixtureBackend(Backend):
    """Deterministic canned outputs. ENGINEERING TESTS ONLY.

    Refuses to start unless SPDQ_ALLOW_FIXTURE=1. Every result carries
    is_fixture=True and the response text carries a visible watermark, so
    `src/gates.py` can hard-block any scientific recommendation.
    """

    name = "fixture"
    is_fixture = True

    def __init__(self, sampling: Optional[SamplingConfig] = None,
                 precision: str = "FIXTURE"):
        self.sampling = sampling or SamplingConfig()
        self.precision = precision
        self.calls = 0

    def start(self) -> None:
        if os.environ.get(FIXTURE_ENV) != "1":
            raise PermissionError(
                "FixtureBackend is disabled by default. It exists only to test the "
                f"harness. Set {FIXTURE_ENV}=1 to unlock it; any run that uses it is "
                "watermarked and can never produce a scientific GO/NO-GO.")

    def warmup(self, max_tokens: int = 8) -> Dict[str, Any]:
        """No-op: there is no model to warm and no call must be counted."""
        return {"performed": False, "backend": self.name,
                "reason": "fixture backend has no model to warm"}

    def chat(self, messages: List[Dict[str, str]],
             max_tokens: Optional[int] = None) -> GenerationResult:
        self.calls += 1
        prompt = json.dumps(messages, sort_keys=True)
        role = "unknown"
        if "synthesis_directive" in prompt and "shared_tasks" in prompt:
            role = "coordinator"
        elif "final_answer" in prompt:
            role = "synthesizer"
        elif "supported_facts" in prompt:
            role = "document_agent"
        if role == "coordinator":
            body = {"shared_tasks": [
                {"task_id": "t1", "instruction": f"{FIXTURE_WATERMARK}: extract the key entity",
                 "required_fields": ["entity"]},
                {"task_id": "t2", "instruction": f"{FIXTURE_WATERMARK}: extract the key claim",
                 "required_fields": ["claim"]}],
                "synthesis_directive": f"{FIXTURE_WATERMARK}: merge findings."}
        elif role == "synthesizer":
            body = {"final_answer": f"{FIXTURE_WATERMARK}", "used_document_ids": [],
                    "used_fact_ids": [], "unresolved_conflicts": []}
        else:
            body = {"question_id": "", "document_id": "", "search_queries": [],
                    "retrieved_chunk_ids": [],
                    "supported_facts": [{"fact": FIXTURE_WATERMARK, "chunk_ids": []}],
                    "insufficient_evidence": False}
        text = json.dumps(body)
        return GenerationResult(text=text, prompt_tokens=len(prompt) // 4,
                                generated_tokens=len(text) // 4, latency_s=0.001,
                                tokens_per_second=0.0, finish_reason="stop",
                                is_fixture=True, backend=self.name,
                                model_sha256="FIXTURE", precision=self.precision,
                                token_accounting="fixture_chars_div_4")

    def provenance(self) -> Dict[str, Any]:
        return {"backend": self.name, "is_fixture": True,
                "watermark": FIXTURE_WATERMARK,
                "sampling": asdict(self.sampling),
                "sampling_hash": self.sampling.hash(),
                "model": {"precision": self.precision, "path": "FIXTURE",
                          "sha256": "FIXTURE", "size_bytes": 0}}


def build_backend(kind: str, model: Optional[ModelSpec], sampling: SamplingConfig,
                  **kwargs) -> Backend:
    kind = (kind or "llama-server").lower()
    if kind in ("fixture", "mock"):
        return FixtureBackend(sampling=sampling,
                              precision=model.precision if model else "FIXTURE")
    if model is None:
        raise ValueError(f"backend {kind!r} requires a ModelSpec")
    if kind in ("llama-server", "server"):
        return LlamaServerBackend(model=model, sampling=sampling, **kwargs)
    if kind in ("llama-cli", "cli"):
        cli_kwargs = {k: v for k, v in kwargs.items()
                      if k in ("binary", "n_gpu_layers", "log_dir", "extra_args",
                               "request_timeout_s", "chat_template",
                               "warmup_on_start", "warmup_max_tokens")}
        return LlamaCliBackend(model=model, sampling=sampling, **cli_kwargs)
    raise ValueError(f"unknown backend {kind!r}")
