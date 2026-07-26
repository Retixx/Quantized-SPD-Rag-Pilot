"""llama.cpp backends: llama-server (preferred), llama-cli (fallback), fixture.

Invariants enforced here, because the science depends on them:
  * one physical model per PRECISION BLOCK, never one per logical agent;
  * identical sampling params, context size, chat template and max tokens
    across precisions -- the sampling config is hashed and recorded;
  * every generation returns provenance (model sha256, backend, is_fixture);
  * the fixture backend is refused unless explicitly unlocked, and everything
    it produces is watermarked so the analysis layer can quarantine it.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import urllib.error
import urllib.request

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hardware import MemorySampler, detect_gpu_offload  # noqa: E402
from logging_utils import sha256_file, sha256_text  # noqa: E402

FIXTURE_ENV = "SPDQ_ALLOW_FIXTURE"
FIXTURE_WATERMARK = "FIXTURE_NOT_REAL_MODEL_OUTPUT"


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


# --------------------------------------------------------------------------


class Backend:
    name = "base"
    is_fixture = False

    def start(self) -> None: ...

    def stop(self) -> None: ...

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
                 extra_args: Optional[List[str]] = None, threads: Optional[int] = None):
        self.model = model
        self.sampling = sampling
        self.binary = binary
        self.n_gpu_layers = n_gpu_layers
        self.host = host
        self.port = port
        self.startup_timeout_s = startup_timeout_s
        self.extra_args = list(extra_args or [])
        self.threads = threads
        self.log_path = Path(log_dir) / f"llama-server.{model.precision}.log"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._proc: Optional[subprocess.Popen] = None
        self._log_fh = None
        self.load_time_s: float = 0.0
        self.offload: Dict[str, Any] = {}
        self.memory: Dict[str, float] = {}
        self._sampler: Optional[MemorySampler] = None
        self.command: List[str] = []

    # -- lifecycle -------------------------------------------------------
    def start(self) -> None:
        if shutil.which(self.binary) is None and not Path(self.binary).exists():
            raise FileNotFoundError(
                f"llama-server binary not found: {self.binary!r}. Build llama.cpp "
                "first (scripts/prepare_models.py --build-llama-cpp).")
        self.port = _free_port(self.port)
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
        self._log_fh = open(self.log_path, "w", encoding="utf-8", errors="replace")
        t0 = time.time()
        self._proc = subprocess.Popen(
            self.command, stdout=self._log_fh, stderr=subprocess.STDOUT,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None)
        self._wait_healthy(t0)
        self.load_time_s = time.time() - t0
        self.offload = detect_gpu_offload(self.read_log())

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

    def stop(self) -> None:
        if self._sampler is not None:
            self.memory = self._sampler.stop()
            self._sampler = None
        if self._proc is not None and self._proc.poll() is None:
            try:
                if hasattr(os, "killpg"):
                    os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
                else:  # pragma: no cover
                    self._proc.terminate()
                self._proc.wait(timeout=60)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        self._proc = None
        if self._log_fh is not None:
            try:
                self._log_fh.flush()
                self._log_fh.close()
            except Exception:
                pass
            self._log_fh = None

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
            with urllib.request.urlopen(req, timeout=1800) as resp:
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
            precision=self.model.precision)

    def provenance(self) -> Dict[str, Any]:
        return {
            "backend": self.name, "is_fixture": False,
            "command": self.command, "model": asdict(self.model),
            "sampling": asdict(self.sampling),
            "sampling_hash": self.sampling.hash(),
            "load_time_s": round(self.load_time_s, 3),
            "gpu_offload": self.offload, "memory": self.memory,
            "server_log": str(self.log_path),
        }


class LlamaCliBackend(Backend):
    """`llama-cli` subprocess fallback. Same params, one process per call."""

    name = "llama-cli"

    def __init__(self, model: ModelSpec, sampling: SamplingConfig,
                 binary: str = "llama-cli", n_gpu_layers: int = 99,
                 log_dir: str = "results/server_logs",
                 extra_args: Optional[List[str]] = None):
        self.model = model
        self.sampling = sampling
        self.binary = binary
        self.n_gpu_layers = n_gpu_layers
        self.extra_args = list(extra_args or [])
        self.log_path = Path(log_dir) / f"llama-cli.{model.precision}.log"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.offload: Dict[str, Any] = {}
        self.load_time_s = 0.0
        self.memory: Dict[str, float] = {}
        self.command: List[str] = []

    def start(self) -> None:
        if shutil.which(self.binary) is None and not Path(self.binary).exists():
            raise FileNotFoundError(f"llama-cli binary not found: {self.binary!r}")

    def chat(self, messages: List[Dict[str, str]],
             max_tokens: Optional[int] = None) -> GenerationResult:
        # llama-cli has no messages-file flag, so render the chat transcript with
        # the ChatML template Qwen2.5-Instruct expects and feed it via -f.
        rendered = "\n".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>"
                             for m in messages) + "\n<|im_start|>assistant\n"
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                         encoding="utf-8") as fh:
            fh.write(rendered)
            msg_file = fh.name
        self.command = [
            self.binary, "-m", self.model.path, "-c", str(self.sampling.n_ctx),
            "-ngl", str(self.n_gpu_layers), "--temp", str(self.sampling.temperature),
            "--top-p", str(self.sampling.top_p), "--top-k", str(self.sampling.top_k),
            "--repeat-penalty", str(self.sampling.repeat_penalty),
            "--seed", str(self.sampling.seed),
            "-n", str(int(max_tokens or self.sampling.max_tokens)),
            "--no-warmup", "-no-cnv", "--simple-io",
            "-f", msg_file,
        ] + self.extra_args
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(self.command, capture_output=True, text=True,
                                  timeout=1800)
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
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(proc.stderr or "")
        self.offload = detect_gpu_offload(proc.stderr or "")
        text = (proc.stdout or "")
        if rendered in text:
            text = text.split(rendered, 1)[1]
        return GenerationResult(text=text.strip(), latency_s=dt, backend=self.name,
                                model_sha256=self.model.sha256,
                                precision=self.model.precision,
                                error=None if proc.returncode == 0
                                else f"exit_code={proc.returncode}")

    def provenance(self) -> Dict[str, Any]:
        return {"backend": self.name, "is_fixture": False,
                "command": self.command, "model": asdict(self.model),
                "sampling": asdict(self.sampling),
                "sampling_hash": self.sampling.hash(),
                "gpu_offload": self.offload}


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
                                model_sha256="FIXTURE", precision=self.precision)

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
                      if k in ("binary", "n_gpu_layers", "log_dir", "extra_args")}
        return LlamaCliBackend(model=model, sampling=sampling, **cli_kwargs)
    raise ValueError(f"unknown backend {kind!r}")
