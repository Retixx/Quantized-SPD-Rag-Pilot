"""Config loading + repo path resolution. Kaggle-safe, never Windows-specific."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve(path: str | os.PathLike) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (repo_root() / p)


def _mini_yaml(text: str) -> Dict[str, Any]:
    """Tiny YAML subset loader (nested maps, lists, scalars, block scalars).

    Used only if PyYAML is unavailable; PyYAML is preferred and in requirements.
    """
    import json as _json

    def cast(tok: str) -> Any:
        tok = tok.strip()
        if tok in ("null", "~", ""):
            return None
        if tok in ("true", "True"):
            return True
        if tok in ("false", "False"):
            return False
        if tok.startswith(("[", "{")):
            try:
                return _json.loads(tok.replace("'", '"'))
            except Exception:
                return tok
        if tok.startswith(('"', "'")) and tok.endswith(('"', "'")):
            return tok[1:-1]
        try:
            return int(tok)
        except ValueError:
            pass
        try:
            return float(tok)
        except ValueError:
            pass
        return tok

    root: Dict[str, Any] = {}
    stack: List[tuple] = [(-1, root)]
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i]
        i += 1
        if not raw.strip() or raw.strip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if line.startswith("- "):
            if not isinstance(parent, list):
                continue
            parent.append(cast(line[2:]))
            continue
        if ":" not in line:
            continue
        key, _, rest = line.partition(":")
        key = key.strip().strip('"')
        rest = rest.strip()
        if rest in (">", "|", ">-", "|-"):
            block: List[str] = []
            while i < len(lines):
                nxt = lines[i]
                if nxt.strip() and (len(nxt) - len(nxt.lstrip(" "))) <= indent:
                    break
                block.append(nxt.strip())
                i += 1
            parent[key] = " ".join(b for b in block if b)
            continue
        if rest == "":
            nxt_indent = None
            j = i
            while j < len(lines):
                if lines[j].strip() and not lines[j].strip().startswith("#"):
                    nxt_indent = len(lines[j]) - len(lines[j].lstrip(" "))
                    is_list = lines[j].strip().startswith("- ")
                    break
                j += 1
            container: Any = [] if (nxt_indent is not None and nxt_indent > indent
                                    and is_list) else {}
            parent[key] = container
            stack.append((indent, container))
            continue
        parent[key] = cast(rest)
    return root


def load_yaml(path: str | os.PathLike) -> Dict[str, Any]:
    text = resolve(path).read_text(encoding="utf-8")
    try:
        import yaml  # noqa: WPS433
        return yaml.safe_load(text) or {}
    except ImportError:
        return _mini_yaml(text)


def load_models_config(path: str = "configs/models.yaml") -> Dict[str, Any]:
    return load_yaml(path)


def load_stage_config(stage: str, path: Optional[str] = None) -> Dict[str, Any]:
    return load_yaml(path or f"configs/{stage}.yaml")


def kaggle_paths() -> Dict[str, Any]:
    """Discover Kaggle input/working layout. Returns POSIX paths only."""
    inp = Path("/kaggle/input")
    work = Path("/kaggle/working")
    on_kaggle = inp.exists() or work.exists()
    inputs: List[str] = []
    if inp.exists():
        inputs = sorted(str(p) for p in inp.iterdir() if p.is_dir())
    return {
        "on_kaggle": on_kaggle,
        "input_root": str(inp) if inp.exists() else None,
        "input_datasets": inputs,
        "working_root": str(work) if work.exists() else None,
        "results_dir": str(work / "results") if work.exists() else "results",
    }


def results_dir(override: Optional[str] = None) -> Path:
    if override:
        p = Path(override)
    else:
        kp = kaggle_paths()
        p = Path(kp["results_dir"]) if kp["on_kaggle"] else resolve("results")
    p.mkdir(parents=True, exist_ok=True)
    return p
