"""Config loading + repo path resolution. Kaggle-safe, never Windows-specific."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve(path: str | os.PathLike) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (repo_root() / p)


_INSTALL_HINT = "install PyYAML (pip install 'PyYAML>=6.0') and re-run"


class MiniYamlUnsupported(ValueError):
    """The fallback YAML parser met input it cannot read the way PyYAML would.

    Raised instead of guessing. A stage config that loads with the wrong
    precision list or unusable decision-gate bands would mis-configure a whole
    run silently; failing here stops the run instead.
    """


# --- scalar resolution -----------------------------------------------------
# These are PyYAML's own YAML 1.1 implicit resolvers, so plain scalars get the
# same type they would under yaml.safe_load. Keep them byte-for-byte in step
# with yaml.resolver.Resolver.yaml_implicit_resolvers.
_RE_NULL = re.compile(r"^(?:~|null|Null|NULL|)$")
_RE_BOOL = re.compile(
    r"^(?:yes|Yes|YES|no|No|NO|true|True|TRUE|false|False|FALSE"
    r"|on|On|ON|off|Off|OFF)$"
)
_RE_INT = re.compile(
    r"""^(?:[-+]?0b[0-1_]+
         |[-+]?0[0-7_]+
         |[-+]?(?:0|[1-9][0-9_]*)
         |[-+]?0x[0-9a-fA-F_]+
         |[-+]?[1-9][0-9_]*(?::[0-5]?[0-9])+)$""",
    re.X,
)
_RE_FLOAT = re.compile(
    r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+][0-9]+)?
         |\.[0-9][0-9_]*(?:[eE][-+][0-9]+)?
         |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*
         |[-+]?\.(?:inf|Inf|INF)
         |\.(?:nan|NaN|NAN))$""",
    re.X,
)
# Plain scalars PyYAML resolves to a type this parser deliberately does not
# reproduce (dates/datetimes, merge and value keys).
_RE_TYPED_ELSEWHERE = re.compile(
    r"^(?:<<|=|[0-9]{4}-[0-9][0-9]?-[0-9][0-9]?(?:[Tt \t].*)?)$"
)
_TRUE = {"yes", "true", "on"}

_ESCAPES = {
    "0": "\x00", "a": "\x07", "b": "\x08", "t": "\t", "\t": "\t", "n": "\n",
    "v": "\x0b", "f": "\x0c", "r": "\r", "e": "\x1b", " ": " ", '"': '"',
    "/": "/", "\\": "\\", "N": "\x85", "_": "\xa0", "L": "\u2028",
    "P": "\u2029",
}
_HEX_ESCAPES = {"x": 2, "u": 4, "U": 8}

_BLOCK_HEADER = re.compile(r"^([|>])([+-]?)([0-9]*)([+-]?)$")
_FLOW_ENDS = ",]}"
# In flow context a `:` only separates a key when whitespace, the end of the
# collection or a nested collection follows it. That is why `{a:1}` is the single
# scalar `'a:1'` while `{a:[1]}` is `{'a': [1]}`.
_ENDS_FLOW_KEY = _FLOW_ENDS + "[{"


def _unsupported(what: str) -> MiniYamlUnsupported:
    return MiniYamlUnsupported(
        f"the built-in YAML fallback cannot parse {what} the way PyYAML does; "
        f"{_INSTALL_HINT}"
    )


def _to_int(tok: str) -> int:
    """Mirror of yaml.constructor.SafeConstructor.construct_yaml_int."""
    tok = tok.replace("_", "")
    sign = -1 if tok[0] == "-" else 1
    if tok[0] in "+-":
        tok = tok[1:]
    if tok == "0":
        return 0
    if tok.startswith("0b"):
        return sign * int(tok[2:], 2)
    if tok.startswith("0x"):
        return sign * int(tok[2:], 16)
    if tok[0] == "0":
        return sign * int(tok, 8)
    if ":" in tok:
        total, base = 0, 1
        for part in reversed(tok.split(":")):
            total += int(part) * base
            base *= 60
        return sign * total
    return sign * int(tok)


def _to_float(tok: str) -> float:
    """Mirror of yaml.constructor.SafeConstructor.construct_yaml_float."""
    tok = tok.replace("_", "").lower()
    sign = -1 if tok[0] == "-" else 1
    if tok[0] in "+-":
        tok = tok[1:]
    if tok == ".inf":
        return sign * float("inf")
    if tok == ".nan":
        return float("nan")
    if ":" in tok:
        total, base = 0.0, 1
        for part in reversed(tok.split(":")):
            total += float(part) * base
            base *= 60
        return sign * total
    return sign * float(tok)


def _plain_scalar(tok: str) -> Any:
    """Resolve an unquoted scalar exactly as yaml.safe_load would."""
    if _RE_NULL.match(tok):
        return None
    if _RE_BOOL.match(tok):
        return tok.lower() in _TRUE
    if _RE_INT.match(tok):
        return _to_int(tok)
    if _RE_FLOAT.match(tok):
        return _to_float(tok)
    if _RE_TYPED_ELSEWHERE.match(tok):
        raise _unsupported(f"the plain scalar {tok!r}")
    return tok


def _read_quoted(s: str, i: int) -> Tuple[str, int]:
    """Read the quoted scalar starting at s[i]; return (value, index after it)."""
    quote = s[i]
    i += 1
    out: List[str] = []
    while i < len(s):
        c = s[i]
        if c == quote:
            if quote == "'" and s[i + 1:i + 2] == "'":  # '' is a literal quote
                out.append("'")
                i += 2
                continue
            return "".join(out), i + 1
        if c == "\\" and quote == '"':
            esc = s[i + 1:i + 2]
            if not esc:  # a backslash at end of line continues the scalar
                break
            if esc in _HEX_ESCAPES:
                width = _HEX_ESCAPES[esc]
                digits = s[i + 2:i + 2 + width]
                if len(digits) != width:
                    raise _unsupported(f"the escape sequence {s[i:i + 2 + width]!r}")
                out.append(chr(int(digits, 16)))
                i += 2 + width
                continue
            if esc not in _ESCAPES:
                raise _unsupported(f"the escape sequence {s[i:i + 2]!r}")
            out.append(_ESCAPES[esc])
            i += 2
            continue
        out.append(c)
        i += 1
    # PyYAML would fold this onto the next line; we refuse rather than guess.
    raise _unsupported("a quoted scalar spanning more than one line")


def _quote_opens_here(s: str, i: int) -> bool:
    """True if s[i] starts a quoted scalar rather than sitting inside a word.

    ``don't`` is a valid plain scalar, so an apostrophe only opens a quoted
    scalar where a scalar is allowed to begin.
    """
    return i == 0 or s[i - 1] in " \t,[{"


def _strip_comment(content: str) -> str:
    """Drop a trailing ``# comment``. `content` must already be left-stripped."""
    i = 0
    while i < len(content):
        c = content[i]
        if c == "#" and (i == 0 or content[i - 1] in " \t"):
            return content[:i].rstrip()
        if c in "\"'" and _quote_opens_here(content, i):
            _, i = _read_quoted(content, i)
            continue
        i += 1
    return content.rstrip()


def _split_key(content: str) -> Optional[Tuple[str, str]]:
    """Split ``key: value`` at the first top-level ``:``, or return None.

    Only a colon followed by whitespace or end-of-line separates a key, which is
    why ``repo: https://example.com`` and the ``:`` inside a flow collection or a
    quoted string are left alone.
    """
    i = 0
    while i < len(content):
        c = content[i]
        if c in "\"'" and _quote_opens_here(content, i):
            _, i = _read_quoted(content, i)
            continue
        if c in "[{":
            _, i = _parse_flow(content, i)
            continue
        if c == ":" and (i + 1 == len(content) or content[i + 1] in " \t"):
            return content[:i].strip(), content[i + 1:].strip()
        i += 1
    return None


# --- flow collections ------------------------------------------------------
def _parse_flow(s: str, i: int) -> Tuple[Any, int]:
    """Parse one flow node starting at or after s[i]; return (value, end index)."""
    while i < len(s) and s[i] in " \t":
        i += 1
    if i >= len(s):
        raise _unsupported("a flow collection spanning more than one line")
    if s[i] == "[":
        return _parse_flow_seq(s, i + 1)
    if s[i] == "{":
        return _parse_flow_map(s, i + 1)
    if s[i] in "\"'":
        return _read_quoted(s, i)
    return _parse_flow_plain(s, i)


def _parse_flow_plain(s: str, i: int) -> Tuple[Any, int]:
    start = i
    while i < len(s):
        c = s[i]
        if c in _FLOW_ENDS:
            break
        if c == ":" and (i + 1 == len(s) or s[i + 1] in " \t"
                         or s[i + 1] in _ENDS_FLOW_KEY):
            break
        if c == "#" and i > start and s[i - 1] in " \t":
            break
        i += 1
    tok = s[start:i].strip()
    if tok and tok[0] in "&*!":
        raise _unsupported("anchors, aliases and explicit tags")
    if tok.startswith(":"):
        # `{:a}` / `[:a]`: PyYAML rejects a flow scalar starting with a colon,
        # though a block one (`k: :a`) is fine.
        raise _unsupported(f"the flow collection {s!r}")
    return _plain_scalar(tok), i


def _skip_flow_space(s: str, i: int) -> int:
    while i < len(s) and s[i] in " \t":
        i += 1
    return i


def _parse_flow_seq(s: str, i: int) -> Tuple[List[Any], int]:
    out: List[Any] = []
    i = _skip_flow_space(s, i)
    if s[i:i + 1] == "]":
        return out, i + 1
    while True:
        start = _skip_flow_space(s, i)
        value, i = _parse_flow(s, i)
        if i == start:  # `[,]` -- PyYAML rejects an omitted entry
            raise _unsupported(f"the flow sequence {s!r}")
        i = _skip_flow_space(s, i)
        if s[i:i + 1] == ":":
            # `[a: 1]` is a sequence of one single-pair mapping.
            pair_value, i = _parse_flow(s, i + 1)
            value = {_check_key(value): pair_value}
            i = _skip_flow_space(s, i)
        out.append(value)
        if s[i:i + 1] == ",":
            i = _skip_flow_space(s, i + 1)
            if s[i:i + 1] == "]":  # trailing comma
                return out, i + 1
            continue
        if s[i:i + 1] == "]":
            return out, i + 1
        raise _unsupported(f"the flow sequence {s!r}")


def _parse_flow_map(s: str, i: int) -> Tuple[Dict[Any, Any], int]:
    out: Dict[Any, Any] = {}
    i = _skip_flow_space(s, i)
    if s[i:i + 1] == "}":
        return out, i + 1
    while True:
        if s[i:i + 1] == "?":
            raise _unsupported("explicit (`? `) mapping keys")
        start = _skip_flow_space(s, i)
        key, i = _parse_flow(s, i)
        if i == start:  # `{: 1}` -- PyYAML rejects an omitted key
            raise _unsupported(f"the flow mapping {s!r}")
        i = _skip_flow_space(s, i)
        value: Any = None
        if s[i:i + 1] == ":":
            # An omitted value is legal here: `{a: }` is `{'a': None}`.
            value, i = _parse_flow(s, i + 1)
            i = _skip_flow_space(s, i)
        out[_check_key(key)] = value
        if s[i:i + 1] == ",":
            i = _skip_flow_space(s, i + 1)
            if s[i:i + 1] == "}":  # trailing comma
                return out, i + 1
            continue
        if s[i:i + 1] == "}":
            return out, i + 1
        raise _unsupported(f"the flow mapping {s!r}")


def _check_key(key: Any) -> Any:
    if isinstance(key, (dict, list)):
        raise _unsupported("a collection used as a mapping key")
    return key


def _parse_value(tok: str) -> Any:
    """Parse the value part of a ``key: value`` line (comment already removed)."""
    if not tok:
        return None
    if tok[0] in "&*!":
        raise _unsupported("anchors, aliases and explicit tags")
    if tok[0] in "[{":
        value, end = _parse_flow(tok, 0)
    elif tok[0] in "\"'":
        value, end = _read_quoted(tok, 0)
    else:
        if _split_key(tok) is not None:
            # PyYAML raises a ScannerError on `a: b: c`; do not invent a value.
            raise _unsupported(f"the nested mapping value {tok!r}")
        return _plain_scalar(tok)
    if tok[end:].strip():
        raise _unsupported(f"trailing content after a value in {tok!r}")
    return value


# --- block structure -------------------------------------------------------
class _Doc:
    """Line-oriented recursive-descent reader for the supported YAML subset."""

    def __init__(self, text: str) -> None:
        self.raw = text.splitlines()

    # -- line helpers
    def blank(self, i: int) -> bool:
        stripped = self.raw[i].strip()
        return not stripped or stripped.startswith("#")

    def indent(self, i: int) -> int:
        """Leading spaces on a structural line. Tabs may not indent YAML."""
        width = self.spaces(i)
        if self.raw[i][width:width + 1] == "\t":
            raise _unsupported("tab characters in indentation")
        return width

    def spaces(self, i: int) -> int:
        """Leading spaces, no tab check -- tabs are legal *inside* a block scalar."""
        line = self.raw[i]
        return len(line) - len(line.lstrip(" "))

    def content(self, i: int) -> str:
        return _strip_comment(self.raw[i].strip())

    def next_sig(self, i: int) -> int:
        while i < len(self.raw) and self.blank(i):
            i += 1
        return i

    def is_item(self, i: int) -> bool:
        content = self.raw[i].strip()
        return content == "-" or content.startswith("- ")

    # -- entry point
    def parse(self) -> Dict[str, Any]:
        i = self.next_sig(0)
        if i < len(self.raw) and self.content(i) == "---":
            i = self.next_sig(i + 1)
        if i >= len(self.raw):
            return {}
        if self.is_item(i):
            raise _unsupported("a document whose root is a sequence")
        value, i = self.parse_map(i, self.indent(i))
        i = self.next_sig(i)
        if i < len(self.raw):
            raise _unsupported(f"trailing content at line {i + 1}")
        return value

    def parse_node(self, i: int, indent: int) -> Tuple[Any, int]:
        if self.is_item(i):
            return self.parse_seq(i, indent)
        return self.parse_map(i, indent)

    def parse_map(self, i: int, indent: int) -> Tuple[Dict[Any, Any], int]:
        out: Dict[Any, Any] = {}
        while i < len(self.raw):
            if self.blank(i):
                i += 1
                continue
            line_indent = self.indent(i)
            if line_indent < indent:
                break
            if line_indent > indent:
                raise _unsupported(f"the indentation at line {i + 1}")
            content = self.content(i)
            if content in ("---", "..."):
                raise _unsupported("multi-document streams")
            if content.startswith("? "):
                raise _unsupported("explicit (`? `) mapping keys")
            if self.is_item(i):
                raise _unsupported(f"the sequence entry at line {i + 1}")
            split = _split_key(content)
            if split is None:
                raise _unsupported(f"line {i + 1}: {content!r}")
            key, rest = split
            if not key:  # `: 1` -- PyYAML rejects an omitted key
                raise _unsupported(f"the omitted key at line {i + 1}")
            value, i = self.parse_pair_value(i, indent, rest)
            out[_check_key(_parse_value(key))] = value
        return out, i

    def parse_pair_value(self, i: int, indent: int, rest: str) -> Tuple[Any, int]:
        """Parse the value for a mapping key on line `i`; return (value, next i)."""
        if _BLOCK_HEADER.match(rest):
            return self.read_block_scalar(i + 1, indent, rest)
        if rest:
            i += 1
            nxt = self.next_sig(i)
            if nxt < len(self.raw) and self.indent(nxt) > indent:
                # PyYAML would fold this into the scalar above.
                raise _unsupported("a plain scalar spanning more than one line")
            return _parse_value(rest), i
        i += 1
        nxt = self.next_sig(i)
        if nxt >= len(self.raw):
            return None, nxt
        child_indent = self.indent(nxt)
        if child_indent > indent:
            return self.parse_node(nxt, child_indent)
        # A block sequence may sit at the same column as its key.
        if child_indent == indent and self.is_item(nxt):
            return self.parse_seq(nxt, indent)
        return None, i

    def parse_seq(self, i: int, indent: int) -> Tuple[List[Any], int]:
        out: List[Any] = []
        while i < len(self.raw):
            if self.blank(i):
                i += 1
                continue
            line_indent = self.indent(i)
            if line_indent < indent:
                break
            if line_indent > indent:
                raise _unsupported(f"the indentation at line {i + 1}")
            if not self.is_item(i):
                break  # a sibling key of the mapping that owns this sequence
            value, i = self.parse_item(i, indent)
            out.append(value)
        return out, i

    def parse_item(self, i: int, indent: int) -> Tuple[Any, int]:
        """Parse the ``- ...`` entry on line `i`; return (value, next i)."""
        line = self.raw[i]
        col = indent + 1
        while col < len(line) and line[col] == " ":
            col += 1
        inner_raw = line[col:]
        inner = _strip_comment(inner_raw.strip())
        if not inner:
            nxt = self.next_sig(i + 1)
            if nxt < len(self.raw) and self.indent(nxt) > indent:
                return self.parse_node(nxt, self.indent(nxt))
            return None, i + 1
        if _BLOCK_HEADER.match(inner):
            return self.read_block_scalar(i + 1, indent, inner)
        # Re-anchor the entry body at its real column so nested maps and
        # sequences (`- key: value`, `- - a`) parse like any other block node.
        # Test for a nested sequence first: `_split_key` would read `- a: 1` as
        # the key `- a`. It correctly reports None for a flow collection or a
        # quoted scalar, and a pair for a quoted key (`- "x y": 1`).
        if inner.startswith("- "):
            self.raw[i] = " " * col + inner_raw
            return self.parse_seq(i, col)
        if _split_key(inner) is not None:
            self.raw[i] = " " * col + inner_raw
            return self.parse_node(i, col)
        i += 1
        nxt = self.next_sig(i)
        if nxt < len(self.raw) and self.indent(nxt) > indent:
            raise _unsupported("a plain scalar spanning more than one line")
        return _parse_value(inner), i

    def read_block_scalar(self, i: int, indent: int, header: str) -> Tuple[str, int]:
        """Read a ``|``/``>`` block whose parent key/entry sits at `indent`."""
        style, chomp_a, digits, chomp_b = _BLOCK_HEADER.match(header).groups()
        if digits:
            raise _unsupported("block scalars with an explicit indentation indicator")
        chomp = chomp_a or chomp_b
        if chomp == "+":
            raise _unsupported("`keep` (`+`) block-scalar chomping")

        body: List[str] = []
        while i < len(self.raw):
            line = self.raw[i]
            if line.strip() and self.spaces(i) <= indent:
                break
            body.append(line)
            i += 1
        while body and not body[-1].strip():
            body.pop()  # trailing blank lines only matter for `keep`, refused above
        if not body:
            return "", i

        block_indent = len(body[0]) - len(body[0].lstrip(" "))
        lines = [ln[block_indent:] if ln.strip() else "" for ln in body]
        text = "\n".join(lines) if style == "|" else _fold(lines)
        return text if chomp == "-" else text + "\n", i


def _fold(lines: List[str]) -> str:
    """Fold a ``>`` block: line breaks become spaces, blank lines become breaks.

    A more-indented line is kept verbatim and the breaks around it stay literal,
    which is what PyYAML does.
    """
    out = lines[0]
    i = 1
    while i < len(lines):
        if lines[i] == "":
            j = i
            while j < len(lines) and lines[j] == "":
                j += 1
            out += "\n" * (j - i) + lines[j]
            i = j + 1
            continue
        more = lines[i][:1] in (" ", "\t")
        prev_more = lines[i - 1][:1] in (" ", "\t")
        out += ("\n" if more or prev_more else " ") + lines[i]
        i += 1
    return out


def _mini_yaml(text: str) -> Dict[str, Any]:
    """Parse the YAML subset used by configs/ exactly as yaml.safe_load would.

    Used only when PyYAML is unavailable. Anything outside that subset raises
    MiniYamlUnsupported instead of being guessed at -- see the class docstring.
    """
    return _Doc(text).parse()


def load_yaml(path: str | os.PathLike) -> Dict[str, Any]:
    text = resolve(path).read_text(encoding="utf-8")
    try:
        import yaml  # noqa: WPS433
    except ImportError:
        try:
            return _mini_yaml(text)
        except MiniYamlUnsupported as exc:
            raise MiniYamlUnsupported(f"{resolve(path)}: {exc}") from exc
    return yaml.safe_load(text) or {}


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
