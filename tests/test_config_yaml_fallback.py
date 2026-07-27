"""The no-PyYAML fallback parser must agree with PyYAML, or refuse to guess.

configs/ is read through config.load_yaml, which drops to config._mini_yaml when
PyYAML is missing. A fallback that parses `precisions: [F16, Q8_0, Q4_K_M]` as a
string, or decision-gate bands as strings, mis-configures a whole run without
failing, so these tests pin it against yaml.safe_load.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import config as C

yaml = pytest.importorskip("yaml", reason="parity is defined against PyYAML")

CONFIGS = sorted((ROOT / "configs").glob("*.yaml"))


def test_configs_are_discovered():
    assert CONFIGS, "no configs/*.yaml found -- the parity test would be vacuous"


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_fallback_matches_pyyaml_on_repo_configs(path):
    text = path.read_text(encoding="utf-8")
    assert C._mini_yaml(text) == yaml.safe_load(text)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_fallback_types_match_pyyaml_recursively(path):
    """Equality alone lets 1 == 1.0 and True == 1 through; compare types too."""
    text = path.read_text(encoding="utf-8")

    def shape(obj):
        if isinstance(obj, dict):
            return {k: (type(k).__name__, shape(v)) for k, v in obj.items()}
        if isinstance(obj, list):
            return [shape(v) for v in obj]
        return type(obj).__name__

    assert shape(C._mini_yaml(text)) == shape(yaml.safe_load(text))


# Each case is a construct the old fallback silently got wrong.
REGRESSIONS = [
    "precisions: [F16, Q8_0, Q4_K_M]\n",
    "conditions: [fixed_verified_evidence]\n",
    "bands:\n  - {max: 0.50, status: \"BLOCKED\"}\n  - {max: 0.70, status: \"OK\"}\n",
    "by_width: {2: 4, 3: 4, 4: 4}\n",
    "by_width:\n  2: 3          # 3 questions needing 2 documents\n"
    '  "3_or_4": 3   # 3 questions needing 3 or 4 documents\n',
    "kind: llama-server        # llama-server | llama-cli | fixture\n",
    "hash_fallback: false      # true ONLY for harness tests\n",
    "max_search_rounds: 2      # \"at most two search rounds\"\n",
    "purpose: >\n  Decide whether the pairing has enough F16 headroom\n"
    "  to expose quantization effects at all.\n",
    "rule: |\n  1. low-width  : width == 2\n  2. high-width : max available width\n"
    "     continued, more indented\n  3. done\n",
]

# Scalars, collections and layouts the fallback claims to support.
SUPPORTED = [
    "",
    "\n\n# only comments\n",
    "---\na: 1\n",
    "a: null\nb: ~\nc:\n",
    "t1: true\nt2: True\nt3: yes\nt4: on\nf1: false\nf2: no\nf3: off\n",
    "n1: 0\nn2: -7\nn3: +7\nn4: 1_000\nn5: 0644\nn6: 0x1f\nn7: 0b1010\nn8: 1:30\n",
    "f1: 0.5\nf2: -0.5\nf3: .5\nf4: 1.\nf5: 1.5e+3\nf6: .inf\nf7: -.Inf\n",
    "s1: y\ns2: n\ns3: 1.5e3\ns4: llama-server\ns5: Qwen/Qwen2.5-3B-Instruct\n",
    "s6: don't do it\ns7: -DGGML_CUDA=ON -DLLAMA_CURL=OFF\n",
    "url: https://github.com/ggml-org/llama.cpp\n",
    "q1: \"a # b\"\nq2: 'a # b'\nq3: 'a''b'\nq4: \"tab\\there\"\nq5: \"\\u00e9\"\n",
    "q6: \"BLOCKED \\u2014 FLOOR\"\nq7: \"x: y\"\nq8: \"[not, a, list]\"\n",
    "empty_seq: []\nempty_map: {}\nspaced_seq: [ ]\nspaced_map: { }\n",
    # An omitted flow *value* is legal (unlike an omitted key). `tok[:1] in "&*!"`
    # is true for the empty string, so this once looked like an anchor.
    "a: {k: }\nb: {k:}\nc: {k}\nd: [a, ]\ne: {k: 1,}\n",
    "empty_block: |\nnext: 1\n",
    "empty_fold: >\nnext: 1\n",
    "nested: [[1, 2], {a: 1}, null]\n",
    "flow_map: {a: 1, b: [2, 3], c: {d: 4}}\n",
    # A space-less `:` separates only before `[` or `{`, so `a:1` and `a:"x"` are
    # each a single scalar key while `a:[1]` is a key with a sequence value.
    "tight: {a:[1], b:{c: 2}, d:1, e:\"x\", f:g}\n",
    "tight_seq: [a:1, b:[1]]\n",
    # A `key: value` inside `[...]` is an entry that is itself a one-pair mapping.
    "pairs: [a: 1, b: 2]\npair_mixed: [a, b: 2]\npair_empty: [a: ]\n",
    "pair_nested: [a: [1, 2], b: {c: 3}]\n",
    "seq:\n  - a\n  - b\n",
    "seq:\n- a\n- b\n",
    "seq:\n  - key: 1\n    other: 2\n  - key: 3\n    other: 4\n",
    "seq:\n  -\n    key: 1\n  -\n    key: 2\n",
    # A quoted key opens a compact block mapping just like a plain one; an
    # earlier revision read this entry as a scalar and refused the line after it.
    'seq:\n  - "x y": 1\n    2: two\n  - {a: 1}\n',
    "seq:\n  - 'q': 1\n    b: 2\n",
    "seq:\n  - \"just a string\"\n  - 'another'\n",
    "seq:\n  - - a\n    - b\n  - - c\n",
    "seq:\n  - - a: 1\n      b: 2\n",
    "outer:\n  inner:\n    deep: 1\n  sibling: 2\ntop: 3\n",
    "a: 1\n\n# comment between entries\nb: 2\n",
    "lit: |\n  a\n   b\n  c\n",
    "lit_strip: |-\n  a\n  b\n",
    "fold: >\n  a\n  b\n",
    "fold_strip: >-\n  a\n  b\n",
    "fold_blank: >\n  a\n\n  b\n",
    "fold_indented: >\n  a\n   b\n  c\n",
    "block_then_key: |\n  body\nnext: 1\n",
    "block_with_blank: |\n  a\n\n  b\nnext: 1\n",
    "block_comment_header: | # trailing comment\n  a\n",
    "hash_in_block: |\n  # not a comment\n  a # nor this\n",
    "tab_in_block: |\n  a\n  \tb\n  c\n",  # tabs may not indent, but are content
    "dash_in_block: |\n  - not a sequence\n  - nor this\n",
    "dupe: 1\ndupe: 2\n",
]


@pytest.mark.parametrize("text", REGRESSIONS + SUPPORTED)
def test_fallback_matches_pyyaml_on_supported_subset(text):
    assert C._mini_yaml(text) == (yaml.safe_load(text) or {})


# Valid YAML that the fallback must refuse rather than mis-parse, because it
# cannot reproduce what PyYAML would return.
UNSUPPORTED = [
    "base: &anchor\n  a: 1\nuse: *anchor\n",             # anchors / aliases
    "a: !!str 1\n",                                      # explicit tags
    "base: &base {a: 1}\nchild:\n  <<: *base\n  b: 2\n",  # merge key
    "when: 2024-01-02\n",                                # timestamp -> date
    "when: 2024-01-02 03:04:05\n",                       # timestamp -> datetime
    "keep: |+\n  a\n\n",                                 # `keep` chomping
    "explicit: |2\n    a\n",                             # indentation indicator
    "a: multi\n  line plain scalar\n",                   # folded plain scalar
    "a: [1,\n  2]\n",                                    # multi-line flow
    'a: "a\\\nb"\n',                                     # escaped line continuation
    "? complex\n: value\n",                              # explicit key
    "- a\n- b\n",                                        # root is a sequence
    "a: !!seq [1]\n",                                    # tagged collection
    "a: 2001-12-14t21:59:43.10-05:00\n",                 # timestamp with zone
]

# Input PyYAML itself rejects. The fallback must not accept it either, or it
# would quietly succeed where the real loader would have stopped the run.
MALFORMED = [
    "a: 1\n---\nb: 2\n",                                 # multi-document stream
    'a: "unterminated\n',                                # multi-line quoted
    "a: b: c\n",                                         # nested mapping value
    "a:\n\t b: 1\n",                                     # tab indentation
    "a: =\n",                                            # the `value` type
    ": 1\n",                                             # omitted block key
    "a: {: 1}\n",                                        # omitted flow key
    "a: [,]\n",                                          # omitted flow entry
    "a: {:b}\n",                                         # flow scalar led by `:`
    "a: [:b]\n",
    "a: :\n",
]


@pytest.mark.parametrize("text", UNSUPPORTED + MALFORMED)
def test_fallback_refuses_what_it_cannot_reproduce(text):
    with pytest.raises(C.MiniYamlUnsupported) as excinfo:
        C._mini_yaml(text)
    assert "PyYAML" in str(excinfo.value)


@pytest.mark.parametrize("text", UNSUPPORTED)
def test_unsupported_cases_are_not_secretly_invalid_yaml(text):
    """Guard the list above: each case must be YAML that PyYAML itself accepts.

    Otherwise a case could pass for the wrong reason -- refused because it is
    malformed, not because the fallback declines a construct it cannot mirror.
    """
    yaml.safe_load(text)


@pytest.mark.parametrize("text", MALFORMED)
def test_malformed_cases_really_are_rejected_by_pyyaml(text):
    with pytest.raises(yaml.YAMLError):
        yaml.safe_load(text)


def test_load_yaml_error_names_the_file(tmp_path, monkeypatch):
    bad = tmp_path / "stage_x.yaml"
    bad.write_text("when: 2024-01-02\n", encoding="utf-8")
    monkeypatch.setitem(sys.modules, "yaml", None)  # force the ImportError path
    with pytest.raises(C.MiniYamlUnsupported) as excinfo:
        C.load_yaml(bad)
    assert "stage_x.yaml" in str(excinfo.value)
    assert "PyYAML" in str(excinfo.value)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_load_yaml_agrees_with_and_without_pyyaml(path, monkeypatch):
    """The public entry point, not just _mini_yaml, must load configs the same."""
    with_yaml = C.load_yaml(path)
    monkeypatch.setitem(sys.modules, "yaml", None)
    assert C.load_yaml(path) == with_yaml
