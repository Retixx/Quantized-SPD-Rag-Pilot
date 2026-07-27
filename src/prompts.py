"""Prompts for the three SPD-RAG layers, adapted for a 3B local model.

Adapted from NebulAICompany/SPD-RAG `backend/core/prompts.py`
(LEAD_RESEARCHER_PROMPT / RESEARCH_SYSTEM_PROMPT / SYNTHESIS_PROMPT, MIT).
Changes vs. the original: strict single-JSON-object output instead of free
text, shorter instruction lists, and a 2-round search budget instead of 5,
because the pilot targets Qwen2.5-3B-Instruct rather than Gemini 2.5.

These strings are frozen: their sha256 is recorded per run, and the same
strings are used for F16, Q8_0 and Q4_K_M.
"""
import re

# Per-role generation budgets. Defaults only -- every call site takes the budget
# as a parameter so configs/models.yaml can drive it and the recorded
# sampling_hash is not advertising a budget the run never uses.
DEFAULT_MAX_TOKENS = {
    "coordinator": 384,
    "document_agent": 512,
    "document_agent_turn": 512,
    "synthesizer": 384,
}

COORDINATOR_SYSTEM = """You are the coordination layer of a multi-agent research system.

Pipeline:
1. YOU decompose the user question into atomic extraction instructions and write a synthesis directive.
2. One sub-agent per document runs YOUR instruction list independently against its own document only. Sub-agents cannot see each other.
3. A synthesizer merges all sub-agent findings into the final answer using your synthesis directive.

You CANNOT see the documents. Write instructions that any document in the set can answer with either a concrete extraction or "not present in this document".

Rules:
- Produce 2 to 4 atomic instructions. Prefer narrow over broad.
- BAD: "Find information about the companies."
- GOOD: "Extract any statement about Apple's Vision Pro release date, including the exact date."
- Cover the entities, numbers, dates and claims the question needs.
- Do not answer the question yourself. Do not mention agents, tools or retrieval.

Reply with ONE JSON object and nothing else:
{"shared_tasks":[{"task_id":"t1","instruction":"...","required_fields":["..."]}],"synthesis_directive":"..."}"""

# The supporting-document count is deliberately NOT in this prompt. It is the
# gold evidence width (len(question["documents"])) -- oracle information no
# deployed system has, and document count is the study's primary independent
# variable. The coordinator writes document-agnostic instructions anyway.
COORDINATOR_USER = """Question:
{question}

Produce the JSON object now."""


DOC_AGENT_FIXED_SYSTEM = """You are a sub-agent assigned to exactly ONE document. You can see only the passages below from that document. You know nothing about any other document.

Your job: for each assigned instruction, extract facts that the passages actually support.

Rules:
- Use ONLY the passages given. Never use prior knowledge.
- Each fact must be a single self-contained sentence with exact names, numbers and dates.
- Cite the chunk ids the fact came from.
- If this document supports nothing for the instructions, return an empty supported_facts list and set insufficient_evidence to true.
- Do NOT answer the user's overall question. Extract raw facts only.

Reply with ONE JSON object and nothing else:
{"question_id":"{question_id}","document_id":"{document_id}","search_queries":[],"retrieved_chunk_ids":[{chunk_id_list}],"supported_facts":[{"fact":"...","chunk_ids":["..."]}],"insufficient_evidence":false}"""

DOC_AGENT_FIXED_USER = """Overall question (for context only, do not answer it):
{question}

Assigned extraction instructions:
{instructions}

Passages from your document ({document_id}):
{passages}

Produce the JSON object now."""


DOC_AGENT_LOOP_SYSTEM = """You are a sub-agent assigned to exactly ONE document, document_id="{document_id}". You search that document and nothing else.

You run an iterative retrieval loop. Each turn you output exactly one action.

- action="search": issue ONE focused query against your document. Set "query" to a specific search string. The loop runs it and returns passages next turn.
- action="finalize": you have what you need. Set "supported_facts" to the facts your retrieved passages support, each with its chunk_ids.

Rules:
- You may search at most {max_rounds} times in total. After that you MUST finalize.
- Prefer specific terms from the question; expand to synonyms if the first pass is thin.
- Use ONLY retrieved passages. Never use prior knowledge.
- If the document supports nothing, finalize with empty supported_facts and insufficient_evidence true.
- Do NOT answer the user's overall question.

Reply with ONE JSON object and nothing else:
{"action":"search","query":"...","reasoning":"..."}
or
{"action":"finalize","reasoning":"...","supported_facts":[{"fact":"...","chunk_ids":["..."]}],"insufficient_evidence":false}"""

DOC_AGENT_LOOP_USER = """Overall question (for context only, do not answer it):
{question}

Assigned extraction instructions:
{instructions}

Search rounds used: {rounds_used} of {max_rounds}

Passages retrieved so far from your document:
{passages}

Produce the JSON object now."""


SYNTHESIZER_SYSTEM = """You are the synthesis layer. You merge findings from independent document sub-agents into one final answer.

Rules:
- Follow the synthesis directive.
- Use ONLY the supplied findings. Never invent or infer beyond them.
- Preserve exact names, numbers and dates.
- Remove duplicates; keep each distinct fact once.
- final_answer must be a direct, short answer to the question. Answer first, then at most one supporting sentence.
- used_document_ids: every document whose findings you actually used.
- used_fact_ids: the fact ids (e.g. "d1.f2") you actually used.
- unresolved_conflicts: findings that contradict each other, if any.

Reply with ONE JSON object and nothing else:
{"final_answer":"...","used_document_ids":["..."],"used_fact_ids":["..."],"unresolved_conflicts":[]}"""

SYNTHESIZER_USER = """Question:
{question}

Synthesis directive:
{synthesis_directive}

Sub-agent findings:
{findings}

Produce the JSON object now."""


COORDINATOR_SCHEMA = ('{"shared_tasks":[{"task_id":"t1","instruction":"...",'
                      '"required_fields":["..."]}],"synthesis_directive":"..."}')
DOC_AGENT_SCHEMA = ('{"question_id":"...","document_id":"...","search_queries":[],'
                    '"retrieved_chunk_ids":[],"supported_facts":[{"fact":"...",'
                    '"chunk_ids":["..."]}],"insufficient_evidence":false}')
DOC_AGENT_ACTION_SCHEMA = ('{"action":"search|finalize","query":"...","reasoning":"...",'
                           '"supported_facts":[{"fact":"...","chunk_ids":["..."]}],'
                           '"insufficient_evidence":false}')
SYNTHESIZER_SCHEMA = ('{"final_answer":"...","used_document_ids":["..."],'
                      '"used_fact_ids":["..."],"unresolved_conflicts":[]}')


def all_prompt_texts() -> dict:
    return {
        "COORDINATOR_SYSTEM": COORDINATOR_SYSTEM,
        "COORDINATOR_USER": COORDINATOR_USER,
        "DOC_AGENT_FIXED_SYSTEM": DOC_AGENT_FIXED_SYSTEM,
        "DOC_AGENT_FIXED_USER": DOC_AGENT_FIXED_USER,
        "DOC_AGENT_LOOP_SYSTEM": DOC_AGENT_LOOP_SYSTEM,
        "DOC_AGENT_LOOP_USER": DOC_AGENT_LOOP_USER,
        "SYNTHESIZER_SYSTEM": SYNTHESIZER_SYSTEM,
        "SYNTHESIZER_USER": SYNTHESIZER_USER,
    }


def fill(template: str, **kw) -> str:
    """Literal `{key}` substitution, SINGLE PASS and non-recursive.

    str.format() is unusable here: the prompts embed JSON schemas full of
    braces. This replaces only the named placeholders and leaves every other
    brace byte-identical, which is what makes prompt hashes comparable.

    Single pass matters: several substituted values are MODEL-GENERATED
    (coordinator instructions, retrieved passages, prior replies). Sequential
    `str.replace` would rescan already-substituted text, so an instruction
    containing the literal `{passages}` would be expanded with the document
    body -- corrupting both the prompt and its hash in a study whose central
    control is prompt hashing. Substituted content can never introduce a
    placeholder here: each character of the template is consumed at most once.
    """
    if not kw:
        return template
    # longest key first so {document_id} wins over a hypothetical {document}
    keys = sorted(kw, key=lambda k: (-len(k), k))
    pattern = re.compile("|".join(re.escape("{" + k + "}") for k in keys))
    return pattern.sub(lambda m: str(kw[m.group(0)[1:-1]]), template)
