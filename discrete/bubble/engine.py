"""Bubble's pipeline, free of UI: field text → retrieval → grounding → streamed writer.

Latency path (each step starts as soon as its input exists):

    field read (AX)         ~5 ms
    plan ‖ query embedding  ~0.6-1.2 s   (parallel; plan capped at PLAN_TIMEOUT_S)
    FTS + dense (local)     ~20 ms
    rerank                  ~0.3-0.6 s   (kept-alive connection)
    writer first token      ~0.5-1 s     (streamed, typed word by word)

versus WorkAA's 4-round research tool loop (5-15 s before the writer started).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Callable

from .. import textkit as tk
from ..config import T, USER_NAME, writer_endpoint
from ..llm import stream
from ..retrieval.grounding import GroundingDecision, evaluate
from ..retrieval.search import Hit, RetrievalResult, retrieve
from ..store import Store
from ..timeutil import now, to_local
from .prompts import MODE_INSTRUCTIONS, WRITER_SYSTEM, WRITER_USER

_PARA = re.compile(r"\n\s*\n")
_SENT_START = re.compile(r"(?<=[.!?])\s+")
_NO_SPACE_BEFORE = set(".,;:!?)]}'\"…")


@dataclass
class Prepared:
    field_text: str
    query: str
    app: str
    field_desc: str
    result: RetrievalResult
    decision: GroundingDecision
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def mode(self) -> str:
        return self.decision.mode


def field_query(text: str, max_chars: int | None = None) -> str:
    """The part of the field that is the actual request: its last paragraph,
    trimmed to a sentence boundary when long (a draft's opening is noise)."""
    max_chars = max_chars or T.BUBBLE_QUERY_CHARS
    t = (text or "").strip()
    if not t:
        return ""
    para = _PARA.split(t)[-1].strip() or t
    if len(para) > max_chars:
        tail = para[-max_chars:]
        m = _SENT_START.search(tail)
        para = tail[m.end():] if m and m.end() < len(tail) - 20 else tail
    return para.strip()


def needs_space(field_text: str, first: str) -> bool:
    return bool(field_text) and not field_text[-1].isspace() and bool(first) and first[0] not in _NO_SPACE_BEFORE


def _when(h: Hit) -> str:
    if not h.occurred_at:
        return ""
    a = to_local(h.occurred_at)
    if h.occurred_to and h.occurred_to - h.occurred_at > 120:
        b = to_local(h.occurred_to)
        return f" · {a:%a %b %d %H:%M}–{b:%H:%M}"
    return f" · {a:%a %b %d %H:%M}"


def render_records(result: RetrievalResult, *, max_chars: int | None = None) -> str:
    """Evidence pack for the writer: typed, time-stamped, id-tagged, budgeted."""
    max_chars = max_chars or T.BUBBLE_CONTEXT_CHARS
    terms = set(result.plan.terms)
    blocks: list[str] = []
    used = 0
    if result.digest:
        blocks.append(f"[day digest]\n{tk.truncate(result.digest, 1200)}")
        used += len(blocks[-1])
    for h in result.hits:
        if h.ref_kind == "claim":
            head = f"[claim {h.ref_id} · {h.claim_kind} · {h.modality}{_when(h)}]"
            body = h.body
        elif h.ref_kind == "observation":
            head = f"[screen {h.ref_id}{_when(h)} · {h.title}]"
            caption, _, screen = h.body.partition("\n")
            snippet = tk.best_window(screen, terms, width=110)[0] if terms and screen else screen
            body = "\n".join(p for p in (caption, tk.truncate(snippet or screen, 900)) if p)
        elif h.ref_kind == "episode":
            head = f"[episode {h.ref_id}{_when(h)}] {h.title}"
            body = h.body
        else:
            head = f"[entity {h.ref_id}] {h.title}"
            body = h.body
        block = f"{head}\n{tk.truncate(body, 1100)}"
        if used + len(block) > max_chars and blocks:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks) or "(no records)"


class BubbleEngine:
    def __init__(self, store: Store, user_name: str = USER_NAME):
        self.store = store
        self.user_name = user_name

    async def prepare(self, field_text: str, *, app: str = "", field_desc: str = "",
                      progress: Callable[[str, object], None] | None = None) -> Prepared:
        t0 = time.perf_counter()
        query = field_query(field_text)
        result = await retrieve(self.store, query, field_text=field_text, progress=progress)
        decision = evaluate(result)
        return Prepared(field_text=field_text, query=query, app=app or "an app", field_desc=field_desc or "text field",
                        result=result, decision=decision, timings={"retrieve_s": time.perf_counter() - t0})

    def messages(self, p: Prepared) -> list[dict]:
        instruction = MODE_INSTRUCTIONS.get(p.mode, MODE_INSTRUCTIONS["grounded"])
        records = "(not needed)" if p.mode == "edit" else render_records(p.result)
        return [
            {"role": "system", "content": WRITER_SYSTEM.format(user_name=self.user_name)},
            {"role": "user", "content": WRITER_USER.format(
                user_name=self.user_name, app=p.app, field=p.field_desc, now=f"{to_local(now()):%a %b %d %Y %H:%M}",
                text=p.field_text, records=records, instruction=instruction)},
        ]

    async def write(self, p: Prepared, emit: Callable[[str], bool],
                    should_continue: Callable[[], bool] = lambda: True) -> str:
        """Stream the writer, emitting at word boundaries so the field never shows half a word.
        `emit` returns False when it could not insert, which stops the run."""
        t0 = time.perf_counter()
        buf, out, first = "", "", True

        def push(chunk: str) -> bool:
            nonlocal out, first
            if first:
                chunk = chunk.lstrip()
                if not chunk:
                    return True
                if needs_space(p.field_text, chunk):
                    chunk = " " + chunk
                first = False
                p.timings["first_chunk_s"] = time.perf_counter() - t0
            if not emit(chunk):
                return False
            out += chunk
            return True

        async for piece in stream(writer_endpoint(), self.messages(p), temperature=0.2,
                                  max_tokens=T.BUBBLE_WRITER_MAX_TOKENS):
            if not should_continue():
                return out
            buf += piece
            if len(buf) < T.BUBBLE_MIN_CHUNK:
                continue
            cut = max(buf.rfind(" "), buf.rfind("\n"))
            if cut <= 0:
                continue
            chunk, buf = buf[:cut], buf[cut:]
            if not push(chunk):
                return out
        tail = buf.rstrip()
        if tail and should_continue():
            push(tail)
        p.timings["write_s"] = time.perf_counter() - t0
        return out
