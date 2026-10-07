"""Serve DIscrete's real pipeline over HTTP so the Agent Memory Benchmark (AMB) can score it.

AMB (github.com/vectorize-io/agent-memory-benchmark) feeds documents to a memory
provider and asks it questions. DIscrete only knows screen observations, so each
benchmark session becomes one observation of a messaging window -- the user reading
that conversation on screen -- stamped at the session's date, and closes its own
episode. Drafting, verification, embedding and consolidation then run exactly as in
`discrete run`, with the clock set to the session's date (a virtual clock: the
benchmark spans months, and DIscrete reads "now" for digests, ages and intervals).

Questions go through Bubble's pipeline (`discrete ask`: plan, hybrid search, rerank,
support check, streamed writer) or, for AMB's rag mode, retrieval alone; the clock is
the question's date. One SQLite store per isolation unit (e.g. per LoCoMo conversation).

    .venv/bin/python tools/amb_bridge.py --port 8891

Endpoints (JSON in, JSON out): POST /prepare, /ingest, /retrieve, /answer.
"""

from __future__ import annotations

import argparse
import asyncio
import contextvars
import json
import logging
import shutil
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from discrete import timeutil

_CLOCK: contextvars.ContextVar[float | None] = contextvars.ContextVar("amb_clock", default=None)
_real_now = timeutil.now


def _virtual_now() -> float:
    at = _CLOCK.get()
    return at if at is not None else _real_now()


def _install_clock() -> None:
    """Point every `now` imported from timeutil at the virtual clock."""
    for name, module in list(sys.modules.items()):
        if (name == "discrete" or name.startswith("discrete.")) and getattr(module, "now", None) is _real_now:
            module.now = _virtual_now


# Imported after timeutil so _install_clock sees every module that bound `now`.
from discrete.bubble.engine import BubbleEngine, EmptyField, _when, render_records  # noqa: E402
from discrete.llm import flush_usage  # noqa: E402
from discrete.models import ObservationDraft  # noqa: E402
from discrete.process.consolidate import run_consolidation  # noqa: E402
from discrete.process.embed_worker import embed_row, observation_text  # noqa: E402
from discrete.process.episodes import EpisodeTracker  # noqa: E402
from discrete.process.pipeline import ProcessPipeline  # noqa: E402
from discrete.retrieval.hybrid import retrieve  # noqa: E402
from discrete.store import Store  # noqa: E402

_install_clock()
log = logging.getLogger("amb_bridge")

#: DIscrete's per-observation text cap; a longer session is split into consecutive
#: observations of the same window, one minute apart, like scrolling through it.
OBS_CHARS = 15_000
DEFAULT_UNIT = "_default"


def _epoch(stamp: str | None, fallback: float) -> float:
    if not stamp:
        return fallback
    try:
        return datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return fallback


def _pieces(text: str) -> list[str]:
    out, rest = [], text
    while len(rest) > OBS_CHARS:
        cut = rest.rfind("\n", 0, OBS_CHARS)
        cut = cut if cut > OBS_CHARS // 2 else OBS_CHARS
        out.append(rest[:cut])
        rest = rest[cut:].lstrip("\n")
    if rest.strip():
        out.append(rest)
    return out


class Bridge:
    def __init__(self) -> None:
        self.root: Path | None = None
        self.stores: dict[str, Store] = {}
        self.locks: dict[str, asyncio.Lock] = {}

    async def prepare(self, store_dir: str, reset: bool = True, **_: object) -> dict:
        for store in self.stores.values():
            await store.close()
        self.stores.clear()
        self.root = Path(store_dir) / "discrete"
        if reset and self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        return {"root": str(self.root)}

    async def _store(self, unit: str | None) -> Store:
        unit = unit or DEFAULT_UNIT
        if unit not in self.stores:
            if self.root is None:
                raise RuntimeError("call /prepare first")
            safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in unit)
            self.stores[unit] = await Store.open(str(self.root / f"{safe}.sqlite3"))
            self.locks[unit] = asyncio.Lock()
        return self.stores[unit]

    async def ingest(self, documents: list[dict], **_: object) -> dict:
        by_unit: dict[str, list[dict]] = {}
        for doc in documents:
            by_unit.setdefault(doc.get("user_id") or DEFAULT_UNIT, []).append(doc)
        report = {}
        for unit, docs in by_unit.items():
            report[unit] = await self._ingest_unit(unit, docs)
        return {"units": report}

    async def _ingest_unit(self, unit: str, docs: list[dict]) -> dict:
        store = await self._store(unit)
        tracker, pipe = EpisodeTracker(store), ProcessPipeline(store)
        base = time.time() - 86400 * 365
        docs = sorted(docs, key=lambda d: _epoch(d.get("timestamp"), base))
        counts = {"sessions": 0, "observations": 0, "drafted": 0}
        async with self.locks[unit]:
            for n, doc in enumerate(docs):
                start = _epoch(doc.get("timestamp"), base + n * 3600)
                title = (doc.get("context") or doc["id"])[:200]
                for i, text in enumerate(_pieces(doc.get("content") or "")):
                    at = start + 60 * i
                    _CLOCK.set(at)
                    draft = ObservationDraft(
                        source="screen", origin="user", event_kind="heartbeat", session_id=doc["id"],
                        started_at=at, ended_at=at + 60, app="Messages", window_title=title,
                        screen_text=text, text_source="ax",
                    )
                    oid, new = await store.add_observation(draft)
                    if not new or oid is None:
                        continue
                    row = await store.get_observation(oid)
                    if row is None or row["origin"] != "user":
                        continue
                    vec = await embed_row(store, "observation", oid, observation_text(row))
                    await tracker.on_observation(oid, row, session_id=doc["id"], vec=vec)
                    counts["observations"] += 1
                # The episode closes, is drafted, and the hourly consolidation runs before the
                # next session -- sessions are days apart, so `discrete run` would have done both.
                _CLOCK.set(start + 3600)
                await tracker.close_current()
                counts["drafted"] += len(await pipe.process_ready(limit=1000))
                await run_consolidation(store)
                counts["sessions"] += 1
                log.info("%s: session %d/%d ingested", unit, n + 1, len(docs))
            await flush_usage(store)
        counts["stats"] = {k: v for k, v in (await store.stats()).items() if isinstance(v, (int, float))}
        return counts

    async def retrieve(self, query: str, k: int = 10, user_id: str | None = None,
                       query_timestamp: str | None = None, **_: object) -> dict:
        store = await self._store(user_id)
        _CLOCK.set(_epoch(query_timestamp, _real_now()))
        result = await retrieve(store, query, field_text=query)
        docs = [
            {"id": f"{h.ref_kind}:{h.ref_id}", "content": f"[{_when(h)}] ({h.ref_kind}) {h.title}\n{h.body}".strip()}
            for h in result.hits[:k]
        ]
        await flush_usage(store)
        return {"documents": docs, "raw": {"intent": result.plan.intent, "hits": len(result.hits)}}

    async def answer(self, query: str, user_id: str | None = None, query_timestamp: str | None = None,
                     **_: object) -> dict:
        store = await self._store(user_id)
        _CLOCK.set(_epoch(query_timestamp, _real_now()))
        engine = BubbleEngine(store)
        try:
            p = await engine.prepare(query)
        except EmptyField:
            return {"answer": "", "context": "", "raw": {"mode": "empty-field"}}
        context = render_records(p.result)
        if p.mode == "empty":
            answer = "(nothing on file about that)"
        else:
            parts: list[str] = []
            await engine.write(p, lambda chunk: parts.append(chunk) or True)
            answer = "".join(parts).strip()
        await flush_usage(store)
        return {"answer": answer, "context": context,
                "raw": {"mode": p.mode, "intent": p.result.plan.intent, "hits": len(p.result.hits),
                        "timings": p.timings}}


def serve(port: int) -> None:
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    bridge = Bridge()
    routes = {"/prepare": bridge.prepare, "/ingest": bridge.ingest, "/retrieve": bridge.retrieve,
              "/answer": bridge.answer}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - http.server API
            route = routes.get(self.path)
            if route is None:
                self.send_error(404)
                return
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            try:
                _install_clock()  # modules imported lazily since the last request
                fut = asyncio.run_coroutine_threadsafe(route(**body), loop)
                payload, status = fut.result(), 200
            except Exception as exc:  # noqa: BLE001 - reported to the caller
                log.exception("%s failed", self.path)
                payload, status = {"error": f"{type(exc).__name__}: {exc}"}, 500
            data = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args: object) -> None:
            pass

    print(f"DIscrete AMB bridge on http://127.0.0.1:{port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8891)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    serve(parser.parse_args().port)
