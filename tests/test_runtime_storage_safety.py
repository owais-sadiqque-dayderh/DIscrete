import asyncio
import threading

import pytest

from discrete.store.base import StoreBase


async def test_repeated_cancel_holds_sql_lock(store):
    entered, release = threading.Event(), threading.Event()
    def blocked():
        entered.set()
        release.wait(3)
    task = asyncio.create_task(store._call(blocked))
    await asyncio.to_thread(entered.wait, 1)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0.01)
    try:
        assert store._lock.locked()
        assert not task.done()
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task


def test_commit_failure_rolls_back(monkeypatch):
    class Connection:
        def __init__(self):
            self.commands = []
        def execute(self, sql):
            self.commands.append(sql)
            if sql == 'COMMIT':
                raise RuntimeError('commit failed')
    conn = Connection()
    monkeypatch.setattr('discrete.store.pages.sync_pages', lambda c: None)
    store = StoreBase(conn)
    store._dirty_kinds.add('claim')
    with pytest.raises(RuntimeError, match='commit failed'):
        with store._tx():
            pass
    assert conn.commands[-1] == 'ROLLBACK'
    assert not store._dirty_kinds


async def test_cli_session_closes_after_flush_failure(monkeypatch):
    from discrete import cli
    closed = []
    class Store:
        async def close(self):
            closed.append(True)
    async def open_store():
        return Store()
    async def fail(store):
        raise RuntimeError('flush failed')
    monkeypatch.setattr(cli.Store, 'open', open_store)
    monkeypatch.setattr(cli, 'flush_usage', fail)
    with pytest.raises(RuntimeError, match='flush failed'):
        async with cli._session():
            pass
    assert closed
