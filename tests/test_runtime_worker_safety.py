import asyncio

import pytest

from discrete.app import Runtime
from discrete.capture.enrich import Enricher
from discrete.models import ObservationDraft


async def test_ingest_store_error_reporting_cannot_kill_worker(store, monkeypatch):
    rt = Runtime(store, bubble=False)
    async def fail(*args):
        raise RuntimeError('store unavailable')
    monkeypatch.setattr(rt, '_ingest', fail)
    monkeypatch.setattr(store, 'bump_metric', fail)
    task = asyncio.create_task(rt._ingest_loop())
    await rt.queue.put(object())
    await asyncio.sleep(0.02)
    try:
        assert not task.done()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_essential_worker_exit_is_fatal(store):
    rt = Runtime(store, bubble=False)
    async def done():
        return
    rt._tasks = [asyncio.create_task(done(), name='essential-test')]
    with pytest.raises(RuntimeError, match='essential-test'):
        await asyncio.wait_for(rt.wait(), 1)


async def test_capture_queue_admission_is_bounded(store):
    rt = Runtime(store, bubble=False)
    assert rt.queue.maxsize > 0
    for _ in range(rt.queue.maxsize):
        rt.queue.put_nowait(object())
    with pytest.raises(asyncio.QueueFull):
        await rt.queue.put(object())


async def test_corrupt_pending_frame_does_not_block_valid_row(store, tmp_path, fake):
    from PIL import Image
    corrupt, valid = tmp_path / 'corrupt.jpg', tmp_path / 'valid.jpg'
    corrupt.write_bytes(b'not an image')
    Image.new('RGB', (16, 16)).save(valid)
    ids = []
    for i, path in enumerate((corrupt, valid)):
        oid, _ = await store.add_observation(ObservationDraft(
            app='test', window_title=str(i), screen_text='synthetic capture ' + str(i),
            caption='base', frame_path=str(path), enrich_state='pending',
            started_at=100 + i, ended_at=101 + i, session_id='test'))
        ids.append(oid)
    enricher = Enricher(store)
    try:
        assert await enricher.fill_pending() == 1
        await asyncio.gather(*enricher._tasks)
        assert (await store.get_observation(ids[0]))['enrich_state'] == 'failed'
        assert (await store.get_observation(ids[1]))['enrich_state'] == 'done'
    finally:
        await enricher.cancel_all()


async def test_finalizer_admission_closes_rejected_coroutine():
    from discrete.aio import spawn
    gate = asyncio.Event()
    owner = set()
    async def work():
        await gate.wait()
    first = spawn(work(), owner=owner, max_pending=1)
    rejected = work()
    assert spawn(rejected, owner=owner, max_pending=1) is None
    assert rejected.cr_frame is None
    gate.set()
    await first
