"""A total foreground budget also bounds provider calls already in flight."""
import asyncio
from unittest.mock import Mock

from discrete.bubble import service
from discrete.capture.accessibility import FocusSnapshot


async def test_total_foreground_deadline_cancels_slow_preparation(store, monkeypatch):
    bubble = service.BubbleService(store, overlay=Mock())
    snap = FocusSnapshot(pid=7, app='Notes', value='Question?')
    cancelled = []
    async def prepare(*args, **kwargs):
        try:
            await asyncio.sleep(10)
        finally:
            cancelled.append(True)
    monkeypatch.setattr(bubble.engine, 'prepare', prepare)
    monkeypatch.setattr(service, '_FOREGROUND_TIMEOUT_S', 0.01)
    monkeypatch.setattr(service, 'hold', lambda: None)
    monkeypatch.setattr(service, 'release', lambda: None)
    monkeypatch.setattr(service, 'denied', lambda *args, **kwargs: '')
    await asyncio.wait_for(bubble._run(snap), 0.1)
    assert cancelled and bubble.stats['failed'] == 1
    assert bubble.stats['cancelled'] == 0
    bubble.overlay.message.assert_called_with('Took too long — stopped', tone='warn')


def test_writer_sees_a_bounded_field_ending_at_the_caret(store):
    from discrete.bubble.engine import BubbleEngine, Prepared
    from discrete.models import QueryPlan, RetrievalResult
    from discrete.retrieval.grounding import GroundingDecision

    field = 'Older draft paragraph. ' * 50_000 + 'What did I decide about helium?'
    p = Prepared(field, 'q', 'Notes', 'text field', RetrievalResult('q', QueryPlan('q')),
                 GroundingDecision('empty', 0, 0))
    user = BubbleEngine(store).messages(p)[1]['content']
    assert len(user) < 20_000 and 'What did I decide about helium?' in user


def test_relative_anchor_sends_event_lines_not_the_whole_screen():
    from discrete.models import AttentionMoment
    from discrete.retrieval.temporal import _hit

    screen = 'unrelated output line\n' * 5000 + '$ deployctl publish --prod\n' + 'trailing log\n' * 2000
    moment = AttentionMoment(1, 1, 10.0, 20.0, {'screen_text': screen, 'caption': 'A terminal', 'window_title': 'zsh'},
                             app='Terminal', session_id='s')
    hit = _hit(moment, via='anchor', terms=['deployctl', 'publish'])
    assert 'deployctl publish' in hit.body and len(hit.body) < 2000
