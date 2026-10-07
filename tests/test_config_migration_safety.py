import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from discrete.config import ConfigError, _typed, caption_endpoint
from discrete.store.db import SCHEMA_VERSION, migrate


@pytest.mark.parametrize('name,default,raw', [
    ('REDACT_SECRETS', True, 'tru'), ('ENRICH_CONCURRENCY', 2, '0'),
    ('ENRICH_CONCURRENCY', 2, 'wrong'), ('POLL_S', 0.5, 'nan'),
    ('POLL_S', 0.5, 'inf'), ('POLL_S', 0.5, '0'),
    ('FRAME_JPEG_QUALITY', 65, '101'), ('MMR_LAMBDA', 0.7, '1.1'),
])
def test_invalid_tunables_fail_with_setting_name(monkeypatch, name, default, raw):
    monkeypatch.setenv('DISCRETE_' + name, raw)
    with pytest.raises(ConfigError, match=name):
        _typed(name, default)


@pytest.mark.parametrize('field,value', [('JSON_MODE', 'tru'), ('TIMEOUT_S', 'nan'),
                                         ('TIMEOUT_S', '0'), ('EXTRA_TOKENS', '-1'),
                                         ('TEMPERATURE', 'inf')])
def test_invalid_endpoint_settings_fail(monkeypatch, field, value):
    monkeypatch.setenv('CAPTION_LM_' + field, value)
    with pytest.raises(ConfigError, match=field):
        caption_endpoint()


def test_intentional_zero_and_false_settings(monkeypatch):
    monkeypatch.setenv('DISCRETE_DREAM_INTERVAL_MIN', '0')
    monkeypatch.setenv('DISCRETE_REDACT_SECRETS', 'false')
    assert _typed('DREAM_INTERVAL_MIN', 1440.0) == 0
    assert _typed('REDACT_SECRETS', True) is False


def test_future_schema_is_refused():
    conn = sqlite3.connect(':memory:', isolation_level=None)
    conn.execute(f'PRAGMA user_version={SCHEMA_VERSION + 1}')
    with pytest.raises(ValueError, match='newer'):
        migrate(conn)
    assert not conn.in_transaction
    conn.close()


def test_concurrent_migration_decision_is_serialized(tmp_path):
    path = str(tmp_path / 'race.db')
    def run():
        conn = sqlite3.connect(path, isolation_level=None, timeout=10)
        try:
            return migrate(conn)
        finally:
            conn.close()
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda _: run(), range(2)))
    assert sorted(results) == [0, SCHEMA_VERSION]


def test_conflicting_pool_limits_fail_at_startup(monkeypatch):
    from discrete.config import T, validate
    monkeypatch.setattr(T, 'ENRICH_CONCURRENCY', T.ENRICH_MAX_PENDING + 1)
    with pytest.raises(ConfigError, match='ENRICH_CONCURRENCY'):
        validate()
