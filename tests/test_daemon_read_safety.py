import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from discrete import aio
from discrete.cli import build_parser


def test_data_dir_singleton_and_stale_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv('DISCRETE_DATA_DIR', str(tmp_path))
    with aio.daemon_lease():
        assert aio.daemon_status()['running']
        with pytest.raises(RuntimeError, match='already running'):
            with aio.daemon_lease():
                pass
    assert not aio.daemon_status()['running']
    # A stale/reused PID file cannot claim ownership without the live lock.
    (tmp_path / 'runtime.lock').write_text(json.dumps({'pid': os.getpid(), 'identity': 'wrong'}))
    assert not aio.daemon_status()['running']


def test_separate_data_dirs_do_not_conflict(tmp_path, monkeypatch):
    monkeypatch.setenv('DISCRETE_DATA_DIR', str(tmp_path / 'a'))
    with aio.daemon_lease():
        monkeypatch.setenv('DISCRETE_DATA_DIR', str(tmp_path / 'b'))
        with aio.daemon_lease():
            assert aio.daemon_status()['running']


def test_daemon_stop_waits_for_owned_shutdown(tmp_path):
    env = os.environ | {'DISCRETE_DATA_DIR': str(tmp_path)}
    proc = subprocess.Popen([sys.executable, '-c', '''
import signal, time
from discrete.aio import daemon_lease
with daemon_lease():
    signal.signal(signal.SIGTERM, lambda *_: (time.sleep(.2), exit(0)))
    print('ready', flush=True)
    while True: time.sleep(.1)
'''], env=env, stdout=subprocess.PIPE, text=True)
    try:
        assert proc.stdout.readline().strip() == 'ready'
        result = subprocess.run([sys.executable, '-c', 'from discrete.aio import daemon_stop; daemon_stop(timeout=3)'],
                                env=env, capture_output=True, text=True, timeout=5)
        assert result.returncode == 0, result.stderr
        assert proc.wait(timeout=1) == 0
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=5)


async def test_read_sql_uses_resolver_and_rejects_writes(store, tmp_db, monkeypatch, capsys):
    from discrete.cli import cmd_read_sql
    monkeypatch.setenv('DISCRETE_DB', tmp_db)
    args = build_parser().parse_args(['read-sql', "SELECT 'synthetic', 7"])
    assert await cmd_read_sql(args) == 0
    assert capsys.readouterr().out.strip() == 'synthetic\t7'
    args.sql = "DELETE FROM observations"
    with pytest.raises(Exception, match='readonly'):
        await cmd_read_sql(args)


def test_jarvis_has_no_broad_process_or_sqlite_patterns():
    script = Path('jarvis').read_text()
    assert 'pgrep' not in script and 'pkill' not in script
    assert 'q() { sqlite3' not in script


def test_encrypted_benchmark_clone_and_reads_use_fake_keychain(tmp_path, monkeypatch, capsys):
    pytest.importorskip('sqlcipher3')
    import asyncio

    from bench.redraft import redraft
    from bench.run_g46 import eval_cost
    from discrete import encryption as enc
    from discrete.cli import cmd_read_sql
    from discrete.store.db import clone_database, connect

    class FakeKeys:
        def __init__(self):
            self.keys = {}
        def put(self, key_id, value):
            self.keys[key_id] = value
        def get(self, key_id):
            return self.keys[key_id]
    monkeypatch.setattr(enc, '_keychain', FakeKeys())
    monkeypatch.setenv('DISCRETE_ENCRYPT_AT_REST', 'true')
    source = tmp_path / 'source.db'
    conn = connect(str(source))
    conn.execute("INSERT INTO perf_calls(at,src,component,model,status,ms,cost) VALUES (100,'eval','test','test','ok',1,.25)")
    conn.close()
    monkeypatch.setenv('DISCRETE_ENCRYPT_AT_REST', 'false')
    target = tmp_path / 'copy.db'
    clone_database(source, target)
    assert not target.read_bytes().startswith(enc.SQLITE_HEADER)
    assert enc.state_path(target).exists()
    assert eval_cost(target, 0) == .25
    monkeypatch.setenv('DISCRETE_DB', str(target))
    args = build_parser().parse_args(['read-sql', 'SELECT cost FROM perf_calls'])
    assert asyncio.run(cmd_read_sql(args)) == 0
    assert capsys.readouterr().out.strip() == '0.25'
    redrafted = tmp_path / 'redrafted.db'
    redraft(source, redrafted)
    assert not redrafted.read_bytes().startswith(enc.SQLITE_HEADER)
