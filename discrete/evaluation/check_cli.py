"""CLI registration for the offline regression gate; never opens a writable Store."""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys

import yaml

from ..config import db_path
from .invariants import check_store


async def cmd_check(args: argparse.Namespace) -> int:
    try:
        report = await asyncio.to_thread(check_store, args.db or db_path(), expect=args.expect, slo=args.slo)
    except (OSError, ValueError, sqlite3.Error, yaml.YAMLError) as exc:
        print(f"check: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report.to_dict(), indent=2))
    return 0 if report.passed else 1


def register_check(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("check", help="read-only store invariants and optional latency SLOs")
    parser.add_argument("--expect", metavar="FILE", help="YAML expected claims and capture/chrome thresholds")
    parser.add_argument("--slo", action="store_true", help="check per-stage latency SLOs; unmeasured stages report n/a")
    parser.set_defaults(func=cmd_check)
