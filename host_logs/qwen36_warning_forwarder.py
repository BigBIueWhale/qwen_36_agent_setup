#!/usr/bin/env python3
"""Host-side forwarder for §7.5 ``model_emit_warning`` log lines.

Why this must exist
-------------------

Patch ``monkey_patch_tool_call_in_think_detector.py`` (§7.5) wraps
``Qwen3ReasoningParser.extract_reasoning`` and emits a single structured
WARNING for every detected ``<tool_call>`` substring inside the
reasoning half of the model's output (see the patch's ``logger.warning``
call at ``monkey_patch_tool_call_in_think_detector.py:83-88``). The
warning lands in vLLM's stderr, which docker captures via the
``json-file`` log driver — but nothing extracts it. Operators have no
way to query "how many tool-call-in-reasoning events fired in the last
hour" or alert on a sudden rate spike. README §8.4 has long mentioned
the gap; this script closes it.

What it does
------------

Tails ``docker logs -f`` of the qwen36 container, parses each
``model_emit_warning kind=tool_call_in_reasoning ...`` line with a
strict regex, and appends a structured JSON Lines record to
``/var/log/qwen36/warnings.jsonl`` (and to stdout, for systemd-journal
fan-out, unless ``--no-stdout`` is set).

Schema
------

One JSON object per line, append-only::

    {
      "ts":           "2026-04-28T13:14:15.234567+00:00",   // ISO-8601 UTC capture timestamp
      "kind":         "tool_call_in_reasoning",              // patch-defined event class (only one today)
      "reasoning_len": 487,                                  // len(reasoning) at the call site
      "marker_count":  2,                                    // reasoning.count("<tool_call>")
      "raw":          "WARNING 04-28 13:14:15 [...] model_emit_warning ..." // verbatim docker line
    }

Run (without systemd)
---------------------

::

    python3 qwen36_warning_forwarder.py \\
        --container qwen36 \\
        --output /var/log/qwen36/warnings.jsonl \\
        --state  /var/lib/qwen36/forwarder_state.json

The script runs as a long-lived foreground process. It exits 0 on
SIGTERM, propagates non-zero on irrecoverable startup errors (bad
permissions on ``--output``, etc.). For systemd deployment see
``qwen36-warning-forwarder.service``.

Operational signals
-------------------

* ``SIGTERM`` -> graceful shutdown: state is flushed, output file is
  closed, exit 0. A one-line ``shutdown:`` banner goes to stderr.
* ``SIGHUP`` -> log rotation: close and reopen the output file (so
  ``logrotate``-style ``copytruncate`` is unnecessary; we honor a real
  rotate sequence). Banner ``rotate:`` goes to stderr.
* ``docker logs -f`` exits when the container restarts; the forwarder
  detects EOF, increments ``restart_count`` in the state file, sleeps
  briefly to avoid CPU pegging on a crashloop, and re-spawns.

Strict / fail-loud idiom
------------------------

The regex is anchored on the patch's exact format string. A line that
contains the marker prefix ``model_emit_warning kind=`` but FAILS to
match the full regex is a parse error: it goes to stderr (so journald
captures it), increments ``parse_errors`` in the state file, and is
NOT silently dropped. Silent drops are precisely what makes monitoring
useless.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import FrameType
from typing import IO, Final, Optional

# ----------------------------------------------------------------------
# Strict regex anchored on the patch's exact format string
# ----------------------------------------------------------------------
#
# Source line under audit: ``monkey_patch_tool_call_in_think_detector.py``
# lines 83-88::
#
#     _logger.warning(
#         "model_emit_warning kind=tool_call_in_reasoning "
#         "reasoning_len=%d marker_count=%d",
#         len(reasoning),
#         reasoning.count(_TOOL_CALL_OPEN),
#     )
#
# vLLM's logger formats every record with a leading prefix; the docker
# json-file driver pre-pends nothing of substance. The shape we see in
# ``docker logs`` is, end-to-end::
#
#     WARNING MM-DD HH:MM:SS [<file>:<lineno>] model_emit_warning \
#       kind=tool_call_in_reasoning reasoning_len=<int> marker_count=<int>
#
# Anything before the marker phrase ``model_emit_warning kind=`` we treat
# as opaque prefix; anything after the structured triplet must match
# exactly. The ``$`` anchor rejects trailing junk (which would mean the
# format drifted upstream and our schema is stale).

_MARKER_PREFIX: Final[str] = "model_emit_warning kind="

# Pre-anchor pattern: match the suffix from the marker onwards. The full
# log line may have ``\r\n`` stripped already; ``$`` is anchored after a
# possible trailing ``\n`` strip in the loop.
_WARNING_RE: Final[re.Pattern[str]] = re.compile(
    r"model_emit_warning"
    r"\s+kind=(?P<kind>[A-Za-z0-9_]+)"
    r"\s+reasoning_len=(?P<reasoning_len>\d+)"
    r"\s+marker_count=(?P<marker_count>\d+)"
    r"\s*$"
)

_DEFAULT_CONTAINER: Final[str] = "qwen36"
_DEFAULT_OUTPUT: Final[str] = "/var/log/qwen36/warnings.jsonl"
_DEFAULT_STATE: Final[str] = "/var/lib/qwen36/forwarder_state.json"

# Cap docker-log restart hot loop to one attempt per `_RESTART_FLOOR_S`.
_RESTART_FLOOR_S: Final[float] = 2.0


# ----------------------------------------------------------------------
# State file (human-readable, single-source-of-truth)
# ----------------------------------------------------------------------


class State:
    """Persistent counters mirrored to ``--state`` via atomic rename.

    All numerical state lives here so an operator can ``cat`` the file
    and learn (a) when we last saw an event, (b) how many docker-logs
    restarts happened, (c) how many lines failed to parse. Anything not
    in this file is recoverable from it on the next start.
    """

    def __init__(self, path: Path) -> None:
        self.path: Path = path
        self.last_event_ts: Optional[str] = None
        self.restart_count: int = 0
        self.events_total: int = 0
        self.parse_errors: int = 0
        self.started_ts: str = _utc_now_iso()

        # Ensure parent directory exists; a fresh box won't have it.
        self.path.parent.mkdir(parents=True, exist_ok=True)

        if self.path.exists():
            self._load()
        else:
            # First boot — create the state file immediately so an
            # operator always has a file to ``cat`` and so monitoring
            # can detect a hung forwarder by stale ``started_ts``.
            self.flush()

    # -- IO -----------------------------------------------------------

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            # A corrupt state file is loud, not silent: refuse to start.
            # The operator can ``rm`` it to reset.
            raise SystemExit(
                f"qwen36-warning-forwarder: corrupt state at "
                f"{self.path}: {exc!r}. Remove and restart."
            ) from exc
        self.last_event_ts = data.get("last_event_ts")
        self.restart_count = int(data.get("restart_count", 0))
        self.events_total = int(data.get("events_total", 0))
        self.parse_errors = int(data.get("parse_errors", 0))

    def flush(self) -> None:
        """Atomic write via tempfile + ``os.replace``."""
        payload = {
            "last_event_ts": self.last_event_ts,
            "restart_count": self.restart_count,
            "events_total": self.events_total,
            "parse_errors": self.parse_errors,
            "started_ts": self.started_ts,
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, self.path)


def _utc_now_iso() -> str:
    """ISO-8601 with microseconds and explicit UTC offset."""
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


# ----------------------------------------------------------------------
# Forwarder
# ----------------------------------------------------------------------


class Forwarder:
    """Single forwarder instance. Long-lived; one container; one output."""

    def __init__(
        self,
        *,
        container: str,
        output_path: Path,
        state_path: Path,
        emit_stdout: bool,
        log_level: str,
    ) -> None:
        self.container: str = container
        self.output_path: Path = output_path
        self.state: State = State(state_path)
        self.emit_stdout: bool = emit_stdout

        logging.basicConfig(
            level=getattr(logging, log_level.upper(), logging.INFO),
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            stream=sys.stderr,
        )
        self.log: logging.Logger = logging.getLogger("qwen36-warnings")

        self._output_fp: Optional[IO[str]] = None
        self._stop_requested: bool = False
        self._rotate_requested: bool = False
        self._docker_proc: Optional[subprocess.Popen[str]] = None

    # -- output file lifecycle ---------------------------------------

    def _open_output(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        # ``a`` so we never clobber prior records; line buffering so a
        # crash leaves the most recent record durable on disk.
        self._output_fp = open(  # noqa: SIM115 — long-lived handle
            self.output_path, "a", buffering=1, encoding="utf-8"
        )

    def _close_output(self) -> None:
        if self._output_fp is not None:
            try:
                self._output_fp.flush()
                self._output_fp.close()
            finally:
                self._output_fp = None

    def _rotate_output(self) -> None:
        self.log.info("rotate: closing %s", self.output_path)
        self._close_output()
        self._open_output()
        print(
            f"rotate: pid={os.getpid()} output={self.output_path}",
            file=sys.stderr,
            flush=True,
        )

    # -- record emission ---------------------------------------------

    def _emit_record(self, record: dict[str, object]) -> None:
        line = json.dumps(record, ensure_ascii=False)
        if self._output_fp is None:
            # Should be unreachable; defensive — a vanished handle is
            # loud, not silent.
            raise RuntimeError("output file is not open")
        self._output_fp.write(line + "\n")
        if self.emit_stdout:
            print(line, flush=True)

    # -- per-line parser ---------------------------------------------

    def handle_line(self, raw: str) -> None:
        """Parse a single docker-logs line and route it.

        The line is the *exact* bytes from docker. We strip trailing
        newlines but otherwise preserve it in the ``raw`` field so the
        operator can recover the original log.
        """
        line = raw.rstrip("\r\n")
        if _MARKER_PREFIX not in line:
            return  # not ours; silent skip is correct here
        match = _WARNING_RE.search(line)
        if match is None:
            # The marker phrase is present but the structure didn't
            # parse. This is a LOUD failure — the patch's format string
            # may have drifted, and silent skip would hide the drift.
            self.state.parse_errors += 1
            self.state.flush()
            self.log.error(
                "parse_error: marker present but regex failed "
                "(parse_errors=%d) raw=%r",
                self.state.parse_errors,
                line,
            )
            return
        try:
            record = {
                "ts": _utc_now_iso(),
                "kind": match.group("kind"),
                "reasoning_len": int(match.group("reasoning_len")),
                "marker_count": int(match.group("marker_count")),
                "raw": line,
            }
        except (ValueError, KeyError) as exc:
            # Regex matched but ints failed — same loud-failure stance.
            self.state.parse_errors += 1
            self.state.flush()
            self.log.error(
                "parse_error: post-match coercion failed (%r) raw=%r",
                exc,
                line,
            )
            return

        self._emit_record(record)
        self.state.events_total += 1
        self.state.last_event_ts = record["ts"]  # type: ignore[assignment]
        # Flush state every event — events are rare (single-digit %),
        # so the IO cost is negligible vs. the operator value of an
        # always-current state file.
        self.state.flush()

    # -- docker-logs subprocess loop ---------------------------------

    def _spawn_docker_logs(self) -> subprocess.Popen[str]:
        """Spawn ``docker logs -f --tail=0`` in line-buffered text mode.

        ``--tail=0`` is correct on cold start and on every restart: we
        do not want to replay history (the state file already records
        what's been processed and the WARNING is informational, not a
        durable command). Replay would also double-count if the
        operator manually started/stopped this script.
        """
        cmd = [
            "docker",
            "logs",
            "-f",
            "--tail=0",
            self.container,
        ]
        return subprocess.Popen(  # noqa: S603 — fixed argv, no shell
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # line-buffered
        )

    def _consume_until_eof_or_signal(self, proc: subprocess.Popen[str]) -> None:
        assert proc.stdout is not None
        for raw in proc.stdout:
            if self._stop_requested:
                return
            if self._rotate_requested:
                self._rotate_requested = False
                self._rotate_output()
            self.handle_line(raw)

    def run(self) -> int:
        self._open_output()

        # Banner BEFORE we install signal handlers, so a SIGTERM during
        # banner print doesn't double-shutdown.
        print(
            f"started: pid={os.getpid()} container={self.container} "
            f"output={self.output_path} state={self.state.path}",
            file=sys.stderr,
            flush=True,
        )

        signal.signal(signal.SIGTERM, self._on_sigterm)
        signal.signal(signal.SIGINT, self._on_sigterm)
        signal.signal(signal.SIGHUP, self._on_sighup)

        try:
            while not self._stop_requested:
                last_spawn = time.monotonic()
                self._docker_proc = self._spawn_docker_logs()
                try:
                    self._consume_until_eof_or_signal(self._docker_proc)
                finally:
                    if self._docker_proc.poll() is None:
                        self._docker_proc.terminate()
                        try:
                            self._docker_proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            self._docker_proc.kill()
                            self._docker_proc.wait(timeout=5)
                if self._stop_requested:
                    break
                # docker logs exited — container restart, daemon
                # restart, or transient error. Rate-limit re-spawn.
                self.state.restart_count += 1
                self.state.flush()
                elapsed = time.monotonic() - last_spawn
                if elapsed < _RESTART_FLOOR_S:
                    time.sleep(_RESTART_FLOOR_S - elapsed)
                self.log.warning(
                    "docker logs exited (restart_count=%d); re-spawning",
                    self.state.restart_count,
                )
        finally:
            self._close_output()
            self.state.flush()
            print(
                f"shutdown: events={self.state.events_total} "
                f"restarts={self.state.restart_count} "
                f"parse_errors={self.state.parse_errors}",
                file=sys.stderr,
                flush=True,
            )
        return 0

    # -- signal handlers ----------------------------------------------

    def _on_sigterm(self, signum: int, frame: Optional[FrameType]) -> None:
        self._stop_requested = True
        # Best-effort wake the docker proc so ``for raw in stdout:``
        # returns. ``terminate()`` is idempotent; the run-loop also
        # terminates it on exit.
        if self._docker_proc is not None and self._docker_proc.poll() is None:
            try:
                self._docker_proc.terminate()
            except ProcessLookupError:
                pass

    def _on_sighup(self, signum: int, frame: Optional[FrameType]) -> None:
        self._rotate_requested = True


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="qwen36_warning_forwarder",
        description=(
            "Forward §7.5 model_emit_warning lines from `docker logs -f` "
            "into a structured JSON Lines file. Strict, fail-loud, "
            "long-lived; intended to run under systemd."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--container",
        default=_DEFAULT_CONTAINER,
        help="Docker container name to tail.",
    )
    p.add_argument(
        "--output",
        default=_DEFAULT_OUTPUT,
        help="Append-only JSON Lines output path.",
    )
    p.add_argument(
        "--state",
        default=_DEFAULT_STATE,
        help="State file path (atomic-rewritten on every event).",
    )
    p.add_argument(
        "--no-stdout",
        action="store_true",
        help=(
            "Suppress stdout JSON emission. Use under systemd when only "
            "the file output is wanted (the file is the source of truth; "
            "stdout is for journald fan-out)."
        ),
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Stderr log level for the forwarder's own diagnostics.",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    fwd = Forwarder(
        container=args.container,
        output_path=Path(args.output),
        state_path=Path(args.state),
        emit_stdout=not args.no_stdout,
        log_level=args.log_level,
    )
    return fwd.run()


if __name__ == "__main__":
    sys.exit(main())
