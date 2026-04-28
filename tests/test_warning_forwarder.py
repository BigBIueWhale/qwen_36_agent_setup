"""Tests for ``host_ops/qwen36_warning_forwarder.py``.

Runs without docker, network, or GPU. Synthesises log lines straight
into the parser, drives signals via Python's ``signal`` module, and
asserts the output JSONL / state file contents. Mirrors
``tests/test_patches_against_master.py`` style: a class-based
``TestRun`` accumulator, no pytest dependency.

Invoke with::

    python3 tests/test_warning_forwarder.py

Exit 0 iff every check passes.
"""

from __future__ import annotations

import io
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
FORWARDER = REPO_ROOT / "host_ops" / "qwen36_warning_forwarder.py"

# Make the forwarder importable without installing.
sys.path.insert(0, str(REPO_ROOT / "host_ops"))

import qwen36_warning_forwarder as fwd  # noqa: E402  — path set above


# --------------------------------------------------------------------
# Test reporting (mirrors test_patches_against_master.TestRun)
# --------------------------------------------------------------------

_GREEN = "\033[32m"
_RED = "\033[31m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


class TestRun:
    """Accumulates test outcomes; non-zero exit code iff any fail."""

    def __init__(self) -> None:
        self.passed: list[str] = []
        self.failed: list[tuple[str, str]] = []

    def expect(self, name: str, ok: bool, detail: str = "") -> None:
        if ok:
            self.passed.append(name)
            print(f"  {_GREEN}PASS{_RESET} {name}")
        else:
            self.failed.append((name, detail))
            print(f"  {_RED}FAIL{_RESET} {name}")
            if detail:
                for line in detail.splitlines():
                    print(f"       {line}")

    def expect_eq(self, name: str, actual: Any, expected: Any) -> None:
        ok = actual == expected
        self.expect(
            name,
            ok,
            "" if ok else f"expected: {expected!r}\nactual:   {actual!r}",
        )

    def section(self, title: str) -> None:
        print(f"\n{_BOLD}{title}{_RESET}")

    def report(self) -> int:
        print()
        if self.failed:
            print(
                f"{_BOLD}{_RED}FAILED{_RESET}: "
                f"{len(self.passed)} passed, {len(self.failed)} failed"
            )
            for name, _ in self.failed:
                print(f"  - {name}")
            return 1
        print(
            f"{_BOLD}{_GREEN}OK{_RESET}: "
            f"{len(self.passed)} passed, 0 failed"
        )
        return 0


run = TestRun()


# --------------------------------------------------------------------
# Fixtures: synthesized docker-log lines that match the §7.5 format
# --------------------------------------------------------------------


def make_warning_line(reasoning_len: int, marker_count: int) -> str:
    """Reproduce the exact shape vLLM's logger.warning + docker prepend.

    The exact format string lives at ``monkey_patch_tool_call_in_think_detector.py``
    lines 83-88; see ``qwen36_warning_forwarder._WARNING_RE`` for the
    machine parser.
    """
    return (
        "WARNING 04-28 13:14:15 "
        "[monkey_patch_tool_call_in_think_detector.py:84] "
        "model_emit_warning kind=tool_call_in_reasoning "
        f"reasoning_len={reasoning_len} marker_count={marker_count}"
    )


# --------------------------------------------------------------------
# Helpers — build a Forwarder against tempfile output/state, no docker
# --------------------------------------------------------------------


def make_forwarder(
    tmpdir: Path, *, emit_stdout: bool = False
) -> fwd.Forwarder:
    output = tmpdir / "warnings.jsonl"
    state = tmpdir / "forwarder_state.json"
    f = fwd.Forwarder(
        container="qwen36",
        output_path=output,
        state_path=state,
        emit_stdout=emit_stdout,
        log_level="ERROR",  # quiet during tests; parse_error tests re-enable
    )
    # Open output as the run-loop normally would.
    f._open_output()
    return f


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


# --------------------------------------------------------------------
# Section 1: Strict-regex sanity
# --------------------------------------------------------------------


def section_1_regex() -> None:
    run.section("1. Strict regex matches §7.5 format string")
    line = make_warning_line(487, 2)
    m = fwd._WARNING_RE.search(line)
    run.expect("regex matches a well-formed warning line", m is not None)
    if m is not None:
        run.expect_eq("kind group", m.group("kind"), "tool_call_in_reasoning")
        run.expect_eq("reasoning_len group", m.group("reasoning_len"), "487")
        run.expect_eq("marker_count group", m.group("marker_count"), "2")

    # Trailing junk must be rejected (loud failure, not silent skip).
    bad = make_warning_line(1, 1) + " UNEXPECTED_TRAILER"
    run.expect(
        "regex rejects trailing junk after marker_count",
        fwd._WARNING_RE.search(bad) is None,
    )

    # Marker prefix is the gating substring used to short-circuit non-§7.5
    # lines; verify the constant is what we expect (locks against drift
    # in the patch's format string).
    run.expect_eq(
        "marker prefix constant",
        fwd._MARKER_PREFIX,
        "model_emit_warning kind=",
    )


# --------------------------------------------------------------------
# Section 2: 5 valid lines -> 5 records, correct fields
# --------------------------------------------------------------------


def section_2_five_valid() -> None:
    run.section("2. Five valid lines -> five records")
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        f = make_forwarder(tmpdir)
        cases = [(100, 1), (487, 2), (1024, 5), (2048, 1), (50, 3)]
        for rl, mc in cases:
            f.handle_line(make_warning_line(rl, mc))
        f._close_output()
        records = read_jsonl(f.output_path)

        run.expect_eq("five records emitted", len(records), 5)
        run.expect_eq(
            "records preserve reasoning_len ordering",
            [r["reasoning_len"] for r in records],
            [rl for rl, _ in cases],
        )
        run.expect_eq(
            "records preserve marker_count ordering",
            [r["marker_count"] for r in records],
            [mc for _, mc in cases],
        )
        for rec in records:
            run.expect_eq(
                f"kind=tool_call_in_reasoning ({rec['reasoning_len']})",
                rec["kind"],
                "tool_call_in_reasoning",
            )
            run.expect(
                f"ts is ISO-8601 with timezone ({rec['reasoning_len']})",
                bool(re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", rec["ts"]))
                and ("+" in rec["ts"] or rec["ts"].endswith("Z")),
            )
            run.expect(
                f"raw field present ({rec['reasoning_len']})",
                "model_emit_warning kind=tool_call_in_reasoning" in rec["raw"],
            )

        # State file reflects exactly five events, no parse errors.
        state = json.loads(f.state.path.read_text())
        run.expect_eq("state.events_total == 5", state["events_total"], 5)
        run.expect_eq("state.parse_errors == 0", state["parse_errors"], 0)


# --------------------------------------------------------------------
# Section 3: Malformed line — parse error, no record
# --------------------------------------------------------------------


def section_3_malformed() -> None:
    run.section("3. Malformed line -> parse_error, no record")
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        f = make_forwarder(tmpdir)

        # Capture the forwarder's stderr-bound logger output so we can
        # assert the parse_error went somewhere visible.
        buf = io.StringIO()
        import logging

        handler = logging.StreamHandler(buf)
        handler.setLevel(logging.ERROR)
        f.log.addHandler(handler)
        f.log.setLevel(logging.ERROR)

        # Marker prefix present, marker_count absent -> regex must miss.
        malformed = (
            "WARNING 04-28 13:14:15 [foo:1] "
            "model_emit_warning kind=tool_call_in_reasoning "
            "reasoning_len=42"  # no marker_count
        )
        f.handle_line(malformed)

        # And another with the marker but a non-numeric reasoning_len.
        also_bad = (
            "WARNING 04-28 13:14:15 [foo:1] "
            "model_emit_warning kind=tool_call_in_reasoning "
            "reasoning_len=NaN marker_count=2"
        )
        f.handle_line(also_bad)

        f._close_output()

        records = read_jsonl(f.output_path)
        run.expect_eq("zero records emitted on malformed lines", len(records), 0)
        state = json.loads(f.state.path.read_text())
        run.expect_eq("state.parse_errors == 2", state["parse_errors"], 2)
        run.expect_eq("state.events_total == 0", state["events_total"], 0)
        log_text = buf.getvalue()
        run.expect(
            "parse_error logged to forwarder stderr logger",
            "parse_error" in log_text,
        )


# --------------------------------------------------------------------
# Section 4: Unrelated docker line — ignored cleanly
# --------------------------------------------------------------------


def section_4_unrelated() -> None:
    run.section("4. Unrelated docker line -> ignored, no error")
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        f = make_forwarder(tmpdir)
        # Real-world neighbor lines from vLLM startup.
        for line in [
            "INFO 04-28 13:14:15 [api_server.py:1245] Started server",
            "WARNING 04-28 13:14:16 [scheduler.py:88] something else",
            "",  # blank line
            "(EngineCore_DP0) some unrelated multiline message",
        ]:
            f.handle_line(line)
        f._close_output()
        run.expect_eq(
            "zero records from unrelated lines",
            len(read_jsonl(f.output_path)),
            0,
        )
        state = json.loads(f.state.path.read_text())
        run.expect_eq(
            "state.parse_errors stays 0 on unrelated lines",
            state["parse_errors"],
            0,
        )


# --------------------------------------------------------------------
# Section 5: SIGHUP rotation — close + reopen produces a fresh handle
# --------------------------------------------------------------------


def section_5_sighup_rotation() -> None:
    run.section("5. SIGHUP rotation: write 2, rotate, write 2")
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        f = make_forwarder(tmpdir)

        # Two records pre-rotation.
        f.handle_line(make_warning_line(100, 1))
        f.handle_line(make_warning_line(200, 2))

        # The "rotate" sequence external tools follow: rename the live
        # file out of the way, then signal the forwarder to reopen.
        rotated = f.output_path.with_suffix(".jsonl.1")
        os.rename(f.output_path, rotated)

        # Trigger the same code path SIGHUP would; we don't deliver a
        # real signal here because that would require running this in
        # the run-loop (the run-loop checks ``self._rotate_requested``).
        f._rotate_output()

        # Two records post-rotation -> go to a fresh file at the
        # original path.
        f.handle_line(make_warning_line(300, 3))
        f.handle_line(make_warning_line(400, 4))
        f._close_output()

        pre = read_jsonl(rotated)
        post = read_jsonl(f.output_path)
        run.expect_eq("2 records in rotated file", len(pre), 2)
        run.expect_eq("2 records in fresh post-rotation file", len(post), 2)
        run.expect_eq(
            "pre-rotation reasoning_len values",
            [r["reasoning_len"] for r in pre],
            [100, 200],
        )
        run.expect_eq(
            "post-rotation reasoning_len values",
            [r["reasoning_len"] for r in post],
            [300, 400],
        )


# --------------------------------------------------------------------
# Section 6: State file atomic write + load
# --------------------------------------------------------------------


def section_6_state_durability() -> None:
    run.section("6. State file: atomic write + reload across restarts")
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        f1 = make_forwarder(tmpdir)
        f1.handle_line(make_warning_line(100, 1))
        f1.handle_line(make_warning_line(200, 2))
        f1._close_output()
        # Simulate restart: a second Forwarder loads the state.
        f2 = make_forwarder(tmpdir)
        run.expect_eq(
            "events_total survives restart",
            f2.state.events_total,
            2,
        )
        run.expect(
            "last_event_ts survives restart",
            f2.state.last_event_ts is not None,
        )
        f2._close_output()

        # Tempfile is gone after atomic rename — no .tmp leftover.
        leftovers = list(tmpdir.glob("forwarder_state.json.tmp"))
        run.expect_eq(
            "no leftover .tmp file after atomic rename",
            len(leftovers),
            0,
        )


# --------------------------------------------------------------------
# Section 7: argparse defaults + --no-stdout flag
# --------------------------------------------------------------------


def section_7_cli_argparse() -> None:
    run.section("7. CLI argparse: defaults + flags parse correctly")
    parser = fwd._build_argparser()
    args = parser.parse_args([])
    run.expect_eq("default container", args.container, "qwen36")
    run.expect_eq(
        "default output", args.output, "/var/log/qwen36/warnings.jsonl"
    )
    run.expect_eq(
        "default state", args.state, "/var/lib/qwen36/forwarder_state.json"
    )
    run.expect_eq("default emit_stdout (no --no-stdout)", args.no_stdout, False)
    args2 = parser.parse_args(
        ["--no-stdout", "--container", "qwen36-test", "--log-level", "DEBUG"]
    )
    run.expect_eq("--no-stdout sets True", args2.no_stdout, True)
    run.expect_eq("--container override", args2.container, "qwen36-test")
    run.expect_eq("--log-level override", args2.log_level, "DEBUG")


# --------------------------------------------------------------------
# Section 8: --help renders without crashing (smoke)
# --------------------------------------------------------------------


def section_8_help_smoke() -> None:
    run.section("8. --help smoke")
    proc = subprocess.run(
        [sys.executable, str(FORWARDER), "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    run.expect_eq("--help exit 0", proc.returncode, 0)
    run.expect("--help mentions --container", "--container" in proc.stdout)
    run.expect("--help mentions --output", "--output" in proc.stdout)
    run.expect("--help mentions --no-stdout", "--no-stdout" in proc.stdout)


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------


def main() -> int:
    print(f"{_BOLD}Verifying qwen36_warning_forwarder.py{_RESET}")
    print(f"  forwarder: {FORWARDER}")

    sections = [
        section_1_regex,
        section_2_five_valid,
        section_3_malformed,
        section_4_unrelated,
        section_5_sighup_rotation,
        section_6_state_durability,
        section_7_cli_argparse,
        section_8_help_smoke,
    ]
    for fn in sections:
        try:
            fn()
        except Exception as e:
            run.expect(
                f"{fn.__name__} raised {type(e).__name__}",
                False,
                f"{type(e).__name__}: {e}",
            )

    return run.report()


if __name__ == "__main__":
    sys.exit(main())
