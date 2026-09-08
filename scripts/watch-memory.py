#!/usr/bin/env python3
"""Guard one exact candidate container against GB10 host-memory exhaustion.

This is an operator-started foreground process.  It never discovers, restarts,
kills, or sends notifications about containers.  On a sustained threshold it
issues one ``docker stop --timeout 30`` for the exact 64-hex ID supplied by the
caller, then verifies and records the resulting container state.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import signal
import stat
import subprocess
import threading
import time
from typing import Callable, Mapping, Sequence


SCHEMA_VERSION = 1
CONTAINER_ID_PATTERN = re.compile(r"[0-9a-f]{64}")
SAMPLE_INTERVAL_SECONDS = 1.0
REQUIRED_CONSECUTIVE_SAMPLES = 5
MEM_AVAILABLE_FLOOR_KIB = 6 * 1024 * 1024
MEM_FREE_FLOOR_KIB = 2 * 1024 * 1024
MEM_FREE_AVAILABLE_GATE_KIB = 10 * 1024 * 1024
DOCKER_INSPECT_TIMEOUT_SECONDS = 10
DOCKER_STOP_TIMEOUT_SECONDS = 30
DOCKER_STOP_COMMAND_TIMEOUT_SECONDS = 40
MAX_CONSECUTIVE_INSPECT_FAILURES = 3
MAX_CONSECUTIVE_MEMORY_FAILURES = 3
STOP_VERIFY_ATTEMPTS = 5

EXIT_OK = 0
EXIT_STOP_FAILED = 3
EXIT_INSPECT_FAILED = 4
EXIT_MEMORY_FAILED = 5
EXIT_GRACE_NOT_VERIFIED = 6


@dataclass(frozen=True, slots=True)
class MemorySample:
    mem_available_kib: int
    mem_free_kib: int
    swap_free_kib: int


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    sample_number: int
    low_available_streak: int
    low_free_gated_streak: int
    triggers: tuple[str, ...]


class MemoryPolicy:
    """Pure consecutive-sample policy; time and I/O stay outside this class."""

    def __init__(self) -> None:
        self.sample_number = 0
        self.low_available_streak = 0
        self.low_free_gated_streak = 0

    def observe(self, sample: MemorySample) -> PolicyDecision:
        self.sample_number += 1
        if sample.mem_available_kib < MEM_AVAILABLE_FLOOR_KIB:
            self.low_available_streak += 1
        else:
            self.low_available_streak = 0

        if (
            sample.mem_free_kib < MEM_FREE_FLOOR_KIB
            and sample.mem_available_kib < MEM_FREE_AVAILABLE_GATE_KIB
        ):
            self.low_free_gated_streak += 1
        else:
            self.low_free_gated_streak = 0

        triggers: list[str] = []
        if self.low_available_streak >= REQUIRED_CONSECUTIVE_SAMPLES:
            triggers.append("mem_available_below_6_gib")
        if self.low_free_gated_streak >= REQUIRED_CONSECUTIVE_SAMPLES:
            triggers.append("mem_free_below_2_gib_while_available_below_10_gib")
        return PolicyDecision(
            sample_number=self.sample_number,
            low_available_streak=self.low_available_streak,
            low_free_gated_streak=self.low_free_gated_streak,
            triggers=tuple(triggers),
        )

    def break_sequence(self) -> None:
        """A missing one-second observation cannot extend a consecutive run."""

        self.low_available_streak = 0
        self.low_free_gated_streak = 0


@dataclass(frozen=True, slots=True)
class ContainerState:
    container_id: str
    name: str
    image_id: str
    running: bool
    status: str
    started_at: str
    exit_code: int
    oom_killed: bool
    paused: bool
    restarting: bool
    dead: bool
    state_error_present: bool
    finished_at: str
    restart_policy: str


@dataclass(frozen=True, slots=True)
class CommandOutcome:
    returncode: int | None
    timed_out: bool


class DockerCommandError(RuntimeError):
    def __init__(
        self,
        operation: str,
        *,
        returncode: int | None = None,
        timed_out: bool = False,
    ) -> None:
        super().__init__(f"docker {operation} failed")
        self.operation = operation
        self.returncode = returncode
        self.timed_out = timed_out

    def record(self) -> dict[str, object]:
        return {
            "operation": self.operation,
            "returncode": self.returncode,
            "timed_out": self.timed_out,
        }


def validate_container_id(value: str) -> str:
    if CONTAINER_ID_PATTERN.fullmatch(value) is None:
        raise ValueError("container ID must be exactly 64 lowercase hex characters")
    return value


def parse_meminfo(content: str) -> MemorySample:
    required = {
        "MemAvailable": "mem_available_kib",
        "MemFree": "mem_free_kib",
        "SwapFree": "swap_free_kib",
    }
    values: dict[str, int] = {}
    for line in content.splitlines():
        key, separator, remainder = line.partition(":")
        if not separator or key not in required:
            continue
        if key in values:
            raise ValueError(f"duplicate /proc/meminfo field: {key}")
        parts = remainder.split()
        if len(parts) != 2 or parts[1] != "kB" or not parts[0].isdigit():
            raise ValueError(f"invalid /proc/meminfo field: {key}")
        values[key] = int(parts[0])
    missing = sorted(set(required).difference(values))
    if missing:
        raise ValueError(f"missing /proc/meminfo fields: {missing}")
    return MemorySample(
        mem_available_kib=values["MemAvailable"],
        mem_free_kib=values["MemFree"],
        swap_free_kib=values["SwapFree"],
    )


def read_memory_sample(meminfo_path: Path = Path("/proc/meminfo")) -> MemorySample:
    return parse_meminfo(meminfo_path.read_text(encoding="utf-8", errors="strict"))


def parse_container_inspect(payload: bytes, expected_id: str) -> ContainerState:
    try:
        decoded = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DockerCommandError("inspect-parse") from error
    if not isinstance(decoded, list) or len(decoded) != 1:
        raise DockerCommandError("inspect-cardinality")
    record = decoded[0]
    if not isinstance(record, dict) or record.get("Id") != expected_id:
        raise DockerCommandError("inspect-identity")
    state = record.get("State")
    host_config = record.get("HostConfig")
    if not isinstance(state, dict) or not isinstance(host_config, dict):
        raise DockerCommandError("inspect-schema")
    restart = host_config.get("RestartPolicy")
    if not isinstance(restart, dict):
        raise DockerCommandError("inspect-restart-policy")
    required_types = {
        "Running": bool,
        "Paused": bool,
        "Restarting": bool,
        "Dead": bool,
        "Status": str,
        "StartedAt": str,
        "ExitCode": int,
        "OOMKilled": bool,
        "Error": str,
        "FinishedAt": str,
    }
    for field, expected_type in required_types.items():
        value = state.get(field)
        if expected_type is int and isinstance(value, bool):
            raise DockerCommandError("inspect-state-schema")
        if not isinstance(value, expected_type):
            raise DockerCommandError("inspect-state-schema")
    name = record.get("Name")
    image_id = record.get("Image")
    restart_name = restart.get("Name", "")
    if not isinstance(name, str) or not isinstance(image_id, str):
        raise DockerCommandError("inspect-schema")
    if not isinstance(restart_name, str):
        raise DockerCommandError("inspect-restart-policy")
    if state["Running"] and not state["StartedAt"]:
        raise DockerCommandError("inspect-lifecycle")
    return ContainerState(
        container_id=expected_id,
        name=name.removeprefix("/"),
        image_id=image_id,
        running=state["Running"],
        status=state["Status"],
        started_at=state["StartedAt"],
        exit_code=state["ExitCode"],
        oom_killed=state["OOMKilled"],
        paused=state["Paused"],
        restarting=state["Restarting"],
        dead=state["Dead"],
        state_error_present=bool(state["Error"]),
        finished_at=state["FinishedAt"],
        restart_policy=restart_name,
    )


class DockerClient:
    def __init__(self, container_id: str) -> None:
        self.container_id = validate_container_id(container_id)

    def inspect(self) -> ContainerState:
        try:
            result = subprocess.run(
                ["docker", "inspect", "--type", "container", self.container_id],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=DOCKER_INSPECT_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as error:
            raise DockerCommandError("inspect", timed_out=True) from error
        except OSError as error:
            raise DockerCommandError("inspect-exec") from error
        if result.returncode:
            raise DockerCommandError("inspect", returncode=result.returncode)
        return parse_container_inspect(result.stdout, self.container_id)

    def stop(self) -> CommandOutcome:
        try:
            result = subprocess.run(
                [
                    "docker",
                    "stop",
                    "--timeout",
                    str(DOCKER_STOP_TIMEOUT_SECONDS),
                    self.container_id,
                ],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=DOCKER_STOP_COMMAND_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return CommandOutcome(returncode=None, timed_out=True)
        except OSError:
            return CommandOutcome(returncode=None, timed_out=False)
        return CommandOutcome(returncode=result.returncode, timed_out=False)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class EvidenceWriter:
    def __init__(self, artifact_dir: Path) -> None:
        requested = artifact_dir.expanduser()
        parent = requested.parent.resolve()
        if not parent.is_dir():
            raise ValueError("artifact directory parent must already exist")
        if requested.exists() or requested.is_symlink():
            raise ValueError("artifact directory must be a fresh, nonexistent path")
        requested.mkdir(mode=0o700, exist_ok=False)
        self.artifact_dir = requested.resolve()
        self.progress_path = self.artifact_dir / "progress.json"
        self.events_path = self.artifact_dir / "progress.jsonl"
        self.result_path = self.artifact_dir / "result.json"
        self._sequence = 0
        self._progress_created = False
        self._events = self.events_path.open("x", encoding="utf-8")

    def _atomic_latest(self, payload: bytes) -> None:
        temporary = self.artifact_dir / f".progress.{os.getpid()}.{self._sequence}.tmp"
        try:
            with temporary.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            if self._progress_created:
                os.replace(temporary, self.progress_path)
            else:
                os.link(temporary, self.progress_path)
                self._progress_created = True
        finally:
            temporary.unlink(missing_ok=True)

    def event(self, kind: str, **fields: object) -> dict[str, object]:
        self._sequence += 1
        record = {
            "schema_version": SCHEMA_VERSION,
            "sequence": self._sequence,
            "recorded_at": utc_now(),
            "kind": kind,
            **fields,
        }
        line = json.dumps(
            record,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        self._events.write(line + "\n")
        self._events.flush()
        os.fsync(self._events.fileno())
        self._atomic_latest(
            (
                json.dumps(record, allow_nan=False, indent=2, sort_keys=True)
                + "\n"
            ).encode()
        )
        return record

    def finish(self, **fields: object) -> None:
        result = {
            "schema_version": SCHEMA_VERSION,
            "recorded_at": utc_now(),
            **fields,
        }
        encoded = (
            json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n"
        ).encode()
        temporary = self.artifact_dir / f".result.{os.getpid()}.tmp"
        try:
            with temporary.open("xb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temporary, self.result_path)
        finally:
            temporary.unlink(missing_ok=True)

    def close(self) -> None:
        self._events.close()


class SignalLatch:
    def __init__(self) -> None:
        self.event = threading.Event()
        self.signal_number: int | None = None

    def handle(self, signum: int, _frame: object) -> None:
        if self.signal_number is None:
            self.signal_number = signum
        self.event.set()


def state_record(state: ContainerState) -> dict[str, object]:
    return asdict(state)


def _verify_after_stop(
    client: DockerClient,
    writer: EvidenceWriter,
    wait: Callable[[float], bool],
    expected_started_at: str,
) -> tuple[ContainerState | None, list[dict[str, object]]]:
    errors: list[dict[str, object]] = []
    for attempt in range(1, STOP_VERIFY_ATTEMPTS + 1):
        try:
            state = client.inspect()
        except DockerCommandError as error:
            errors.append(error.record())
            writer.event("stop_verify_inspect_error", attempt=attempt, error=error.record())
        else:
            writer.event(
                "stop_verify",
                attempt=attempt,
                container=state_record(state),
            )
            if state.started_at != expected_started_at:
                writer.event(
                    "stop_verify_lifecycle_mismatch",
                    expected_started_at=expected_started_at,
                    container=state_record(state),
                )
                return None, errors
            if not state.running:
                return state, errors
        if attempt < STOP_VERIFY_ATTEMPTS:
            wait(SAMPLE_INTERVAL_SECONDS)
    return None, errors


def _finish_existing_exit(
    writer: EvidenceWriter,
    state: ContainerState,
    *,
    initial: bool,
) -> int:
    outcome = "container_already_exited" if initial else "container_exited"
    writer.event(outcome, container=state_record(state))
    writer.finish(
        outcome=outcome,
        exit_code=EXIT_OK,
        stop_attempted=False,
        container=state_record(state),
    )
    return EXIT_OK


def _finish_interrupted(
    writer: EvidenceWriter,
    container_id: str,
    signal_latch: SignalLatch,
) -> int:
    writer.event(
        "guard_interrupted",
        container_id=container_id,
        signal=signal_latch.signal_number,
    )
    writer.finish(
        outcome="guard_interrupted",
        exit_code=EXIT_OK,
        stop_attempted=False,
        signal=signal_latch.signal_number,
    )
    return EXIT_OK


def _is_active_generation(state: ContainerState, started_at: str) -> bool:
    return (
        state.running
        and not state.paused
        and not state.restarting
        and not state.dead
        and state.started_at == started_at
    )


def run_guard(
    container_id: str,
    artifact_dir: Path,
    *,
    memory_reader: Callable[[], MemorySample] = read_memory_sample,
    monotonic: Callable[[], float] = time.monotonic,
    latch: SignalLatch | None = None,
) -> int:
    exact_id = validate_container_id(container_id)
    writer = EvidenceWriter(artifact_dir)
    client = DockerClient(exact_id)
    policy = MemoryPolicy()
    signal_latch = latch or SignalLatch()
    started = monotonic()

    def wait(seconds: float) -> bool:
        return signal_latch.event.wait(seconds)

    try:
        writer.event(
            "guard_started",
            container_id=exact_id,
            thresholds_kib={
                "mem_available_floor": MEM_AVAILABLE_FLOOR_KIB,
                "mem_free_floor": MEM_FREE_FLOOR_KIB,
                "mem_free_available_gate": MEM_FREE_AVAILABLE_GATE_KIB,
                "required_consecutive_samples": REQUIRED_CONSECUTIVE_SAMPLES,
                "sample_interval_seconds": SAMPLE_INTERVAL_SECONDS,
            },
        )
        try:
            initial_state = client.inspect()
        except DockerCommandError as error:
            writer.event("initial_inspect_failed", error=error.record())
            writer.finish(
                outcome="initial_inspect_failed",
                exit_code=EXIT_INSPECT_FAILED,
                stop_attempted=False,
                error=error.record(),
            )
            return EXIT_INSPECT_FAILED
        if initial_state.restart_policy not in {"", "no"}:
            writer.event(
                "restart_policy_rejected",
                container=state_record(initial_state),
            )
            writer.finish(
                outcome="restart_policy_rejected",
                exit_code=EXIT_INSPECT_FAILED,
                stop_attempted=False,
                container=state_record(initial_state),
            )
            return EXIT_INSPECT_FAILED
        writer.event("container_bound", container=state_record(initial_state))
        if not initial_state.running:
            return _finish_existing_exit(writer, initial_state, initial=True)
        if initial_state.paused or initial_state.restarting or initial_state.dead:
            writer.event("initial_container_not_active", container=state_record(initial_state))
            writer.finish(
                outcome="initial_container_not_active",
                exit_code=EXIT_INSPECT_FAILED,
                stop_attempted=False,
                container=state_record(initial_state),
            )
            return EXIT_INSPECT_FAILED
        bound_started_at = initial_state.started_at

        inspect_failures = 0
        memory_failures = 0
        next_sample_at = monotonic()
        while True:
            if signal_latch.event.is_set():
                return _finish_interrupted(writer, exact_id, signal_latch)

            delay = next_sample_at - monotonic()
            if delay > 0 and wait(delay):
                continue
            next_sample_at = monotonic() + SAMPLE_INTERVAL_SECONDS

            try:
                state = client.inspect()
            except DockerCommandError as error:
                inspect_failures += 1
                policy.break_sequence()
                writer.event(
                    "inspect_error",
                    container_id=exact_id,
                    consecutive_failures=inspect_failures,
                    error=error.record(),
                )
                if inspect_failures >= MAX_CONSECUTIVE_INSPECT_FAILURES:
                    writer.finish(
                        outcome="inspect_failure_limit",
                        exit_code=EXIT_INSPECT_FAILED,
                        stop_attempted=False,
                        consecutive_failures=inspect_failures,
                    )
                    return EXIT_INSPECT_FAILED
                continue
            inspect_failures = 0
            if not state.running:
                return _finish_existing_exit(writer, state, initial=False)
            if state.started_at != bound_started_at:
                policy.break_sequence()
                writer.event(
                    "container_lifecycle_changed",
                    original_started_at=bound_started_at,
                    container=state_record(state),
                )
                writer.finish(
                    outcome="container_lifecycle_changed",
                    exit_code=EXIT_OK,
                    stop_attempted=False,
                    container=state_record(state),
                )
                return EXIT_OK
            if state.paused or state.restarting or state.dead:
                policy.break_sequence()
                writer.event("container_not_active", container=state_record(state))
                writer.finish(
                    outcome="container_not_active",
                    exit_code=EXIT_INSPECT_FAILED,
                    stop_attempted=False,
                    container=state_record(state),
                )
                return EXIT_INSPECT_FAILED

            try:
                sample = memory_reader()
            except (OSError, UnicodeError, ValueError) as error:
                memory_failures += 1
                policy.break_sequence()
                writer.event(
                    "memory_sample_error",
                    container_id=exact_id,
                    consecutive_failures=memory_failures,
                    error_type=type(error).__name__,
                )
                if memory_failures >= MAX_CONSECUTIVE_MEMORY_FAILURES:
                    writer.finish(
                        outcome="memory_failure_limit",
                        exit_code=EXIT_MEMORY_FAILED,
                        stop_attempted=False,
                        consecutive_failures=memory_failures,
                    )
                    return EXIT_MEMORY_FAILED
                continue
            memory_failures = 0
            decision = policy.observe(sample)
            writer.event(
                "memory_sample",
                elapsed_seconds=round(monotonic() - started, 6),
                container=state_record(state),
                memory_kib=asdict(sample),
                policy=asdict(decision),
            )
            if not decision.triggers:
                continue

            writer.event(
                "memory_threshold_triggered",
                container=state_record(state),
                memory_kib=asdict(sample),
                policy=asdict(decision),
            )
            try:
                pre_stop_state = client.inspect()
            except DockerCommandError as error:
                writer.event("pre_stop_inspect_failed", error=error.record())
                writer.finish(
                    outcome="pre_stop_inspect_failed",
                    exit_code=EXIT_INSPECT_FAILED,
                    stop_attempted=False,
                    triggers=list(decision.triggers),
                    error=error.record(),
                )
                return EXIT_INSPECT_FAILED
            if not pre_stop_state.running:
                return _finish_existing_exit(writer, pre_stop_state, initial=False)
            if not _is_active_generation(pre_stop_state, bound_started_at):
                policy.break_sequence()
                writer.event(
                    "pre_stop_lifecycle_mismatch",
                    original_started_at=bound_started_at,
                    container=state_record(pre_stop_state),
                )
                writer.finish(
                    outcome="pre_stop_lifecycle_mismatch",
                    exit_code=EXIT_OK,
                    stop_attempted=False,
                    container=state_record(pre_stop_state),
                )
                return EXIT_OK
            if signal_latch.event.is_set():
                return _finish_interrupted(writer, exact_id, signal_latch)

            writer.event(
                "graceful_stop_started",
                container=state_record(pre_stop_state),
                command={
                    "operation": "docker stop",
                    "timeout_seconds": DOCKER_STOP_TIMEOUT_SECONDS,
                    "container_id": exact_id,
                },
                triggers=list(decision.triggers),
            )
            if signal_latch.event.is_set():
                return _finish_interrupted(writer, exact_id, signal_latch)
            stop_outcome = client.stop()
            writer.event("graceful_stop_command_finished", outcome=asdict(stop_outcome))
            final_state, verify_errors = _verify_after_stop(
                client,
                writer,
                wait,
                bound_started_at,
            )
            if final_state is None:
                writer.finish(
                    outcome="graceful_stop_failed_or_unverified",
                    exit_code=EXIT_STOP_FAILED,
                    stop_attempted=True,
                    stop_command=asdict(stop_outcome),
                    triggers=list(decision.triggers),
                    verify_errors=verify_errors,
                )
                return EXIT_STOP_FAILED

            command_succeeded = (
                stop_outcome.returncode == 0 and not stop_outcome.timed_out
            )
            grace_verified = (
                command_succeeded
                and not final_state.oom_killed
                and not final_state.state_error_present
                and final_state.exit_code != 137
            )
            if not grace_verified:
                outcome = "container_stopped_but_grace_not_verified"
                exit_code = EXIT_GRACE_NOT_VERIFIED
            else:
                outcome = "graceful_stop_verified"
                exit_code = EXIT_OK
            writer.event(
                outcome,
                container=state_record(final_state),
                stop_command=asdict(stop_outcome),
                triggers=list(decision.triggers),
            )
            writer.finish(
                outcome=outcome,
                exit_code=exit_code,
                stop_attempted=True,
                actual_stopped=not final_state.running,
                graceful_stop_verified=grace_verified,
                container=state_record(final_state),
                stop_command=asdict(stop_outcome),
                triggers=list(decision.triggers),
            )
            return exit_code
    finally:
        writer.close()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--container-id",
        required=True,
        type=validate_container_id,
        help="exact 64-character lowercase container ID",
    )
    parser.add_argument(
        "--artifact-dir",
        required=True,
        type=Path,
        help="fresh nonexistent directory for progress.json, JSONL and result",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if platform.system() != "Linux" or platform.machine() not in {"aarch64", "arm64"}:
        raise SystemExit("memory guard must run on the target ARM64 Linux Spark")
    latch = SignalLatch()
    previous_handlers: dict[int, signal.Handlers] = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, latch.handle)
    try:
        return run_guard(args.container_id, args.artifact_dir, latch=latch)
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
