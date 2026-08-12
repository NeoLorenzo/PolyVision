from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional


class SaveTimeoutError(TimeoutError):
    pass


class FileStabilityError(TimeoutError):
    pass


@dataclass(frozen=True)
class FileState:
    size: int
    mtime_ns: int
    inode: int


class SaveWatcher:
    def __init__(
        self,
        save_dir: Path,
        poll_interval: float = 0.1,
        stability_seconds: float = 0.75,
        sleep_fn: Callable[[float], None] = time.sleep,
        monotonic_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self.save_dir = save_dir
        self.poll_interval = float(poll_interval)
        self.stability_seconds = float(stability_seconds)
        self.sleep = sleep_fn
        self.monotonic = monotonic_fn

    @staticmethod
    def _state(path: Path) -> FileState:
        stat = path.stat()
        return FileState(
            size=int(stat.st_size),
            mtime_ns=int(stat.st_mtime_ns),
            inode=int(getattr(stat, "st_ino", 0)),
        )

    def snapshot(self) -> Dict[str, FileState]:
        snapshot = {}
        for path in self.save_dir.glob("*.state"):
            if not path.is_file():
                continue
            try:
                snapshot[path.name] = self._state(path)
            except OSError:
                continue
        return snapshot

    def changed_saves(self, baseline: Dict[str, FileState]) -> list[Path]:
        candidates: list[tuple[int, Path]] = []
        for path in self.save_dir.glob("*.state"):
            if not path.is_file():
                continue
            try:
                current = self._state(path)
            except OSError:
                continue
            previous = baseline.get(path.name)
            if previous is None or current != previous:
                candidates.append((current.mtime_ns, path))
        return [path for _, path in sorted(candidates, key=lambda item: item[0], reverse=True)]

    def wait_for_changed_save(
        self, baseline: Dict[str, FileState], timeout_seconds: float
    ) -> Path:
        deadline = self.monotonic() + float(timeout_seconds)
        while self.monotonic() < deadline:
            changed = self.changed_saves(baseline)
            if changed:
                return changed[0]
            self.sleep(self.poll_interval)
        raise SaveTimeoutError(
            f"no new or modified .state file appeared within {timeout_seconds:.1f} seconds"
        )

    def wait_until_stable(
        self, path: Path, timeout_seconds: Optional[float] = None
    ) -> FileState:
        timeout = float(timeout_seconds or max(10.0, self.stability_seconds * 10.0))
        deadline = self.monotonic() + timeout
        stable_since: Optional[float] = None
        previous: Optional[FileState] = None

        while self.monotonic() < deadline:
            try:
                current = self._state(path)
                with path.open("rb") as handle:
                    handle.read(1)
            except OSError:
                stable_since = None
                previous = None
                self.sleep(self.poll_interval)
                continue

            now = self.monotonic()
            if current.size > 0 and current == previous:
                if stable_since is None:
                    stable_since = now
                if now - stable_since >= self.stability_seconds:
                    return current
            else:
                stable_since = None
            previous = current
            self.sleep(self.poll_interval)

        raise FileStabilityError(f"save did not become stable before timeout: {path}")
