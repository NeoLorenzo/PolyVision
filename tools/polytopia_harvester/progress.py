from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta
from statistics import fmean
from typing import Iterable, Optional


ETA_WINDOW_SIZE = 10


def format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    days, remainder = divmod(total, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    if days:
        return f"{days}d {hours:02d}h {minutes:02d}m"
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def average_seconds(values: Iterable[float]) -> Optional[float]:
    valid = [float(value) for value in values if float(value) > 0]
    return fmean(valid) if valid else None


def estimate_remaining_seconds(
    remaining_maps: int, cycle_times: Iterable[float]
) -> Optional[float]:
    average = average_seconds(cycle_times)
    if average is None:
        return None
    return max(0, int(remaining_maps)) * average


class ProgressTracker:
    def __init__(self, target_count: int, historical_times: Iterable[float] = ()) -> None:
        self.target_count = int(target_count)
        self.started_at = datetime.now().astimezone()
        self.recent = deque(maxlen=ETA_WINDOW_SIZE)
        for value in list(historical_times)[-ETA_WINDOW_SIZE:]:
            if float(value) > 0:
                self.recent.append(float(value))

    @property
    def average(self) -> Optional[float]:
        return average_seconds(self.recent)

    def record(self, cycle_seconds: float) -> None:
        self.recent.append(float(cycle_seconds))

    def progress_line(self, current_count: int, filename: str, cycle_seconds: float) -> str:
        self.record(cycle_seconds)
        remaining = max(0, self.target_count - int(current_count))
        average = self.average or 0.0
        remaining_seconds = estimate_remaining_seconds(remaining, self.recent) or 0.0
        eta = datetime.now().astimezone() + timedelta(seconds=remaining_seconds)
        percent = 100.0 * int(current_count) / max(1, self.target_count)
        return (
            f"[{current_count:,}/{self.target_count:,} - {percent:.2f}%] {filename} OK | "
            f"{cycle_seconds:.2f}s | avg {average:.2f}s | "
            f"remaining ~{format_duration(remaining_seconds)} | ETA {eta.strftime('%Y-%m-%d %H:%M %Z')}"
        )
