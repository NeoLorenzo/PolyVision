from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple


BUTTON_SEQUENCE: Tuple[Tuple[str, str], ...] = (
    ("new_game", "New Game"),
    ("creative", "Creative"),
    ("bardur", "Bardur"),
    ("pick", "Pick"),
    ("start_game", "Start Game"),
    ("ok", "OK"),
    ("settings", "Settings"),
    ("exit_to_menu", "Exit To Menu"),
)


class ConfigError(RuntimeError):
    pass


@dataclass
class TimingConfig:
    poll_interval: float = 0.1
    file_stability_seconds: float = 0.75
    save_timeout_seconds: float = 20.0
    max_duplicate_retries: int = 3

    def validate(self) -> None:
        positive = {
            "poll_interval": self.poll_interval,
            "file_stability_seconds": self.file_stability_seconds,
            "save_timeout_seconds": self.save_timeout_seconds,
        }
        for name, value in positive.items():
            if float(value) <= 0:
                raise ConfigError(f"timing.{name} must be > 0")
        if int(self.max_duplicate_retries) < 0:
            raise ConfigError("timing.max_duplicate_retries must be >= 0")


def default_click_delays() -> Dict[str, float]:
    """Seconds to wait after each named click before the next operation."""
    return {
        "new_game": 0.005,
        "creative": 0.005,
        # Bardur needs extra time to reveal/enable Pick.
        "bardur": 0.25,
        "pick": 0.005,
        # Start Game needs time for the OK dialog to appear.
        "start_game": 3.5,
        "ok": 0.005,
        "settings": 0.005,
        "exit_to_menu": 0.005,
    }


@dataclass
class HarvesterConfig:
    buttons: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    screen_width: int = 0
    screen_height: int = 0
    save_dir: str = ""
    output_dir: str = ""
    foreground_title_contains: str = "Polytopia"
    click_delays: Dict[str, float] = field(default_factory=default_click_delays)
    timing: TimingConfig = field(default_factory=TimingConfig)

    def validate(self, require_calibration: bool = True) -> None:
        self.timing.validate()
        missing_delays = [key for key, _ in BUTTON_SEQUENCE if key not in self.click_delays]
        if missing_delays:
            raise ConfigError(f"click_delays is missing buttons: {', '.join(missing_delays)}")
        for key, value in self.click_delays.items():
            if key in {name for name, _ in BUTTON_SEQUENCE} and float(value) < 0:
                raise ConfigError(f"click_delays.{key} must be >= 0")
        if require_calibration:
            missing = [key for key, _ in BUTTON_SEQUENCE if key not in self.buttons]
            if missing:
                raise ConfigError(f"calibration is missing buttons: {', '.join(missing)}")
            if self.screen_width <= 0 or self.screen_height <= 0:
                raise ConfigError("calibrated screen resolution is missing or invalid")
        for key, coordinate in self.buttons.items():
            if key not in {name for name, _ in BUTTON_SEQUENCE}:
                continue
            if not isinstance(coordinate, (tuple, list)) or len(coordinate) != 2:
                raise ConfigError(f"button {key!r} must contain [x, y]")
            x, y = int(coordinate[0]), int(coordinate[1])
            if x < 0 or y < 0:
                raise ConfigError(f"button {key!r} has a negative coordinate")
            if self.screen_width > 0 and x >= self.screen_width:
                raise ConfigError(f"button {key!r} x={x} is outside the calibrated screen")
            if self.screen_height > 0 and y >= self.screen_height:
                raise ConfigError(f"button {key!r} y={y} is outside the calibrated screen")

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["screen"] = {
            "width": payload.pop("screen_width"),
            "height": payload.pop("screen_height"),
        }
        payload["buttons"] = {
            key: [int(value[0]), int(value[1])] for key, value in self.buttons.items()
        }
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "HarvesterConfig":
        screen = payload.get("screen", {}) or {}
        timing_raw = payload.get("timing", {}) or {}
        timing_defaults = asdict(TimingConfig())
        timing_defaults.update(
            {key: value for key, value in timing_raw.items() if key in timing_defaults}
        )
        click_delays = default_click_delays()
        click_delays.update(
            {
                str(key): float(value)
                for key, value in (payload.get("click_delays", {}) or {}).items()
            }
        )
        buttons = {
            str(key): (int(value[0]), int(value[1]))
            for key, value in (payload.get("buttons", {}) or {}).items()
        }
        return cls(
            buttons=buttons,
            screen_width=int(screen.get("width", 0)),
            screen_height=int(screen.get("height", 0)),
            save_dir=str(payload.get("save_dir", "") or ""),
            output_dir=str(payload.get("output_dir", "") or ""),
            foreground_title_contains=str(
                payload.get("foreground_title_contains", "Polytopia") or ""
            ),
            click_delays=click_delays,
            timing=TimingConfig(**timing_defaults),
        )


def default_config_path() -> Path:
    return Path(__file__).resolve().parent / "harvester_config.json"


def default_output_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "polytopia_maps"


def load_config(path: Path) -> HarvesterConfig:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"configuration does not exist: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"could not read configuration {path}: {exc}") from exc
    try:
        config = HarvesterConfig.from_dict(payload)
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        raise ConfigError(f"configuration has invalid values: {path}: {exc}") from exc
    config.validate(require_calibration=True)
    return config


def save_config(config: HarvesterConfig, path: Path) -> None:
    config.validate(require_calibration=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_screen_resolution(
    config: HarvesterConfig, current: Tuple[int, int]
) -> None:
    expected = (int(config.screen_width), int(config.screen_height))
    actual = (int(current[0]), int(current[1]))
    if expected != actual:
        raise ConfigError(
            "screen resolution differs from calibration: "
            f"calibrated={expected[0]}x{expected[1]}, current={actual[0]}x{actual[1]}. "
            "Run again with --calibrate."
        )


def discover_save_directories(user_profile: Optional[Path] = None) -> list[Path]:
    profile = user_profile or Path(os.environ.get("USERPROFILE", str(Path.home())))
    base = profile / "AppData" / "LocalLow" / "Midjiwan" / "Polytopia"
    if not base.is_dir():
        return []
    candidates = {
        path.resolve()
        for path in base.rglob("Singleplayer")
        if path.is_dir()
    }
    return sorted(candidates, key=lambda item: str(item).lower())


def validate_save_directory(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise ConfigError(f"Polytopia save directory does not exist: {resolved}")
    try:
        list(resolved.iterdir())
    except OSError as exc:
        raise ConfigError(f"Polytopia save directory is not readable: {resolved}") from exc
    if not os.access(resolved, os.R_OK | os.W_OK):
        raise ConfigError(f"Polytopia save directory is not readable and writable: {resolved}")
    return resolved


def choose_save_directory(
    explicit: Optional[Path], stored: str, candidates: Optional[Iterable[Path]] = None
) -> Path:
    if explicit is not None:
        return validate_save_directory(explicit)
    if stored:
        try:
            return validate_save_directory(Path(stored))
        except ConfigError:
            print(f"Stored save directory is no longer valid: {stored}")

    found = list(candidates if candidates is not None else discover_save_directories())
    if len(found) == 1:
        print(f"Discovered Polytopia save directory: {found[0]}")
        return validate_save_directory(found[0])
    if len(found) > 1:
        print("Multiple Polytopia Singleplayer directories were found:")
        for index, path in enumerate(found, start=1):
            print(f"  {index}. {path}")
        while True:
            raw = input("Select a directory number: ").strip()
            try:
                selected = int(raw)
                if 1 <= selected <= len(found):
                    return validate_save_directory(found[selected - 1])
            except ValueError:
                pass
            print("Enter one of the numbers shown above.")

    while True:
        raw = input("Enter the full Polytopia Singleplayer save directory: ").strip().strip('"')
        try:
            return validate_save_directory(Path(raw))
        except ConfigError as exc:
            print(exc)
