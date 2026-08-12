from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Optional

from calibration import calibrate
from config import (
    BUTTON_SEQUENCE,
    ConfigError,
    HarvesterConfig,
    choose_save_directory,
    default_config_path,
    default_output_dir,
    load_config,
    save_config,
    validate_screen_resolution,
)
from manifest import CopyIntegrityError, DatasetManager, DuplicateStateError
from progress import (
    ETA_WINDOW_SIZE,
    ProgressTracker,
    average_seconds,
    estimate_remaining_seconds,
    format_duration,
)
from save_watcher import FileStabilityError, SaveTimeoutError, SaveWatcher
from ui_controller import UIController, warn_if_polytopia_not_foreground


LOGGER = logging.getLogger("polytopia_harvester")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Harvest genuine Polytopia .state saves through fixed-coordinate UI automation."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="desired total number of unique harvested states (not an additional count)",
    )
    parser.add_argument(
        "--calibrate", action="store_true", help="capture and overwrite all button coordinates"
    )
    parser.add_argument("--save-dir", type=Path, help="Polytopia Data/Singleplayer directory")
    parser.add_argument("--output-dir", type=Path, help="dataset directory")
    parser.add_argument(
        "--config", type=Path, default=default_config_path(), help="local calibration JSON path"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print resolved configuration and sequence without clicking"
    )
    parser.add_argument("--verbose", action="store_true", help="enable verbose file logging")
    args = parser.parse_args(argv)
    if args.count is not None and args.count < 0:
        parser.error("--count must be >= 0")
    if args.count is None and not args.calibrate and not args.dry_run:
        parser.error("--count is required unless only calibrating or using --dry-run")
    return args


def import_pyautogui() -> Any:
    if os.name != "nt":
        raise RuntimeError("the Polytopia harvester is Windows-only")
    try:
        import pyautogui  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "PyAutoGUI is required. Run: "
            "python -m pip install -r tools/polytopia_harvester/requirements.txt"
        ) from exc
    pyautogui.FAILSAFE = True
    return pyautogui


def configure_logging(output_dir: Path, verbose: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "harvester.log"
    LOGGER.setLevel(logging.DEBUG if verbose else logging.INFO)
    LOGGER.handlers.clear()
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.propagate = False


def load_or_create_config(path: Path) -> tuple[HarvesterConfig, bool]:
    try:
        return load_config(path), True
    except ConfigError:
        return HarvesterConfig(output_dir=str(default_output_dir())), False


def resolve_config(
    args: argparse.Namespace, pyautogui_module: Optional[Any]
) -> HarvesterConfig:
    config, calibrated = load_or_create_config(args.config)
    if args.output_dir is not None:
        config.output_dir = str(args.output_dir.expanduser().resolve())
    elif not config.output_dir:
        config.output_dir = str(default_output_dir())
    configure_logging(Path(config.output_dir).expanduser().resolve(), args.verbose)

    if args.dry_run and args.save_dir is None and not config.save_dir:
        save_dir = Path("<not configured>")
    else:
        save_dir = choose_save_directory(args.save_dir, config.save_dir)
        config.save_dir = str(save_dir)

    if args.calibrate or (not calibrated and not args.dry_run):
        if pyautogui_module is None:
            raise RuntimeError("PyAutoGUI is required for calibration")
        if not calibrated:
            print("No valid calibration was found; entering calibration mode automatically.")
        config = calibrate(pyautogui_module, config, args.config)
        calibrated = True
    elif calibrated:
        config.validate(require_calibration=True)

    if calibrated and str(save_dir) != "<not configured>":
        config.save_dir = str(save_dir)
        save_config(config, args.config)
    return config


def print_startup_summary(
    config: HarvesterConfig,
    dataset: DatasetManager,
    target_count: int,
    current_resolution: Optional[tuple[int, int]],
) -> None:
    existing = dataset.existing_count()
    remaining = dataset.remaining_to_target(target_count)
    history = dataset.historical_cycle_times()
    recent_history = history[-ETA_WINDOW_SIZE:]
    historical_average = average_seconds(recent_history)
    estimate = estimate_remaining_seconds(remaining, recent_history)

    print("\nPOLYVISION - POLYTOPIA MAP HARVESTER\n")
    print(f"Target dataset:       {target_count:,} maps")
    print(f"Existing maps:        {existing:,}")
    print(f"Remaining:            {remaining:,}")
    if remaining and existing:
        print(f"Resuming from:        {dataset.next_index():,}")
    print("\nSave directory:")
    print(config.save_dir or "<not configured>")
    print("\nOutput directory:")
    print(dataset.output_dir)
    if current_resolution is not None:
        print(
            f"\nResolution:           {current_resolution[0]}x{current_resolution[1]} (matches calibration)"
        )
    calibrated = (
        all(key in config.buttons for key, _ in BUTTON_SEQUENCE)
        and config.screen_width > 0
        and config.screen_height > 0
    )
    print(f"Calibration:          {'loaded' if calibrated else 'not configured'}")
    if historical_average is None or estimate is None:
        print("Historical average:   unavailable")
        print("Estimated runtime:    unknown until the first few maps complete")
    else:
        print(f"Historical average:   {historical_average:.2f} s/map")
        print(f"Estimated runtime:    {format_duration(estimate)}")


def print_safety_prompt() -> None:
    print(
        "\nIMPORTANT:\n"
        "- Put Polytopia on the main menu.\n"
        "- Do not move or resize the game window.\n"
        "- Do not use the mouse while harvesting.\n"
        "- Prevent the PC from sleeping.\n"
        "- Move the mouse to the upper-left corner to emergency stop.\n"
        "- Press Ctrl+C for a graceful stop.\n"
    )
    input("Press ENTER to begin. ")


def countdown(seconds: int = 3) -> None:
    for remaining in range(seconds, 0, -1):
        print(f"Starting in {remaining}...")
        time.sleep(1.0)


def harvest_one_map(
    ui: UIController,
    watcher: SaveWatcher,
    dataset: DatasetManager,
    index: int,
    timing: Any,
) -> tuple[Optional[object], float]:
    cycle_start = time.monotonic()
    ui.click("new_game")
    ui.click("creative")
    ui.click("bardur")
    ui.click("pick")

    baseline = watcher.snapshot()
    ui.click("start_game")
    ui.click("ok")

    changed_save = watcher.wait_for_changed_save(
        baseline, timeout_seconds=timing.save_timeout_seconds
    )
    watcher.wait_until_stable(
        changed_save, timeout_seconds=timing.save_timeout_seconds
    )
    extraction_seconds = time.monotonic() - cycle_start

    try:
        record = dataset.store_unique_state(
            changed_save,
            index=index,
            # Include the configured exit-navigation waits in the persisted
            # estimate so historical ETA does not systematically omit them.
            cycle_seconds=(
                extraction_seconds
                + float(ui.click_delays["settings"])
                + float(ui.click_delays["exit_to_menu"])
            ),
        )
    finally:
        # Once a changed save has been found, the game is known to be in-game and
        # this is the only bounded recovery/navigation path used by the MVP.
        ui.click("settings")
        ui.click("exit_to_menu")
    return record, time.monotonic() - cycle_start


def run_harvest(
    config: HarvesterConfig,
    dataset: DatasetManager,
    target_count: int,
    pyautogui_module: Any,
) -> int:
    timing = config.timing
    watcher = SaveWatcher(
        Path(config.save_dir),
        poll_interval=timing.poll_interval,
        stability_seconds=timing.file_stability_seconds,
    )
    ui = UIController(pyautogui_module, config.buttons, config.click_delays)
    progress = ProgressTracker(target_count, dataset.historical_cycle_times())
    duplicate_retries = 0

    while dataset.existing_count() < target_count:
        index = dataset.next_index()
        try:
            record, total_cycle_seconds = harvest_one_map(
                ui, watcher, dataset, index, timing
            )
            duplicate_retries = 0
            current = dataset.existing_count()
            LOGGER.info(
                "harvested index=%s filename=%s cycle_seconds=%.3f sha256=%s",
                record.index,
                record.filename,
                total_cycle_seconds,
                record.sha256,
            )
            print(
                progress.progress_line(
                    current, record.filename, total_cycle_seconds
                )
            )
        except DuplicateStateError as exc:
            duplicate_retries += 1
            LOGGER.warning(
                "duplicate retry=%s/%s index=%s existing=%s sha256=%s",
                duplicate_retries,
                timing.max_duplicate_retries,
                index,
                exc.existing_filename,
                exc.sha256,
            )
            print(
                f"Duplicate save (matches {exc.existing_filename}); retrying "
                f"[{duplicate_retries}/{timing.max_duplicate_retries}]."
            )
            if duplicate_retries > timing.max_duplicate_retries:
                raise RuntimeError(
                    "duplicate retry limit exceeded; stopping to avoid an unsafe click loop"
                ) from exc
        except (SaveTimeoutError, FileStabilityError) as exc:
            LOGGER.error("save detection failed at index=%s: %s", index, exc)
            raise RuntimeError(
                f"{exc}. Screen state is uncertain; restore Polytopia to the main menu and rerun."
            ) from exc
        except CopyIntegrityError as exc:
            LOGGER.error("copy integrity failed at index=%s: %s", index, exc)
            raise RuntimeError(
                f"{exc}. The source was not modified; inspect Polytopia and resume manually."
            ) from exc
    return dataset.existing_count()


def dry_run(config: HarvesterConfig, dataset: DatasetManager, count: int) -> None:
    print_startup_summary(config, dataset, count, None)
    print("\nDRY RUN - no mouse clicks or harvested state/manifest writes will occur.")
    print("Click sequence:")
    for _, label in BUTTON_SEQUENCE[:6]:
        print(f"  -> {label}")
    print("  -> Detect changed .state, wait for stability, copy and verify")
    for _, label in BUTTON_SEQUENCE[6:]:
        print(f"  -> {label}")


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    pyautogui_module = None
    if args.calibrate or not args.dry_run:
        pyautogui_module = import_pyautogui()

    config = resolve_config(args, pyautogui_module)
    output_dir = Path(config.output_dir).expanduser().resolve()
    dataset = DatasetManager(output_dir)
    LOGGER.info(
        "startup target=%s save_dir=%s output_dir=%s config=%s dry_run=%s",
        args.count,
        config.save_dir,
        output_dir,
        args.config,
        args.dry_run,
    )

    if args.count is None:
        print("Calibration complete. No harvest was requested.")
        return 0
    if args.dry_run:
        dry_run(config, dataset, args.count)
        return 0

    current_size = pyautogui_module.size()
    current_resolution = (int(current_size.width), int(current_size.height))
    validate_screen_resolution(config, current_resolution)
    print_startup_summary(config, dataset, args.count, current_resolution)
    if dataset.remaining_to_target(args.count) == 0:
        print("Target already satisfied; nothing to harvest.")
        return 0

    print_safety_prompt()
    countdown(3)
    if not warn_if_polytopia_not_foreground(config.foreground_title_contains):
        raise RuntimeError(
            "Polytopia is not foregrounded; no clicks were sent. Focus the game and rerun."
        )

    try:
        completed = run_harvest(
            config, dataset, args.count, pyautogui_module
        )
    except KeyboardInterrupt:
        completed = dataset.existing_count()
        LOGGER.info("interrupted completed=%s target=%s", completed, args.count)
        print(f"\nInterrupted safely. Harvested maps retained: {completed:,}/{args.count:,}")
        return 130
    except pyautogui_module.FailSafeException:
        completed = dataset.existing_count()
        LOGGER.warning("PyAutoGUI failsafe completed=%s target=%s", completed, args.count)
        print(f"\nEmergency stop triggered. Harvested maps retained: {completed:,}/{args.count:,}")
        return 130

    LOGGER.info("complete completed=%s target=%s", completed, args.count)
    print(f"\nHarvest complete: {completed:,}/{args.count:,} unique states.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        LOGGER.info("interrupted before or outside the active harvest loop")
        print("\nInterrupted safely. No completed harvested files were removed.")
        raise SystemExit(130)
    except (ConfigError, RuntimeError) as exc:
        LOGGER.exception("fatal error: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
