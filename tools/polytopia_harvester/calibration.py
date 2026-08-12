from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from config import BUTTON_SEQUENCE, HarvesterConfig, save_config


LOGGER = logging.getLogger("polytopia_harvester")


def calibrate(
    pyautogui_module: Any,
    config: HarvesterConfig,
    config_path: Path,
) -> HarvesterConfig:
    print("\nPOLYVISION - POLYTOPIA MAP HARVESTER\n")
    print("Calibration mode. Coordinates are captured when you press ENTER.\n")
    buttons = {}
    for key, label in BUTTON_SEQUENCE:
        input(f'Move your mouse over the "{label}" button. Press ENTER when ready. ')
        position = pyautogui_module.position()
        coordinate = (int(position.x), int(position.y))
        buttons[key] = coordinate
        print(f"Captured {label}: {coordinate}\n")
        LOGGER.info("calibrated button=%s x=%s y=%s", key, coordinate[0], coordinate[1])

    size = pyautogui_module.size()
    config.buttons = buttons
    config.screen_width = int(size.width)
    config.screen_height = int(size.height)
    save_config(config, config_path)
    LOGGER.info(
        "calibration saved path=%s resolution=%sx%s",
        config_path,
        config.screen_width,
        config.screen_height,
    )
    print(f"Calibration saved to: {config_path}")
    return config
