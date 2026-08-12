from __future__ import annotations

import ctypes
import time
from typing import Any, Callable, Dict, Tuple


class UIError(RuntimeError):
    pass


class UIController:
    def __init__(
        self,
        pyautogui_module: Any,
        buttons: Dict[str, Tuple[int, int]],
        click_delays: Dict[str, float],
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.pyautogui = pyautogui_module
        self.buttons = buttons
        self.click_delays = click_delays
        self.sleep = sleep_fn
        self.pyautogui.FAILSAFE = True
        self.pyautogui.PAUSE = 0.05

    def click(self, button_name: str) -> None:
        if button_name not in self.buttons:
            raise UIError(f"button is not calibrated: {button_name}")
        x, y = self.buttons[button_name]
        self.pyautogui.click(int(x), int(y))
        self.sleep(float(self.click_delays[button_name]))


def foreground_window_title() -> str:
    if not hasattr(ctypes, "windll"):
        return ""
    user32 = ctypes.windll.user32
    handle = user32.GetForegroundWindow()
    length = user32.GetWindowTextLengthW(handle)
    if length <= 0:
        return ""
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(handle, buffer, length + 1)
    return buffer.value


def warn_if_polytopia_not_foreground(title_fragment: str = "Polytopia") -> bool:
    title = foreground_window_title()
    if title_fragment and title_fragment.lower() not in title.lower():
        print(
            "WARNING: Polytopia does not appear to be the foreground window "
            f"(current title: {title or '<unknown>'})."
        )
        print("The countdown gives you time to focus Polytopia before clicks begin.")
        return False
    return True
