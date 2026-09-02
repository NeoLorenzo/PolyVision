#!/usr/bin/env python3
"""Compile the Tribes engine and run PolyVision's cheap Phase 1 CI gate."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TRIBES_ROOT = REPOSITORY_ROOT / "pol_env" / "Tribes"
SOURCE_ROOT = TRIBES_ROOT / "src"
JSON_JAR = TRIBES_ROOT / "lib" / "json.jar"
OUTPUT_ROOT = TRIBES_ROOT / "out"
TEST_FILES = (
    "pol_env/Tribes/py/tests/test_environment_contract.py",
    "pol_env/Tribes/py/tests/test_parity_001_city_state.py",
    "pol_env/Tribes/py/tests/test_parity_002_human_information_parity.py",
)


def run_stage(name: str, command: list[str]) -> None:
    print(f"\n==> {name}")
    print("$", subprocess.list2cmdline(command))
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)


def main() -> int:
    if shutil.which("javac") is None:
        print("ERROR: javac is unavailable. Install JDK 11 and ensure javac is on PATH.", file=sys.stderr)
        return 1

    missing_paths = [path for path in (SOURCE_ROOT, JSON_JAR) if not path.exists()]
    if missing_paths:
        for path in missing_paths:
            print(f"ERROR: required path does not exist: {path}", file=sys.stderr)
        return 1

    java_sources = sorted(SOURCE_ROOT.rglob("*.java"))
    if not java_sources:
        print(f"ERROR: no Java sources found under {SOURCE_ROOT}", file=sys.stderr)
        return 1

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    compile_command = [
        "javac",
        "-cp",
        str(JSON_JAR),
        "-d",
        str(OUTPUT_ROOT),
        "-sourcepath",
        str(SOURCE_ROOT),
        *map(str, java_sources),
    ]

    try:
        run_stage("Compiling Tribes Java engine", compile_command)
        run_stage("Running Phase 1 environment and parity tests", [sys.executable, "-m", "unittest", *TEST_FILES])
    except subprocess.CalledProcessError as error:
        print(f"ERROR: {error.cmd[0]} exited with status {error.returncode}.", file=sys.stderr)
        return error.returncode or 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
