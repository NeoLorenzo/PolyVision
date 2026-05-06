import argparse
import os
import subprocess
import sys


def run(cmd, cwd):
    print(">", " ".join(cmd))
    return subprocess.run(cmd, cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Generate and freeze a Tribes map CSV from a seed.")
    parser.add_argument("--seed", type=int, required=True, help="Level generation seed")
    parser.add_argument("--size", type=int, default=16, help="Square map size (e.g., 11, 14, 16)")
    parser.add_argument("--output", type=str, default="levels/phase1_fixed.csv", help="Output CSV path relative to Tribes dir")
    parser.add_argument("--tribes", nargs="+", default=["BARDUR", "XIN_XI"], help="Tribes (at least 2) for level generation")
    parser.add_argument(
        "--initial-land",
        type=float,
        default=1.0,
        help="Initial land ratio for generator (1.0 enforces drylands-like all-land starts).",
    )
    parser.add_argument(
        "--map-type",
        type=str,
        default="DRYLANDS",
        help="Map profile (e.g., DRYLANDS, LAKES, CONTINENTS, PANGEA, ARCHIPELAGO, WATERWORLD).",
    )
    args = parser.parse_args()

    tribes_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    src_file = os.path.join(tribes_dir, "src", "core", "levelgen", "GenerateLevelCli.java")
    levelgen_src = os.path.join(tribes_dir, "src", "core", "levelgen", "LevelGenerator.java")
    out_dir = os.path.join(tribes_dir, "out")
    json_jar = os.path.join(tribes_dir, "lib", "json.jar")

    if not os.path.exists(src_file):
        raise FileNotFoundError(f"Missing source file: {src_file}")
    if not os.path.exists(levelgen_src):
        raise FileNotFoundError(f"Missing source file: {levelgen_src}")

    os.makedirs(out_dir, exist_ok=True)

    # Compile helper.
    compile_cmd = [
        "javac",
        "-cp",
        f"{out_dir}{os.pathsep}{json_jar}",
        "-d",
        out_dir,
        levelgen_src,
        src_file,
    ]
    run(compile_cmd, cwd=tribes_dir)

    # Run generator.
    run_cmd = [
        "java",
        "-cp",
        f"{out_dir}{os.pathsep}{json_jar}",
        "core.levelgen.GenerateLevelCli",
        str(args.seed),
        str(args.size),
        args.output,
        str(args.initial_land),
        str(args.map_type),
    ] + args.tribes
    run(run_cmd, cwd=tribes_dir)

    output_abs = os.path.join(tribes_dir, args.output)
    print(f"\nFrozen map written to: {output_abs}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        sys.exit(e.returncode)
