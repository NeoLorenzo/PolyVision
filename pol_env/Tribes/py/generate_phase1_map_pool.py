import argparse
import os
import subprocess
import sys


def run(cmd, cwd):
    print(">", " ".join(cmd))
    return subprocess.run(cmd, cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a deterministic pool of Phase1 12x12 maps for generalized training."
    )
    parser.add_argument("--base-seed", type=int, default=1001, help="Base seed for map-seed stream.")
    parser.add_argument("--num-maps", type=int, default=32, help="Number of maps to generate.")
    parser.add_argument("--size", type=int, default=12, help="Map size (12 for Phase1).")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="levels/phase1_pool",
        help="Output directory relative to Tribes root.",
    )
    parser.add_argument(
        "--tribes",
        nargs="+",
        default=["BARDUR", "XIN_XI"],
        help="Tribes to use for generation.",
    )
    parser.add_argument(
        "--seed-step",
        type=int,
        default=7919,
        help="Stride between generated map seeds.",
    )
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
    out_pool_dir = os.path.join(tribes_dir, args.output_dir)
    os.makedirs(out_pool_dir, exist_ok=True)

    if not os.path.exists(src_file):
        raise FileNotFoundError(f"Missing source file: {src_file}")
    if not os.path.exists(levelgen_src):
        raise FileNotFoundError(f"Missing source file: {levelgen_src}")

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

    for i in range(int(args.num_maps)):
        map_seed = int(args.base_seed) + (i * int(args.seed_step))
        rel_out = os.path.join(args.output_dir, f"phase1_12x12_pool_{i:03d}.csv").replace("\\", "/")
        run_cmd = [
            "java",
            "-cp",
            f"{out_dir}{os.pathsep}{json_jar}",
            "core.levelgen.GenerateLevelCli",
            str(map_seed),
            str(args.size),
            rel_out,
            str(args.initial_land),
            str(args.map_type),
        ] + list(args.tribes)
        run(run_cmd, cwd=tribes_dir)

    print(f"\nGenerated {args.num_maps} maps in: {out_pool_dir}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        sys.exit(e.returncode)
