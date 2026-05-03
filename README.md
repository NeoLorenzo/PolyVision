## Polyvision

Polyvision is a reinforcement learning project for training, evaluating, and benchmarking Polytopia agents using a headless Java game engine (`Tribes`) bridged to Python with `Py4J`.

This README is the canonical high-level project and vision document.

### Vision
Polyvision aims to evolve from a reproducible research environment into a competitive AI battleground with community benchmarking, and eventually into a Computer Vision-driven agent that can execute trained strategies in the live commercial game.

### Phase 1 MVP
To minimize friction and guarantee a working baseline, Phase 1 focuses on economy-first learning.

MVP constraints:
- Tribe: Bardur
- Map type: Drylands
- Turn limit: 10 turns
- Game mode: single-player (no enemies, no combat)
- Objective: maximize Stars Per Turn (SPT) by end of turn 10

MVP economic action focus:
- Move / explore
- Chop forest
- Hunt animals
- Upgrade city
- Research economy-relevant technologies

### Architecture and Tech Stack
- Game engine: legacy Java `Tribes` (headless background execution)
- Bridge: `Py4J` between Java engine and Python runtime
- Environment layer: Python + Gymnasium-compatible wrappers
- Learning approaches: PPO/A2C-style RL and experimentation with semantic/action-quality variants

### Roadmap
Phase 2: Full game loop and combat
- Expand horizon to standard longer matches
- Introduce opponent bots / multi-agent training
- Expand action space to military actions and city capture
- Shift reward focus toward score/capture outcomes

Phase 3: "Polyvision" live-game integration
- Reuse strongest trained weights from headless environment
- Add Computer Vision + device/screen control integration
- Execute live-play decisions against humans or native bots

Phase 4: Community leaderboard
- Standardize submission/evaluation contract
- Accept model submissions via pull requests
- Rank entries on repeatable benchmark matches

### Immediate Next Steps
1. Isolate MVP constraints in the Gym wrapper (Bardur/Drylands/10-turn/economy-only behavior).
2. Finalize SPT-centered reward logic for Phase 1.
3. Run the first baseline training loop with documented scripts.

### Attribution and Lineage
This repository is a fork and continuation of substantial upstream work.

- Fork lineage:
  - Upstream: `https://github.com/ClaireBookworm/polytopia_rl`
  - This repo: `https://github.com/NeoLorenzo/PolyVision`
- Core game engine + bridge foundations come from the upstream `polytopia_rl` project and its contributors.
- RL training code under `py_rl/cleanrl` includes upstream CleanRL-based components; preserve original licenses and notices when modifying or redistributing.

Polyvision extends that foundation with project-specific roadmap, evaluation priorities, and environment-facing documentation.

### Documentation Map
- Plain-English project guide: `docs/PLAIN_ENGLISH_GUIDE.md`
- Environment API: `docs/ENVIRONMENT_API.md`
- Phase 1 implementation spec: `docs/MVP_PHASE1_SPEC.md`
- Testing guide: `docs/TESTING.md`
- Training guide: `docs/TRAINING.md`
- Tribes-specific setup details: `pol_env/Tribes/README.md`

### Quickstart (Recommended)
1) Compile Java engine classes
```bash
cd pol_env/Tribes
mkdir -p out
find src -name "*.java" -exec javac -cp "lib/json.jar" -d out -sourcepath src {} +
```

2) Create and activate virtual environment
```bash
cd /workspaces/PolyVision
python3 -m venv .venv
source .venv/bin/activate
```

3) Install Python dependencies for bridge + wrappers + training scripts
```bash
pip install -U pip wheel
pip install -r pol_env/Tribes/py/requirements.txt
pip install -r py_rl/requirements.txt
```

4) Run smoke test
```bash
cd pol_env/Tribes/py
python run_gym.py
```

Expected behavior in headless environments:
- Simulation should reset and step successfully.
- A Java GUI warning (`No X11 DISPLAY variable`) may appear while simulation continues.

### Current Scope
- Working Java-to-Python environment bridge and Gymnasium-compatible wrapper (`Tribes-v0`) are in place.
- Phase 1 constraints are fully specified in docs and are in-progress for strict enforcement in runtime behavior.
