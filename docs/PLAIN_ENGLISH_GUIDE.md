# Polyvision in Plain English

This guide explains the project in simple terms.

## What is Polyvision?

Polyvision is a project that teaches an AI to play Polytopia better over time.

Think of it like this:
- The game is the "world"
- The AI is the "student"
- Training is repeated practice
- Rewards are the scorecards that tell the student if it made good choices

Over many games, the AI learns patterns that lead to stronger outcomes.

## What is the goal of the project?

The long-term goal is to build strong AI players for Polytopia and create a fair way to compare them.

Short-term (Phase 1) goal:
- Focus only on economy growth (not combat)
- Keep training setup simple and stable
- Optimize for better "Stars Per Turn" by turn 10

Later phases add combat, stronger opponents, and community benchmarking.

## How does this codebase work?

The project has three main layers:

1) Java game engine (`Tribes`)
- This is the game rules system.
- It handles turns, actions, scoring, and game state.

2) Python bridge (`Py4J`)
- This is the translator between Python and Java.
- Python can ask the Java engine: "reset game", "take action", "give me state".

3) Python training side (Gym-style env + RL scripts)
- This wraps the game in a standard training interface.
- RL scripts use that interface to run many episodes and learn.

In plain terms: Java runs the game, Python runs the learning, and Py4J connects them.

## What happens during training?

Each training loop usually looks like this:

1. Start a new game (`reset`)
2. Ask what actions are possible
3. Model chooses an action
4. Game applies action (`step`)
5. Model gets reward/feedback
6. Repeat until game ends
7. Update model so good choices become more likely

This loop runs thousands of times.

## Where are the important files?

- Core environment bridge:
  - `pol_env/Tribes/py/gym_env.py`
  - `pol_env/Tribes/py/register_env.py`
- Quick runner:
  - `pol_env/Tribes/py/run_gym.py`
- Training/evaluation scripts:
  - `py_rl/cleanrl/cleanrl/ppo.py`
  - `py_rl/cleanrl/our_cleanrl/ppo_semantic.py`
  - `py_rl/cleanrl/our_cleanrl/ppo_action_quality.py`
  - `quick_eval.py`
  - `evaluate_models.py`

## How could someone design their own model?

You do not need to start from scratch. A practical path is:

1. Pick a baseline
- Start with PPO (`py_rl/cleanrl/cleanrl/ppo.py`)
- Run a short test first so everything works

2. Decide what your model should optimize
- Economy only?
- Fast expansion?
- Balanced growth and safety?

3. Choose model strategy
- Baseline PPO: simple starting point
- Semantic model: uses action meaning/type
- Action-quality model: learns which actions are likely best

4. Define reward clearly
- Reward is your teaching signal
- If reward is unclear, model learns noisy behavior
- For Phase 1, tie reward closely to economy outcomes

5. Train in small steps
- Run short experiments first (sanity check)
- Then increase timesteps and compare results

6. Compare runs fairly
- Keep seeds and key settings consistent
- Log results, commands, and model settings
- Compare score trends, not just one lucky run

## How could someone train their own model (simple checklist)?

1. Set up environment
- Compile Java classes
- Create Python venv
- Install requirements

2. Confirm the game loop works
- Run `python test_simple.py`
- Run `cd pol_env/Tribes/py && python run_gym.py`

3. Run a baseline training job
- `python py_rl/cleanrl/cleanrl/ppo.py --total-timesteps 5000 --num-steps 32 --track False`

4. Try a custom variant
- `python py_rl/cleanrl/our_cleanrl/ppo_semantic.py ...`
- `python py_rl/cleanrl/our_cleanrl/ppo_action_quality.py ...`

5. Evaluate and compare
- `python quick_eval.py`
- `python evaluate_models.py`

## What does "good progress" look like?

In early development, good progress means:
- Runs complete without crashing
- Reward trends are stable (not random spikes only)
- Average outcomes improve over repeated runs
- Behavior matches your training objective

## Final note

This project is a fork built on substantial upstream work. Polyvision extends that foundation with a clear roadmap, focused MVP, and accessible training workflow for contributors.
