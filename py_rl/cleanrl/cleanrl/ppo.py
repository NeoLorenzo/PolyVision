# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import json
import random
import time
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

import importlib.util
import sys

# add repo root (two levels up) so `pol_env` can be imported by name
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# try normal import first; fall back to loading by file path if needed
try:
    import pol_env.Tribes.py.register_env as register_env  # adjust if folder name differs
except Exception:
    _fpath = os.path.join(_repo_root, "pol_env", "Tribes", "py", "register_env.py")
    spec = importlib.util.spec_from_file_location("register_env", _fpath)
    register_env = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(register_env)

from pol_env.Tribes.py.register_env import TribesGymWrapper  # adjust if folder name differs

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save the trained model to disk at the end of training"""
    model_path: Optional[str] = None
    """optional output path for saved model; defaults to runs/{run_name}/{exp_name}.cleanrl_model"""
    save_frequency: int = 500000
    """checkpoint frequency in environment steps when --save-model is enabled"""

    # Algorithm specific arguments
    # env_id: str = "CartPole-v1"
    env_id: str = "Tribes-v0"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 12
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: Optional[float] = None
    """the target KL divergence threshold"""
    startup_jitter_min_s: float = 0.1
    """minimum randomized startup delay (seconds) before each env launches its JVM"""
    startup_jitter_max_s: float = 2.0
    """maximum randomized startup delay (seconds) before each env launches its JVM"""
    enable_step_diagnostics: bool = False
    """if toggled, compute and log extra per-step wrapper diagnostics (slower)"""
    validate_action_interface: bool = True
    """if toggled, run strict pre-training action-interface validation and fail on any issue"""
    validation_states: int = 10000
    """number of decision states to validate before training starts"""
    validation_seed: int = 12345
    """seed for pre-training action-interface validation"""
    max_illegal_sample_rate: float = 0.0001
    """abort training if illegal_sample_rate exceeds this threshold (0.0001 = 0.01%)"""
    max_fallback_end_turn_rate: float = 0.0001
    """abort training if fallback_end_turn_rate exceeds this threshold (0.0001 = 0.01%)"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name, startup_jitter_min_s=0.1, startup_jitter_max_s=2.0):
    def thunk():
        # Spread out JVM launches to avoid a process-creation boot storm.
        startup_delay = random.uniform(startup_jitter_min_s, startup_jitter_max_s)
        time.sleep(startup_delay)
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None, action_mask=None):
        logits = self.actor(x)
        if action_mask is not None:
            # Mask invalid actions by pushing logits to a very large negative value.
            # action_mask is expected to be 1 for valid actions, 0 for invalid.
            logits = logits.masked_fill(action_mask <= 0, -1e8)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


def _extract_vector_action_mask(infos, num_envs, action_dim, device):
    """Extracts action masks from vectorized infos dict.

    Returns a tensor of shape [num_envs, action_dim] with 1.0 for valid actions and 0.0 for invalid.
    Falls back to all-ones if mask is unavailable.
    """
    mask = np.ones((num_envs, action_dim), dtype=np.float32)
    if infos is not None and "action_mask" in infos:
        raw_mask = infos["action_mask"]
        valid_mask = infos.get("_action_mask", None)
        if valid_mask is None:
            arr = np.asarray(raw_mask, dtype=np.float32)
            if arr.ndim == 1:
                arr = np.tile(arr, (num_envs, 1))
            if arr.shape[0] == num_envs and arr.shape[1] == action_dim:
                mask = arr
        else:
            arr = np.asarray(raw_mask, dtype=np.float32)
            vmask = np.asarray(valid_mask, dtype=bool)
            if arr.ndim == 2 and vmask.ndim == 1 and len(vmask) == num_envs:
                for i in range(num_envs):
                    if vmask[i]:
                        mask[i] = arr[i]
    elif infos is not None and "legal_global_ids" in infos:
        # Fast path: each env returns sparse legal global IDs; rebuild dense mask locally.
        mask = np.zeros((num_envs, action_dim), dtype=np.float32)
        raw_ids = infos["legal_global_ids"]
        valid_mask = infos.get("_legal_global_ids", None)

        def _fill_from_ids(row, ids_like):
            try:
                ids_arr = np.asarray(ids_like).reshape(-1)
            except Exception:
                ids_arr = np.asarray([], dtype=np.int64)
            for gid in ids_arr:
                try:
                    g = int(gid)
                except Exception:
                    continue
                if 0 <= g < action_dim:
                    row[g] = 1.0

        try:
            if valid_mask is None:
                if len(raw_ids) == num_envs:
                    for i in range(num_envs):
                        _fill_from_ids(mask[i], raw_ids[i])
                else:
                    # Broadcast singleton payload if needed.
                    for i in range(num_envs):
                        _fill_from_ids(mask[i], raw_ids)
            else:
                vmask = np.asarray(valid_mask, dtype=bool)
                if len(vmask) == num_envs and len(raw_ids) == num_envs:
                    for i in range(num_envs):
                        if vmask[i]:
                            _fill_from_ids(mask[i], raw_ids[i])
                        else:
                            mask[i] = 1.0
        except Exception:
            # Keep robust fallback behavior.
            mask = np.ones((num_envs, action_dim), dtype=np.float32)
    return torch.tensor(mask, dtype=torch.float32, device=device)


def _extract_action_mask_from_info_dict(info, action_dim):
    """Single-env mask extraction for validator; supports dense and sparse formats."""
    if not isinstance(info, dict):
        return None
    if "action_mask" in info:
        arr = np.asarray(info.get("action_mask", None), dtype=np.float32)
        if arr.ndim == 1 and arr.shape[0] == action_dim:
            return arr
    if "legal_global_ids" in info:
        mask = np.zeros((action_dim,), dtype=np.float32)
        try:
            ids_arr = np.asarray(info.get("legal_global_ids", [])).reshape(-1)
        except Exception:
            ids_arr = np.asarray([], dtype=np.int64)
        for gid in ids_arr:
            try:
                g = int(gid)
            except Exception:
                continue
            if 0 <= g < action_dim:
                mask[g] = 1.0
        return mask
    return None


def _extract_vector_field(infos, key, num_envs, default_value=None):
    """Extract a per-env field from Gymnasium vector infos with optional validity mask.

    Returns a Python list of length num_envs with values or default_value.
    """
    out = [default_value for _ in range(num_envs)]
    if infos is None or key not in infos:
        return out

    raw = infos[key]
    mask_key = f"_{key}"
    valid_mask = infos.get(mask_key, None)

    try:
        if valid_mask is None:
            if len(raw) == num_envs:
                for i in range(num_envs):
                    out[i] = raw[i]
            else:
                # Broadcast scalar/singleton-like values.
                for i in range(num_envs):
                    out[i] = raw
        else:
            if len(raw) == num_envs and len(valid_mask) == num_envs:
                for i in range(num_envs):
                    if valid_mask[i]:
                        out[i] = raw[i]
    except Exception:
        pass
    return out


def _validate_action_interface(env_id: str, states: int, seed: int):
    env = gym.make(env_id)
    try:
        obs, info = env.reset(seed=seed)
        checked = 0
        saw_spawn_warrior = False
        saw_resource_animal = False
        saw_capture_village = False
        saw_research_forestry = False
        saw_clear_forest = False

        # Gate: no modulo mapping remnants in wrapper selection path.
        try:
            wrapper_path = os.path.join(_repo_root, "pol_env", "Tribes", "py", "register_env.py")
            src = open(wrapper_path, "r", encoding="utf-8").read()
            if "selected_allowed_pos" in src or "% len(allowed_indices)" in src:
                raise RuntimeError("Modulo-based allowed-index mapping remnants detected in register_env.py")
        except Exception as e:
            raise RuntimeError(f"Failed modulo-mapping gate check: {e}")

        while checked < int(states):
            if not isinstance(info, dict):
                raise RuntimeError("Validator expected dict info payload from env reset/step.")

            legal_total = int(info.get("legal_actions_total", 0))
            canonicalized = int(info.get("canonicalized_legal_actions", 0))
            uncanonicalized = int(info.get("uncanonicalized_legal_actions", 0))
            collisions = int(info.get("duplicate_global_id_collisions", 0))
            mask_ones = int(info.get("mask_ones", -1))
            unique_ids = int(info.get("unique_legal_global_ids", -1))

            if collisions != 0:
                raise RuntimeError(f"Validator failed: collisions={collisions} at checked_state={checked}")
            if uncanonicalized != 0:
                raise RuntimeError(
                    f"Validator failed: uncanonicalized={uncanonicalized} at checked_state={checked}; "
                    f"examples={info.get('uncanonicalized_repr_examples', [])}"
                )
            if mask_ones != unique_ids:
                raise RuntimeError(
                    f"Validator failed: mask_ones ({mask_ones}) != unique_legal_global_ids ({unique_ids}) "
                    f"at checked_state={checked}"
                )
            if canonicalized != unique_ids:
                raise RuntimeError(
                    f"Validator failed: canonicalized ({canonicalized}) != unique_ids ({unique_ids}) "
                    f"at checked_state={checked}"
                )
            if legal_total < canonicalized:
                raise RuntimeError(
                    f"Validator failed: legal_total ({legal_total}) < canonicalized ({canonicalized}) "
                    f"at checked_state={checked}"
                )

            action_mask = _extract_action_mask_from_info_dict(info, env.action_space.n)
            if action_mask is None or action_mask.ndim != 1:
                raise RuntimeError("Validator failed: action_mask missing or not 1D.")
            if int(np.sum(action_mask)) != mask_ones:
                raise RuntimeError("Validator failed: action_mask sum mismatch with mask_ones.")

            valid_ids = np.where(action_mask > 0)[0]
            if len(valid_ids) == 0:
                raise RuntimeError("Validator failed: no valid masked IDs.")

            # Situation-specific coverage checks against current legal actions.
            try:
                uw = env.unwrapped
                legal_actions = uw.tribes_env.list_actions()
                for a in legal_actions:
                    a_type = str(a.get("type", "")).upper()
                    gid, _reason = uw._canonicalize_action_to_global_id(a, uw.tribes_env._last_obs)
                    if gid is None:
                        continue
                    if action_mask[int(gid)] <= 0:
                        continue
                    repr_s = str(a.get("repr", "")).upper()
                    if a_type in ("SPAWN", "TRAIN") and "WARRIOR" in repr_s:
                        saw_spawn_warrior = True
                    if a_type == "RESOURCE_GATHERING" and "ANIMAL" in repr_s:
                        saw_resource_animal = True
                    if a_type == "CAPTURE" and "VILLAGE" in repr_s:
                        saw_capture_village = True
                    if a_type == "RESEARCH_TECH" and "FORESTRY" in repr_s:
                        saw_research_forestry = True
                    if a_type == "CLEAR_FOREST":
                        saw_clear_forest = True
            except Exception as e:
                raise RuntimeError(f"Validator failed in situation-specific coverage checks: {e}")

            sampled = int(np.random.choice(valid_ids))
            obs, reward, terminated, truncated, info = env.step(sampled)

            if bool(info.get("illegal_sampled_global_id", False)):
                raise RuntimeError(f"Validator failed: sampled legal id {sampled} marked illegal at checked_state={checked}")
            if bool(info.get("fallback_to_end_turn", False)):
                raise RuntimeError(f"Validator failed: fallback_to_end_turn triggered for sampled legal id {sampled}")
            if int(info.get("selected_global_id", -1)) != sampled:
                raise RuntimeError(
                    f"Validator failed: selected_global_id {info.get('selected_global_id')} "
                    f"!= sampled {sampled}"
                )

            # Situational coverage checks from exposed type counts and selected legal actions.
            by_type = info.get("legal_action_count_by_type", {}) if isinstance(info.get("legal_action_count_by_type", {}), dict) else {}
            if by_type.get("SPAWN", 0) > 0 and by_type.get("SPAWN", 0) != 0 and int(info.get("mask_ones", 0)) <= 0:
                raise RuntimeError("Validator failed: SPAWN opportunities present but no masked legal IDs.")
            if by_type.get("CAPTURE", 0) > 0 and int(info.get("mask_ones", 0)) <= 0:
                raise RuntimeError("Validator failed: CAPTURE opportunities present but no masked legal IDs.")
            if by_type.get("RESOURCE_GATHERING", 0) > 0 and int(info.get("mask_ones", 0)) <= 0:
                raise RuntimeError("Validator failed: RESOURCE_GATHERING opportunities present but no masked legal IDs.")
            if by_type.get("RESEARCH_TECH", 0) > 0 and int(info.get("mask_ones", 0)) <= 0:
                raise RuntimeError("Validator failed: RESEARCH_TECH opportunities present but no masked legal IDs.")
            if by_type.get("CLEAR_FOREST", 0) > 0 and int(info.get("mask_ones", 0)) <= 0:
                raise RuntimeError("Validator failed: CLEAR_FOREST opportunities present but no masked legal IDs.")

            checked += 1
            if terminated or truncated:
                obs, info = env.reset()

        print(f"[ACTION_VALIDATOR] Passed strict validation over {checked} decision states.")
        print(
            "[ACTION_VALIDATOR] Situation coverage seen:"
            f" spawn_warrior={saw_spawn_warrior},"
            f" resource_animal={saw_resource_animal},"
            f" capture_village={saw_capture_village},"
            f" research_forestry={saw_research_forestry},"
            f" clear_forest={saw_clear_forest}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    run_dir = os.path.join("runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    if args.track:
        import wandb
        import os
        import sys
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(run_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    if args.validate_action_interface:
        _validate_action_interface(args.env_id, args.validation_states, args.validation_seed)

    # env setup
    envs = gym.vector.AsyncVectorEnv(
        [
            make_env(
                args.env_id,
                i,
                args.capture_video,
                run_name,
                args.startup_jitter_min_s,
                args.startup_jitter_max_s,
            )
            for i in range(args.num_envs)
        ],
        context="spawn",
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, reset_infos = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    next_action_mask = _extract_vector_action_mask(
        reset_infos,
        args.num_envs,
        envs.single_action_space.n,
        device,
    )
    if next_action_mask.shape[-1] != envs.single_action_space.n:
        raise RuntimeError(
            f"action_mask.shape[-1]={next_action_mask.shape[-1]} does not match action space n={envs.single_action_space.n}"
        )
    if agent.actor[-1].out_features != envs.single_action_space.n:
        raise RuntimeError(
            f"ppo_policy_output_dim={agent.actor[-1].out_features} does not match env.action_space.n={envs.single_action_space.n}"
        )
    action_interface_meta = {
        "catalog_version": None,
        "canonicalizer_version": None,
        "map_width": None,
        "map_height": None,
        "global_action_space_n": int(envs.single_action_space.n),
        "action_offset_table_hash": None,
    }
    for k in ["catalog_version", "canonicalizer_version", "map_width", "map_height", "global_action_space_n", "action_offset_table_hash"]:
        vals = _extract_vector_field(reset_infos, k, args.num_envs, default_value=None)
        first = next((v for v in vals if v is not None), None)
        action_interface_meta[k] = first
    if action_interface_meta.get("global_action_space_n") is not None:
        if int(action_interface_meta["global_action_space_n"]) != int(envs.single_action_space.n):
            raise RuntimeError(
                f"Reported global_action_space_n={action_interface_meta['global_action_space_n']} "
                f"does not match env action space n={envs.single_action_space.n}"
            )

    for iteration in range(1, args.num_iterations + 1):
        # Per-iteration diagnostics for dashboard clarity.
        iter_valid_actions_sum = 0.0
        iter_valid_actions_count = 0
        iter_non_endturn_count = 0
        iter_action_count = 0
        iter_delta_spt_sum = 0.0
        iter_delta_spt_count = 0

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            prev_global_step = global_step
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs, action_mask=next_action_mask)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
            next_action_mask = _extract_vector_action_mask(
                infos,
                args.num_envs,
                envs.single_action_space.n,
                device,
            )

            collision_values = _extract_vector_field(infos, "duplicate_global_id_collisions", args.num_envs, default_value=0)
            if any(int(v) > 0 for v in collision_values if v is not None):
                raise RuntimeError(f"Detected duplicate_global_id_collisions in training step at global_step={global_step}")

            uncanonicalized_values = _extract_vector_field(infos, "uncanonicalized_legal_actions", args.num_envs, default_value=0)
            if any(int(v) > 0 for v in uncanonicalized_values if v is not None):
                raise RuntimeError(
                    f"Detected uncanonicalized_legal_actions in training step at global_step={global_step}"
                )

            illegal_rate_values = _extract_vector_field(infos, "illegal_sample_rate", args.num_envs, default_value=0.0)
            fallback_rate_values = _extract_vector_field(infos, "fallback_end_turn_rate", args.num_envs, default_value=0.0)
            max_illegal_rate = max(float(v) for v in illegal_rate_values if v is not None) if illegal_rate_values else 0.0
            max_fallback_rate = max(float(v) for v in fallback_rate_values if v is not None) if fallback_rate_values else 0.0
            if max_illegal_rate > float(args.max_illegal_sample_rate):
                raise RuntimeError(
                    f"illegal_sample_rate={max_illegal_rate:.6f} exceeded threshold={args.max_illegal_sample_rate:.6f} "
                    f"at global_step={global_step}"
                )
            if max_fallback_rate > float(args.max_fallback_end_turn_rate):
                raise RuntimeError(
                    f"fallback_end_turn_rate={max_fallback_rate:.6f} exceeded threshold={args.max_fallback_end_turn_rate:.6f} "
                    f"at global_step={global_step}"
                )

            if args.enable_step_diagnostics:
                # ---- Custom diagnostics from wrapper infos ----
                valid_actions_values = _extract_vector_field(infos, "valid_actions", args.num_envs, default_value=None)
                for v in valid_actions_values:
                    if v is not None:
                        iter_valid_actions_sum += float(v)
                        iter_valid_actions_count += 1

                selected_action_type_values = _extract_vector_field(
                    infos, "selected_action_type", args.num_envs, default_value=None
                )
                for a_type in selected_action_type_values:
                    if a_type is not None:
                        iter_action_count += 1
                        if str(a_type) != "END_TURN":
                            iter_non_endturn_count += 1

                delta_spt_values = _extract_vector_field(infos, "delta_spt", args.num_envs, default_value=None)
                for dspt in delta_spt_values:
                    if dspt is not None:
                        iter_delta_spt_sum += float(dspt)
                        iter_delta_spt_count += 1

                # Core action-interface telemetry (means across envs where available).
                for key, chart_name in [
                    ("turn", "charts/turn"),
                    ("unit_count", "charts/unit_count"),
                    ("stars", "charts/stars"),
                    ("reward", "charts/reward"),
                    ("selected_global_id", "charts/selected_global_id"),
                    ("selected_raw_java_index", "charts/selected_raw_java_index"),
                ]:
                    vals = _extract_vector_field(infos, key, args.num_envs, default_value=None)
                    clean = [float(v) for v in vals if v is not None]
                    if len(clean) > 0:
                        writer.add_scalar(chart_name, float(np.mean(clean)), global_step)

                # Custom Phase-1 telemetry:
                # Log SPT for envs that just ended (typically via truncation at turn horizon).
                for env_idx in range(args.num_envs):
                    if truncations[env_idx] or terminations[env_idx]:
                        spt_value = None
                        city_count_value = None
                        fog_cleared_total_value = None

                        def _extract_done_metric(metric_key):
                            if metric_key not in infos:
                                return None
                            raw = infos[metric_key]
                            mask = infos.get(f"_{metric_key}", None)
                            if mask is None:
                                if len(raw) > env_idx:
                                    return raw[env_idx]
                            else:
                                if len(mask) > env_idx and mask[env_idx] and len(raw) > env_idx:
                                    return raw[env_idx]
                            return None

                        # Case 1: vector info carries per-env arrays and optional validity mask.
                        spt_value = _extract_done_metric("spt")
                        city_count_value = _extract_done_metric("city_count")
                        fog_cleared_total_value = _extract_done_metric("fog_tiles_cleared_total")

                        # Case 2: final_info often carries final per-env info dicts.
                        if "final_info" in infos and len(infos["final_info"]) > env_idx:
                            finfo = infos["final_info"][env_idx]
                            if finfo is not None:
                                if spt_value is None and "spt" in finfo:
                                    spt_value = finfo["spt"]
                                if city_count_value is None and "city_count" in finfo:
                                    city_count_value = finfo["city_count"]
                                if fog_cleared_total_value is None and "fog_tiles_cleared_total" in finfo:
                                    fog_cleared_total_value = finfo["fog_tiles_cleared_total"]

                        if spt_value is not None:
                            writer.add_scalar("charts/custom_spt_return", float(spt_value), global_step)
                            writer.add_scalar("charts/episode_end_spt", float(spt_value), global_step)
                        if city_count_value is not None:
                            writer.add_scalar("charts/episode_end_village_count_t10", float(city_count_value), global_step)
                        if fog_cleared_total_value is not None:
                            writer.add_scalar("charts/episode_end_fog_tiles_cleared_t10", float(fog_cleared_total_value), global_step)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    if args.enable_step_diagnostics and info and "spt" in info:
                        writer.add_scalar("charts/custom_spt_return", float(info["spt"]), global_step)

            # Periodic checkpointing. Save every crossed frequency milestone in case
            # num_envs does not divide save_frequency exactly.
            if args.save_model and args.save_frequency > 0 and global_step > 0:
                first_checkpoint = ((prev_global_step // args.save_frequency) + 1) * args.save_frequency
                last_checkpoint = (global_step // args.save_frequency) * args.save_frequency
                if first_checkpoint <= last_checkpoint:
                    for checkpoint_step in range(first_checkpoint, last_checkpoint + 1, args.save_frequency):
                        if checkpoint_step > 0 and checkpoint_step % args.save_frequency == 0:
                            checkpoint_path = os.path.join(run_dir, f"model_checkpoint_{checkpoint_step}.cleanrl_model")
                            torch.save(agent.state_dict(), checkpoint_path)
                            meta_path = checkpoint_path + ".action_interface.json"
                            try:
                                with open(meta_path, "w", encoding="utf-8") as f:
                                    json.dump(action_interface_meta, f, indent=2, sort_keys=True, default=str)
                            except Exception as e:
                                print(f"warning: failed to save action interface metadata: {e}")
                            print(f"checkpoint_saved={checkpoint_path}")

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds],
                    b_actions.long()[mb_inds],
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        if args.enable_step_diagnostics and iter_valid_actions_count > 0:
            writer.add_scalar(
                "charts/mean_valid_actions",
                iter_valid_actions_sum / iter_valid_actions_count,
                global_step,
            )
        if args.enable_step_diagnostics and iter_action_count > 0:
            writer.add_scalar(
                "charts/non_endturn_rate",
                iter_non_endturn_count / iter_action_count,
                global_step,
            )
        if args.enable_step_diagnostics and iter_delta_spt_count > 0:
            writer.add_scalar(
                "charts/mean_delta_spt",
                iter_delta_spt_sum / iter_delta_spt_count,
                global_step,
            )
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = args.model_path or os.path.join(run_dir, f"{args.exp_name}.cleanrl_model")
        torch.save(agent.state_dict(), model_path)
        meta_path = model_path + ".action_interface.json"
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(action_interface_meta, f, indent=2, sort_keys=True, default=str)
        except Exception as e:
            print(f"warning: failed to save action interface metadata: {e}")
        print(f"model_saved={model_path}")

    envs.close()
    writer.close()
