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
    actor_mode: str = "legal_only"
    """policy actor mode: legal_only (default) or dense_debug"""
    max_legal_actions: int = 1024
    """fixed legal-action slot tensor length for legal_only actor mode"""
    old_logprob_recompute_tol: float = 1e-5
    """absolute tolerance for pre-update old_logprob recomputation invariant"""

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
    def __init__(self, envs, actor_mode: str = "legal_only", max_legal_actions: int = 1024):
        super().__init__()
        self.actor_mode = str(actor_mode).strip().lower()
        if self.actor_mode not in ("legal_only", "dense_debug"):
            raise ValueError(f"Unsupported actor_mode={actor_mode}. Expected legal_only or dense_debug.")
        self.max_legal_actions = int(max_legal_actions)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        if self.actor_mode == "dense_debug":
            self.actor = nn.Sequential(
                layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
            )
            self.state_encoder = None
            self.action_embedding = None
        else:
            self.state_encoder = nn.Sequential(
                layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
            )
            self.action_embedding = nn.Embedding(envs.single_action_space.n, 64)
            nn.init.normal_(self.action_embedding.weight, mean=0.0, std=0.02)
            self.actor = None

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(
        self,
        x,
        action=None,
        action_mask=None,
        legal_global_ids=None,
        legal_action_valid_mask=None,
        selected_slot=None,
    ):
        if self.actor_mode == "dense_debug":
            logits = self.actor(x)
            if action_mask is not None:
                logits = logits.masked_fill(action_mask <= 0, -1e8)
            probs = Categorical(logits=logits)
            if action is None:
                action = probs.sample()
            return action, None, probs.log_prob(action), probs.entropy(), self.critic(x)

        if legal_global_ids is None or legal_action_valid_mask is None:
            raise RuntimeError("legal_only actor_mode requires legal_global_ids and legal_action_valid_mask.")
        if legal_global_ids.ndim != 2 or legal_action_valid_mask.ndim != 2:
            raise RuntimeError("legal tensors must be rank-2: [batch, max_legal_actions].")
        if legal_global_ids.shape != legal_action_valid_mask.shape:
            raise RuntimeError(
                f"legal tensor shape mismatch: ids={tuple(legal_global_ids.shape)} "
                f"mask={tuple(legal_action_valid_mask.shape)}"
            )

        h = self.state_encoder(x)
        legal_ids = legal_global_ids.long()
        valid = legal_action_valid_mask.bool()
        # Score only legal candidate IDs using state-action dot products.
        legal_emb = self.action_embedding(legal_ids)
        logits = torch.einsum("bd,bkd->bk", h, legal_emb)
        logits = logits.masked_fill(~valid, -1e8)
        probs = Categorical(logits=logits)

        if selected_slot is None:
            selected_slot = probs.sample()
        else:
            selected_slot = selected_slot.long()
        if selected_slot.ndim != 1:
            selected_slot = selected_slot.view(-1)
        batch_n = legal_ids.shape[0]
        if selected_slot.shape[0] != batch_n:
            raise RuntimeError(
                f"selected_slot batch mismatch: got {selected_slot.shape[0]}, expected {batch_n}"
            )
        if torch.any(selected_slot < 0) or torch.any(selected_slot >= legal_ids.shape[1]):
            raise RuntimeError("selected_slot out of range for legal slot tensor.")
        slot_valid = valid.gather(1, selected_slot.unsqueeze(1)).squeeze(1)
        if not torch.all(slot_valid):
            raise RuntimeError("selected_slot points to invalid/padded legal slot.")

        selected_global_id = legal_ids.gather(1, selected_slot.unsqueeze(1)).squeeze(1)
        return selected_global_id, selected_slot, probs.log_prob(selected_slot), probs.entropy(), self.critic(x)


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
    elif infos is not None and "legal_global_ids_padded" in infos and "legal_action_valid_mask" in infos:
        mask = np.zeros((num_envs, action_dim), dtype=np.float32)
        raw_ids = infos["legal_global_ids_padded"]
        raw_valid = infos["legal_action_valid_mask"]
        ids_present = infos.get("_legal_global_ids_padded", None)
        valid_present = infos.get("_legal_action_valid_mask", None)

        def _fill_from_padded(row, ids_like, valid_like):
            ids_arr = np.asarray(ids_like, dtype=np.int64).reshape(-1)
            valid_arr = np.asarray(valid_like, dtype=bool).reshape(-1)
            if ids_arr.shape[0] != valid_arr.shape[0]:
                return
            ids_arr = ids_arr[valid_arr]
            for gid in ids_arr:
                g = int(gid)
                if 0 <= g < action_dim:
                    row[g] = 1.0

        try:
            if ids_present is None or valid_present is None:
                for i in range(num_envs):
                    _fill_from_padded(mask[i], raw_ids[i], raw_valid[i])
            else:
                ip = np.asarray(ids_present, dtype=bool).reshape(-1)
                vp = np.asarray(valid_present, dtype=bool).reshape(-1)
                for i in range(num_envs):
                    if i < len(ip) and i < len(vp) and ip[i] and vp[i]:
                        _fill_from_padded(mask[i], raw_ids[i], raw_valid[i])
                    else:
                        mask[i] = 1.0
        except Exception:
            mask = np.ones((num_envs, action_dim), dtype=np.float32)
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
    if "legal_global_ids_padded" in info and "legal_action_valid_mask" in info:
        mask = np.zeros((action_dim,), dtype=np.float32)
        try:
            ids_arr = np.asarray(info.get("legal_global_ids_padded", []), dtype=np.int64).reshape(-1)
            valid_arr = np.asarray(info.get("legal_action_valid_mask", []), dtype=bool).reshape(-1)
            if ids_arr.shape[0] != valid_arr.shape[0]:
                return None
            valid_ids = ids_arr[valid_arr]
        except Exception:
            return None
        if valid_ids.size > 0 and np.unique(valid_ids).size != valid_ids.size:
            raise RuntimeError("Validator failed: duplicate IDs in valid legal_global_ids_padded.")
        for gid in valid_ids:
            g = int(gid)
            if 0 <= g < action_dim:
                mask[g] = 1.0
        return mask
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


def _extract_vector_legal_tensors(infos, num_envs, max_legal_actions, action_dim, device):
    if infos is None:
        raise RuntimeError("Missing infos payload; cannot extract legal tensors for legal_only actor.")
    if "legal_global_ids_padded" not in infos or "legal_action_valid_mask" not in infos:
        raise RuntimeError("Infos missing legal_global_ids_padded/legal_action_valid_mask for legal_only actor.")

    raw_ids = np.asarray(infos["legal_global_ids_padded"])
    raw_valid = np.asarray(infos["legal_action_valid_mask"])
    ids_mask = infos.get("_legal_global_ids_padded", None)
    valid_mask = infos.get("_legal_action_valid_mask", None)

    ids_out = np.zeros((num_envs, max_legal_actions), dtype=np.int64)
    valid_out = np.zeros((num_envs, max_legal_actions), dtype=np.bool_)

    def _copy_row(i, src_ids, src_valid):
        ids_row = np.asarray(src_ids, dtype=np.int64).reshape(-1)
        valid_row = np.asarray(src_valid, dtype=bool).reshape(-1)
        if ids_row.shape[0] != max_legal_actions or valid_row.shape[0] != max_legal_actions:
            raise RuntimeError(
                f"legal tensor shape mismatch at env={i}: ids={ids_row.shape}, valid={valid_row.shape}, "
                f"expected=({max_legal_actions},)"
            )
        if valid_row.any():
            valid_ids = ids_row[valid_row]
            if np.unique(valid_ids).size != valid_ids.size:
                raise RuntimeError(f"Duplicate valid legal_global_ids_padded detected at env={i}.")
            if np.any(valid_ids < 0) or np.any(valid_ids >= action_dim):
                raise RuntimeError(f"Out-of-range legal global ID detected at env={i}.")
        ids_out[i] = ids_row
        valid_out[i] = valid_row

    if raw_ids.ndim == 2 and raw_valid.ndim == 2 and raw_ids.shape[0] == num_envs and raw_valid.shape[0] == num_envs:
        for i in range(num_envs):
            _copy_row(i, raw_ids[i], raw_valid[i])
    else:
        if ids_mask is None or valid_mask is None:
            raise RuntimeError(
                f"Cannot align legal tensors in vector infos: ids_shape={raw_ids.shape}, valid_shape={raw_valid.shape}"
            )
        ids_mask_arr = np.asarray(ids_mask, dtype=bool).reshape(-1)
        valid_mask_arr = np.asarray(valid_mask, dtype=bool).reshape(-1)
        if ids_mask_arr.shape[0] != num_envs or valid_mask_arr.shape[0] != num_envs:
            raise RuntimeError("Invalid _legal_* validity masks in vector infos.")
        for i in range(num_envs):
            if ids_mask_arr[i] and valid_mask_arr[i]:
                _copy_row(i, raw_ids[i], raw_valid[i])
            else:
                raise RuntimeError(f"Missing legal tensor row for env={i} in legal_only actor mode.")

    return (
        torch.tensor(ids_out, dtype=torch.long, device=device),
        torch.tensor(valid_out, dtype=torch.bool, device=device),
    )


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
            if "legal_global_ids_padded" in info and "legal_action_valid_mask" in info:
                ids_arr = np.asarray(info.get("legal_global_ids_padded", []), dtype=np.int64).reshape(-1)
                valid_arr = np.asarray(info.get("legal_action_valid_mask", []), dtype=bool).reshape(-1)
                if ids_arr.shape[0] != valid_arr.shape[0]:
                    raise RuntimeError("Validator failed: legal ids/mask length mismatch.")
                valid_ids = ids_arr[valid_arr]
                if valid_ids.size > 0 and np.unique(valid_ids).size != valid_ids.size:
                    raise RuntimeError("Validator failed: duplicate IDs in valid legal_global_ids_padded.")
                legal_count = int(info.get("legal_action_count", -1))
                if legal_count >= 0 and legal_count != int(valid_arr.sum()):
                    raise RuntimeError(
                        f"Validator failed: legal_action_count ({legal_count}) != valid_mask.sum ({int(valid_arr.sum())})."
                    )

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
    args.actor_mode = str(args.actor_mode).strip().lower()
    if args.actor_mode not in ("legal_only", "dense_debug"):
        raise RuntimeError(f"Unsupported --actor-mode {args.actor_mode}. Expected legal_only or dense_debug.")
    if int(args.max_legal_actions) <= 0:
        raise RuntimeError("--max-legal-actions must be > 0.")
    os.environ["POLYVISION_MAX_LEGAL_ACTIONS"] = str(int(args.max_legal_actions))
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    run_dir = os.path.join("runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    wandb_run = None
    if args.track:
        import wandb
        import os
        import sys
        wandb_run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=False,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
        # Force W&B-native charts to use global_step as x-axis.
        wandb_run.define_metric("global_step")
        wandb_run.define_metric("*", step_metric="global_step")
    writer = SummaryWriter(run_dir)

    def log_scalar(name, value, step):
        writer.add_scalar(name, value, step)
        if wandb_run is not None:
            wandb_run.log({"global_step": int(step), name: float(value)})

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

    agent = Agent(envs, actor_mode=args.actor_mode, max_legal_actions=args.max_legal_actions).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    selected_slots = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long, device=device)
    legal_global_ids_buf = torch.zeros(
        (args.num_steps, args.num_envs, args.max_legal_actions),
        dtype=torch.long,
        device=device,
    )
    legal_action_valid_mask_buf = torch.zeros(
        (args.num_steps, args.num_envs, args.max_legal_actions),
        dtype=torch.bool,
        device=device,
    )
    action_masks = None
    if args.actor_mode == "dense_debug":
        action_masks = torch.zeros(
            (args.num_steps, args.num_envs, envs.single_action_space.n),
            dtype=torch.bool,
            device=device,
        )

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, reset_infos = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    next_action_mask = None
    next_legal_global_ids = None
    next_legal_action_valid_mask = None
    if args.actor_mode == "dense_debug":
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
    else:
        next_legal_global_ids, next_legal_action_valid_mask = _extract_vector_legal_tensors(
            reset_infos,
            args.num_envs,
            int(args.max_legal_actions),
            envs.single_action_space.n,
            device,
        )
    if args.actor_mode == "dense_debug":
        if agent.actor[-1].out_features != envs.single_action_space.n:
            raise RuntimeError(
                f"ppo_policy_output_dim={agent.actor[-1].out_features} does not match env.action_space.n={envs.single_action_space.n}"
            )
    else:
        if agent.action_embedding is None:
            raise RuntimeError("legal_only mode requires action_embedding to be initialized.")
        if int(agent.action_embedding.num_embeddings) != int(envs.single_action_space.n):
            raise RuntimeError(
                f"action_embedding.num_embeddings={agent.action_embedding.num_embeddings} "
                f"does not match env.action_space.n={envs.single_action_space.n}"
            )
    action_interface_meta = {
        "actor_mode": args.actor_mode,
        "catalog_version": None,
        "canonicalizer_version": None,
        "map_width": None,
        "map_height": None,
        "global_action_space_n": int(envs.single_action_space.n),
        "action_offset_table_hash": None,
        "max_legal_actions": int(args.max_legal_actions),
    }
    for k in ["catalog_version", "canonicalizer_version", "map_width", "map_height", "global_action_space_n", "action_offset_table_hash", "max_legal_actions"]:
        vals = _extract_vector_field(reset_infos, k, args.num_envs, default_value=None)
        first = next((v for v in vals if v is not None), None)
        action_interface_meta[k] = first
    if action_interface_meta.get("global_action_space_n") is not None:
        if int(action_interface_meta["global_action_space_n"]) != int(envs.single_action_space.n):
            raise RuntimeError(
                f"Reported global_action_space_n={action_interface_meta['global_action_space_n']} "
                f"does not match env action space n={envs.single_action_space.n}"
            )
    info_mode_values = _extract_vector_field(reset_infos, "info_mode", args.num_envs, default_value=None)
    detected_info_mode = next((str(v).strip().lower() for v in info_mode_values if v is not None), "fast")
    debug_chart_mode = detected_info_mode == "debug"
    print(
        "[ACTION_INTERFACE] "
        f"actor_mode={args.actor_mode} "
        f"info_mode={detected_info_mode} "
        f"catalog_version={action_interface_meta.get('catalog_version')} "
        f"action_space_n={envs.single_action_space.n} "
        f"max_legal_actions={args.max_legal_actions}"
    )
    writer.add_text("meta/action_interface", json.dumps(action_interface_meta, sort_keys=True, default=str))

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
            if args.actor_mode == "dense_debug":
                action_masks[step] = next_action_mask > 0
            else:
                legal_global_ids_buf[step] = next_legal_global_ids
                legal_action_valid_mask_buf[step] = next_legal_action_valid_mask

            # ALGO LOGIC: action logic
            with torch.no_grad():
                if args.actor_mode == "dense_debug":
                    action, action_slot, logprob, _, value = agent.get_action_and_value(
                        next_obs,
                        action_mask=next_action_mask,
                    )
                else:
                    action, action_slot, logprob, _, value = agent.get_action_and_value(
                        next_obs,
                        legal_global_ids=next_legal_global_ids,
                        legal_action_valid_mask=next_legal_action_valid_mask,
                    )
                    slot_valid = next_legal_action_valid_mask.gather(1, action_slot.unsqueeze(1)).squeeze(1)
                    if not bool(torch.all(slot_valid)):
                        raise RuntimeError("Illegal selected_slot sampled in rollout (invalid/padded slot).")
                    selected_global_from_slot = next_legal_global_ids.gather(1, action_slot.unsqueeze(1)).squeeze(1)
                    if not bool(torch.equal(selected_global_from_slot.long(), action.long())):
                        raise RuntimeError("selected_global_id mismatch with selected_slot mapping during rollout.")
                values[step] = value.flatten()
            actions[step] = action
            if action_slot is not None:
                selected_slots[step] = action_slot.long()
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
            if args.actor_mode == "dense_debug":
                next_action_mask = _extract_vector_action_mask(
                    infos,
                    args.num_envs,
                    envs.single_action_space.n,
                    device,
                )
            else:
                next_legal_global_ids, next_legal_action_valid_mask = _extract_vector_legal_tensors(
                    infos,
                    args.num_envs,
                    int(args.max_legal_actions),
                    envs.single_action_space.n,
                    device,
                )
                selected_global_values = _extract_vector_field(infos, "selected_global_id", args.num_envs, default_value=None)
                for env_i, v in enumerate(selected_global_values):
                    if v is None:
                        continue
                    if int(v) != int(action[env_i].item()):
                        raise RuntimeError(
                            f"Env selected_global_id mismatch at env={env_i}: env={int(v)} actor={int(action[env_i].item())}"
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
                diagnostic_series = [
                    ("unit_count", "charts/unit_count"),
                    ("stars", "charts/stars"),
                    ("reward", "charts/reward"),
                ]
                # Low-signal internals are debug-only to keep dashboards clean.
                if debug_chart_mode:
                    diagnostic_series.extend(
                        [
                            ("turn", "charts/turn"),
                            ("selected_global_id", "charts/selected_global_id"),
                            ("selected_raw_java_index", "charts/selected_raw_java_index"),
                        ]
                    )

                for key, chart_name in diagnostic_series:
                    vals = _extract_vector_field(infos, key, args.num_envs, default_value=None)
                    clean = [float(v) for v in vals if v is not None]
                    if len(clean) > 0:
                        log_scalar(chart_name, float(np.mean(clean)), global_step)

                # Custom Phase-1 telemetry:
                # Log SPT for envs that just ended (typically via truncation at turn horizon).
                forestry_t10_values = []
                organization_t10_values = []
                first_visible_turn_values = []
                second_city_turn_values = []
                for env_idx in range(args.num_envs):
                    if truncations[env_idx] or terminations[env_idx]:
                        spt_value = None
                        city_count_value = None
                        fog_cleared_total_value = None
                        avg_city_level_value = None
                        techs_researched_value = None
                        forestry_researched_value = None
                        organization_researched_value = None
                        first_visible_turn_value = None
                        second_city_turn_value = None

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
                        avg_city_level_value = _extract_done_metric("avg_city_level")
                        techs_researched_value = _extract_done_metric("techs_researched")
                        forestry_researched_value = _extract_done_metric("forestry_researched")
                        organization_researched_value = _extract_done_metric("organization_researched")
                        first_visible_turn_value = _extract_done_metric("turn_first_uncaptured_village_visible")
                        second_city_turn_value = _extract_done_metric("turn_second_city_captured")

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
                                if avg_city_level_value is None and "avg_city_level" in finfo:
                                    avg_city_level_value = finfo["avg_city_level"]
                                if techs_researched_value is None and "techs_researched" in finfo:
                                    techs_researched_value = finfo["techs_researched"]
                                if forestry_researched_value is None and "forestry_researched" in finfo:
                                    forestry_researched_value = finfo["forestry_researched"]
                                if organization_researched_value is None and "organization_researched" in finfo:
                                    organization_researched_value = finfo["organization_researched"]
                                if first_visible_turn_value is None and "turn_first_uncaptured_village_visible" in finfo:
                                    first_visible_turn_value = finfo["turn_first_uncaptured_village_visible"]
                                if second_city_turn_value is None and "turn_second_city_captured" in finfo:
                                    second_city_turn_value = finfo["turn_second_city_captured"]

                        if spt_value is not None:
                            log_scalar("charts/custom_spt_return", float(spt_value), global_step)
                            log_scalar("charts/episode_end_spt", float(spt_value), global_step)
                        if city_count_value is not None:
                            log_scalar("charts/episode_end_village_count_t10", float(city_count_value), global_step)
                        if fog_cleared_total_value is not None:
                            log_scalar("charts/episode_end_fog_tiles_cleared_t10", float(fog_cleared_total_value), global_step)
                        if avg_city_level_value is not None:
                            log_scalar("charts/episode_end_avg_city_level_t10", float(avg_city_level_value), global_step)
                        if techs_researched_value is not None:
                            log_scalar("research/episode_end_techs_researched_t10", float(techs_researched_value), global_step)
                        if forestry_researched_value is not None:
                            forestry_t10_values.append(float(forestry_researched_value))
                        if organization_researched_value is not None:
                            organization_t10_values.append(float(organization_researched_value))
                        if first_visible_turn_value is not None and float(first_visible_turn_value) >= 0:
                            first_visible_turn_values.append(float(first_visible_turn_value))
                        if second_city_turn_value is not None and float(second_city_turn_value) >= 0:
                            second_city_turn_values.append(float(second_city_turn_value))
                if forestry_t10_values:
                    log_scalar(
                        "research/episode_end_forestry_researched_t10_rate",
                        float(np.mean(forestry_t10_values)),
                        global_step,
                    )
                if organization_t10_values:
                    log_scalar(
                        "research/episode_end_organization_researched_t10_rate",
                        float(np.mean(organization_t10_values)),
                        global_step,
                    )
                if first_visible_turn_values:
                    log_scalar(
                        "charts/avg_turn_first_uncaptured_village_visible",
                        float(np.mean(first_visible_turn_values)),
                        global_step,
                    )
                if second_city_turn_values:
                    log_scalar(
                        "charts/avg_turn_second_city_captured",
                        float(np.mean(second_city_turn_values)),
                        global_step,
                    )

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        log_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        log_scalar("charts/episodic_length", info["episode"]["l"], global_step)

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
        b_action_masks = None
        b_selected_slots = selected_slots.reshape(-1)
        b_legal_global_ids = legal_global_ids_buf.reshape((-1, int(args.max_legal_actions)))
        b_legal_action_valid_mask = legal_action_valid_mask_buf.reshape((-1, int(args.max_legal_actions)))
        if args.actor_mode == "dense_debug":
            b_action_masks = action_masks.reshape((-1, envs.single_action_space.n))

        if args.actor_mode == "legal_only":
            with torch.no_grad():
                slot_valid = b_legal_action_valid_mask.gather(1, b_selected_slots.unsqueeze(1)).squeeze(1)
                if not bool(torch.all(slot_valid)):
                    raise RuntimeError("Pre-update invariant failed: selected_slot points to invalid slot.")
                slot_global_ids = b_legal_global_ids.gather(1, b_selected_slots.unsqueeze(1)).squeeze(1)
                if not bool(torch.equal(slot_global_ids.long(), b_actions.long().view(-1))):
                    raise RuntimeError("Pre-update invariant failed: selected_slot does not map to selected_global_id.")
                _, _, recomputed_old_logprob, _, _ = agent.get_action_and_value(
                    b_obs,
                    legal_global_ids=b_legal_global_ids,
                    legal_action_valid_mask=b_legal_action_valid_mask,
                    selected_slot=b_selected_slots,
                )
                max_abs_diff = float(torch.max(torch.abs(recomputed_old_logprob - b_logprobs)).item())
                if max_abs_diff > float(args.old_logprob_recompute_tol):
                    raise RuntimeError(
                        f"Pre-update old_logprob invariant failed: max_abs_diff={max_abs_diff:.8f} "
                        f"> tol={float(args.old_logprob_recompute_tol):.8f}"
                    )

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                if args.actor_mode == "dense_debug":
                    _, _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        b_obs[mb_inds],
                        action=b_actions.long()[mb_inds],
                        action_mask=b_action_masks[mb_inds],
                    )
                else:
                    mb_slots = b_selected_slots[mb_inds]
                    mb_legal_ids = b_legal_global_ids[mb_inds]
                    mb_legal_valid = b_legal_action_valid_mask[mb_inds]
                    slot_global_ids = mb_legal_ids.gather(1, mb_slots.unsqueeze(1)).squeeze(1)
                    if not bool(torch.equal(slot_global_ids.long(), b_actions.long().view(-1)[mb_inds])):
                        raise RuntimeError("Minibatch invariant failed: legal_global_ids_padded[selected_slot] != selected_global_id.")
                    _, _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        b_obs[mb_inds],
                        legal_global_ids=mb_legal_ids,
                        legal_action_valid_mask=mb_legal_valid,
                        selected_slot=mb_slots,
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
        log_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        log_scalar("losses/value_loss", v_loss.item(), global_step)
        log_scalar("losses/policy_loss", pg_loss.item(), global_step)
        log_scalar("losses/entropy", entropy_loss.item(), global_step)
        log_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        log_scalar("losses/approx_kl", approx_kl.item(), global_step)
        log_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        log_scalar("losses/explained_variance", explained_var, global_step)
        if args.enable_step_diagnostics and iter_valid_actions_count > 0:
            log_scalar(
                "charts/mean_valid_actions",
                iter_valid_actions_sum / iter_valid_actions_count,
                global_step,
            )
        if args.enable_step_diagnostics and iter_action_count > 0:
            log_scalar(
                "charts/non_endturn_rate",
                iter_non_endturn_count / iter_action_count,
                global_step,
            )
        if args.enable_step_diagnostics and iter_delta_spt_count > 0:
            log_scalar(
                "charts/mean_delta_spt",
                iter_delta_spt_sum / iter_delta_spt_count,
                global_step,
            )
        print("SPS:", int(global_step / (time.time() - start_time)))
        log_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

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
