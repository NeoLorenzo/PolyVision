# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import json
import hashlib
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
    step_diagnostics_log_every: int = 3
    """log high-frequency step diagnostics every N environment steps (1 = every step)"""
    validate_action_interface: bool = True
    """if toggled, run strict pre-training action-interface validation and fail on any issue"""
    validation_states: int = 10000
    """number of decision states to validate before training starts"""
    validation_seed: int = 12345
    """seed for pre-training action-interface validation"""
    validation_cache_enabled: bool = True
    """reuse a cached successful action-interface validation when interface code/config is unchanged"""
    force_revalidate_action_interface: bool = False
    """force running action-interface validation even if a matching cached result exists"""
    max_illegal_sample_rate: float = 0.0001
    """abort training if illegal_sample_rate exceeds this threshold (0.0001 = 0.01%)"""
    max_fallback_end_turn_rate: float = 0.0001
    """abort training if fallback_end_turn_rate exceeds this threshold (0.0001 = 0.01%)"""
    actor_mode: str = "legal_only"
    """policy actor mode: legal_only (default), legal_features, or dense_debug"""
    max_legal_actions: int = 1024
    """fixed legal-action slot tensor length for legal_only actor mode"""
    legal_action_feature_dim: int = int(getattr(TribesGymWrapper, "ACTION_FEATURE_DIM", 22))
    """per-legal-slot feature width for legal_features actor mode"""
    old_logprob_recompute_tol: float = 1e-5
    """absolute tolerance for pre-update old_logprob recomputation invariant"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

def _env_flag_bool(key: str, default: bool = False) -> bool:
    raw = os.environ.get(key, None)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _env_flag_int(key: str, default: int) -> int:
    raw = os.environ.get(key, None)
    if raw is None:
        return int(default)
    try:
        return int(str(raw).strip())
    except Exception:
        return int(default)


class _TimerStats:
    def __init__(self):
        self.total_s = 0.0
        self.count = 0
        self.samples_s = []

    def add(self, duration_s: float):
        if duration_s is None:
            return
        d = float(duration_s)
        if d < 0:
            return
        self.total_s += d
        self.count += 1
        self.samples_s.append(d)

    def summary_ms(self):
        if self.count <= 0:
            return {
                "count": 0,
                "total_ms": 0.0,
                "mean_ms": 0.0,
                "median_ms": 0.0,
                "p95_ms": 0.0,
                "max_ms": 0.0,
            }
        arr = np.asarray(self.samples_s, dtype=np.float64) * 1000.0
        return {
            "count": int(self.count),
            "total_ms": float(self.total_s * 1000.0),
            "mean_ms": float(np.mean(arr)),
            "median_ms": float(np.median(arr)),
            "p95_ms": float(np.percentile(arr, 95)),
            "max_ms": float(np.max(arr)),
        }


class _ScalarStats:
    def __init__(self):
        self.total = 0.0
        self.count = 0
        self.min_v = None
        self.max_v = None
        self.samples = []

    def add(self, value):
        if value is None:
            return
        try:
            v = float(value)
        except Exception:
            return
        self.total += v
        self.count += 1
        self.samples.append(v)
        if self.min_v is None or v < self.min_v:
            self.min_v = v
        if self.max_v is None or v > self.max_v:
            self.max_v = v

    def summary(self):
        if self.count <= 0:
            return {
                "count": 0,
                "mean": 0.0,
                "median": 0.0,
                "p95": 0.0,
                "min": 0.0,
                "max": 0.0,
            }
        arr = np.asarray(self.samples, dtype=np.float64)
        return {
            "count": int(self.count),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
            "min": float(self.min_v),
            "max": float(self.max_v),
        }


class SPSProfiler:
    def __init__(self, enabled: bool, every_n_env_steps: int, write_json: bool, output_dir: str):
        self.enabled = bool(enabled)
        self.every_n_env_steps = max(1, int(every_n_env_steps))
        self.write_json = bool(write_json)
        self.output_dir = str(output_dir)
        self.start_perf = time.perf_counter()
        self.last_report_env_step = 0
        self.timers = {}
        self.scalars = {}
        self.env_timer_key_map = {
            "profile_env_step_total_s": "env/step_total",
            "profile_env_step_pre_fast_forward_s": "env/pre_step_fast_forward",
            "profile_env_step_pre_legal_generation_s": "env/pre_legal_action_generation",
            "profile_env_step_action_decode_s": "env/action_decode_canonicalization",
            "profile_env_step_java_apply_s": "env/java_action_apply",
            "profile_env_step_post_fast_forward_s": "env/post_action_fast_forward",
            "profile_env_step_reward_calc_s": "env/reward_calculation",
            "profile_env_step_post_legal_generation_s": "env/post_legal_action_generation",
            "profile_env_step_slot_mask_build_s": "env/legal_action_padding_mask_build",
            "profile_env_step_feature_build_s": "env/legal_action_feature_build",
            "profile_env_step_info_build_s": "env/info_dict_build",
            "profile_env_step_diag_build_s": "env/step_diagnostics_build",
            "profile_env_step_obs_flatten_s": "env/observation_flatten",
            "profile_env_step_sanitize_info_s": "env/info_sanitize",
            "profile_env_step_info_build_scalar_s": "env/info_build_scalar_telemetry",
            "profile_env_step_info_build_diag_counter_s": "env/info_build_diag_counters",
            "profile_env_step_info_build_legal_summary_s": "env/info_build_legal_summary",
            "profile_env_step_info_build_large_payload_s": "env/info_build_large_payload",
            "profile_env_step_info_build_action_repr_s": "env/info_build_action_repr",
            "profile_env_step_info_build_terminal_episode_s": "env/info_build_terminal_episode",
            "profile_env_step_info_build_metric_packaging_s": "env/info_build_metric_packaging",
            "profile_env_step_feature_alloc_zero_fill_s": "env/feature_build_alloc_zero_fill",
            "profile_env_step_feature_mask_construct_s": "env/feature_build_mask_construct",
            "profile_env_step_feature_precompute_s": "env/feature_build_step_precompute",
            "profile_env_step_feature_loop_iteration_s": "env/feature_build_loop_iteration",
            "profile_env_step_feature_gid_decode_s": "env/feature_build_gid_decode",
            "profile_env_step_feature_canonical_lookup_s": "env/feature_build_canonical_lookup",
            "profile_env_step_feature_static_metadata_s": "env/feature_build_static_metadata",
            "profile_env_step_feature_dynamic_state_s": "env/feature_build_dynamic_state",
            "profile_env_step_feature_padding_write_s": "env/feature_build_padding_write",
            "profile_env_step_feature_action_repr_s": "env/feature_build_action_repr",
            "profile_env_step_feature_java_calls_s": "env/feature_build_java_calls",
            "profile_env_step_reward_spt_delta_s": "env/reward_spt_delta",
            "profile_env_step_reward_city_capture_delta_s": "env/reward_city_capture_delta",
            "profile_env_step_reward_fog_calc_s": "env/reward_fog_calculation",
            "profile_env_step_reward_village_reveal_s": "env/reward_village_reveal",
            "profile_env_step_reward_move_village_shaping_s": "env/reward_move_village_shaping",
            "profile_env_step_reward_tactical_diag_s": "env/reward_tactical_diagnostics",
            "profile_env_step_reward_resource_upgrade_checks_s": "env/reward_resource_upgrade_checks",
            "profile_env_step_reward_legal_action_scans_s": "env/reward_legal_action_scans",
            "profile_env_step_reward_board_scans_s": "env/reward_board_scans",
            "profile_env_step_reward_java_calls_s": "env/reward_java_calls",
            "profile_env_step_post_legal_java_fetch_s": "env/post_legal_java_fetch",
            "profile_env_step_post_legal_java_compute_bridge_s": "env/post_legal_java_compute_bridge",
            "profile_env_step_post_legal_java_list_materialize_s": "env/post_legal_java_list_materialize",
            "profile_env_step_post_legal_java_json_parse_s": "env/post_legal_java_json_parse",
            "profile_env_step_post_legal_action_filter_s": "env/post_legal_action_filter",
            "profile_env_step_post_legal_filter_allowed_type_s": "env/post_legal_filter_allowed_type",
            "profile_env_step_post_legal_filter_oob_move_s": "env/post_legal_filter_oob_move",
            "profile_env_step_post_legal_filter_resource_upgrade_s": "env/post_legal_filter_resource_upgrade",
            "profile_env_step_post_legal_filter_city_count_tactical_s": "env/post_legal_filter_city_count_tactical",
            "profile_env_step_post_legal_filter_capture_priority_s": "env/post_legal_filter_capture_priority",
            "profile_env_step_post_legal_filter_move_visible_village_s": "env/post_legal_filter_move_visible_village",
            "profile_env_step_post_legal_filter_closest_reduce_distance_s": "env/post_legal_filter_closest_reduce_distance",
            "profile_env_step_post_legal_filter_early_backtrack_s": "env/post_legal_filter_early_backtrack",
            "profile_env_step_post_legal_filter_board_city_village_scans_s": "env/post_legal_filter_board_city_village_scans",
            "profile_env_step_post_legal_canonicalize_s": "env/post_legal_canonicalize",
            "profile_env_step_post_legal_collision_checks_s": "env/post_legal_collision_checks",
            "profile_env_step_post_legal_id_list_build_s": "env/post_legal_id_list_build",
            "profile_env_step_post_legal_mask_build_s": "env/post_legal_mask_build",
            "profile_env_step_post_legal_diag_build_s": "env/post_legal_diag_build",
            "profile_env_step_post_legal_padding_mask_s": "env/post_legal_padding_mask",
            "profile_env_step_post_legal_feature_build_s": "env/post_legal_feature_build",
            "profile_env_step_post_legal_info_attach_s": "env/post_legal_info_attach",
            "profile_env_step_post_legal_terminal_path_s": "env/post_legal_terminal_path",
            "profile_env_step_java_apply_action_serialize_s": "env/java_apply_action_serialize",
            "profile_env_step_java_apply_call_s": "env/java_apply_call",
            "profile_env_step_java_apply_response_parse_s": "env/java_apply_response_parse",
            "profile_env_step_java_apply_obs_fetch_s": "env/java_apply_obs_fetch",
            "profile_env_step_java_apply_done_fetch_s": "env/java_apply_done_fetch",
            "profile_env_step_java_apply_scores_fetch_s": "env/java_apply_scores_fetch",
            "profile_env_step_java_apply_tick_fetch_s": "env/java_apply_tick_fetch",
            "profile_env_step_java_apply_active_fetch_s": "env/java_apply_active_fetch",
            "profile_env_step_java_apply_spt_compute_s": "env/java_apply_spt_compute",
            "profile_env_step_java_apply_active_check_pre_s": "env/java_apply_active_check_pre",
            "profile_env_step_java_apply_active_check_post_s": "env/java_apply_active_check_post",
            "profile_env_step_java_apply_solo_fast_forward_pre_s": "env/java_apply_solo_fast_forward_pre",
            "profile_env_step_java_apply_solo_fast_forward_post_s": "env/java_apply_solo_fast_forward_post",
            "profile_env_reset_total_s": "env/reset_total",
            "profile_env_reset_java_reset_s": "env/reset_java_reset",
            "profile_env_reset_opening_script_s": "env/reset_opening_script",
            "profile_env_reset_legal_generation_s": "env/reset_legal_action_generation",
            "profile_env_reset_slot_mask_build_s": "env/reset_legal_action_padding_mask_build",
            "profile_env_reset_feature_build_s": "env/reset_legal_action_feature_build",
            "profile_env_reset_info_build_s": "env/reset_info_dict_build",
            "profile_env_reset_obs_flatten_s": "env/reset_observation_flatten",
            "profile_env_reset_sanitize_info_s": "env/reset_info_sanitize",
        }
        self.env_scalar_key_map = {
            "profile_env_step_post_legal_raw_actions_count": "env/post_legal_raw_actions_count",
            "profile_env_step_post_legal_canonical_actions_count": "env/post_legal_canonical_actions_count",
            "profile_env_step_post_legal_allowed_after_base_count": "env/post_legal_allowed_after_base_count",
            "profile_env_step_post_legal_allowed_after_tactical_count": "env/post_legal_allowed_after_tactical_count",
            "profile_env_step_post_legal_allowed_final_count": "env/post_legal_allowed_final_count",
            "profile_env_step_post_legal_raw_action_total_chars": "env/post_legal_raw_action_total_chars",
            "raw_valid_actions": "env/raw_valid_actions",
            "valid_actions": "env/valid_actions",
        }

    def add(self, key: str, duration_s: float):
        if not self.enabled:
            return
        if key not in self.timers:
            self.timers[key] = _TimerStats()
        self.timers[key].add(duration_s)

    def add_scalar(self, key: str, value):
        if not self.enabled:
            return
        if key not in self.scalars:
            self.scalars[key] = _ScalarStats()
        self.scalars[key].add(value)

    def ingest_env_profile_infos(self, infos, num_envs: int):
        if not self.enabled or infos is None:
            return
        for info_key, timer_key in self.env_timer_key_map.items():
            vals = _extract_vector_field(infos, info_key, num_envs, default_value=None)
            for v in vals:
                if v is None:
                    continue
                try:
                    self.add(timer_key, float(v))
                except Exception:
                    continue
        for info_key, scalar_key in self.env_scalar_key_map.items():
            vals = _extract_vector_field(infos, info_key, num_envs, default_value=None)
            for v in vals:
                if v is None:
                    continue
                self.add_scalar(scalar_key, v)

    def maybe_report(self, global_step: int, num_envs: int):
        if not self.enabled:
            return
        env_step = int(global_step // max(1, int(num_envs)))
        if (env_step - self.last_report_env_step) < self.every_n_env_steps:
            return
        self.last_report_env_step = env_step
        elapsed = max(1e-9, time.perf_counter() - self.start_perf)
        sps = float(global_step) / elapsed
        env_total = self.timers.get("env/step_total")
        env_total_ms = env_total.summary_ms()["mean_ms"] if env_total is not None else 0.0
        env_step_wall = self.timers.get("trainer/envs_step_wall")
        env_step_wall_ms = env_step_wall.summary_ms()["mean_ms"] if env_step_wall is not None else 0.0
        print(
            "[SPS_PROFILE] "
            f"global_step={global_step} env_steps={env_step} sps={sps:.2f} "
            f"mean_env_step_wall_ms={env_step_wall_ms:.3f} "
            f"mean_env_internal_step_ms_per_env={env_total_ms:.3f}"
        )

    def build_summary(self, global_step: int):
        elapsed = max(1e-9, time.perf_counter() - self.start_perf)
        out = {
            "global_step": int(global_step),
            "total_wall_s": float(elapsed),
            "sps": float(global_step / elapsed),
            "timers": {},
            "scalars": {},
        }
        for k, v in sorted(self.timers.items()):
            out["timers"][k] = v.summary_ms()
        for k, v in sorted(self.scalars.items()):
            out["scalars"][k] = v.summary()
        wall_total = float(elapsed)
        for k, stats in out["timers"].items():
            stats["pct_of_wall"] = float((stats["total_ms"] / 1000.0) / wall_total * 100.0) if wall_total > 0 else 0.0
        env_total_ms = out["timers"].get("env/step_total", {}).get("total_ms", 0.0)
        if env_total_ms > 0:
            for k, stats in out["timers"].items():
                if k.startswith("env/"):
                    stats["pct_of_env_step_total"] = float(stats["total_ms"] / env_total_ms * 100.0)
        return out


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
    def __init__(
        self,
        envs,
        actor_mode: str = "legal_only",
        max_legal_actions: int = 1024,
        legal_action_feature_dim: int = int(getattr(TribesGymWrapper, "ACTION_FEATURE_DIM", 22)),
    ):
        super().__init__()
        self.actor_mode = str(actor_mode).strip().lower()
        if self.actor_mode not in ("legal_only", "legal_features", "dense_debug"):
            raise ValueError(f"Unsupported actor_mode={actor_mode}. Expected legal_only, legal_features, or dense_debug.")
        self.max_legal_actions = int(max_legal_actions)
        self.legal_action_feature_dim = int(legal_action_feature_dim)
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
            if self.actor_mode == "legal_features":
                self.action_feature_encoder = nn.Sequential(
                    layer_init(nn.Linear(self.legal_action_feature_dim, 32)),
                    nn.Tanh(),
                    layer_init(nn.Linear(32, 32)),
                    nn.Tanh(),
                )
                self.action_scorer = nn.Sequential(
                    layer_init(nn.Linear(64 + 64 + 32, 64)),
                    nn.Tanh(),
                    layer_init(nn.Linear(64, 1), std=0.01),
                )
            else:
                self.action_feature_encoder = None
                self.action_scorer = None

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(
        self,
        x,
        action=None,
        action_mask=None,
        legal_global_ids=None,
        legal_action_valid_mask=None,
        legal_action_features=None,
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
        legal_emb = self.action_embedding(legal_ids)
        if self.actor_mode == "legal_only":
            # Score only legal candidate IDs using state-action dot products.
            logits = torch.einsum("bd,bkd->bk", h, legal_emb)
        else:
            if legal_action_features is None:
                raise RuntimeError("legal_features actor_mode requires legal_action_features.")
            if legal_action_features.ndim != 3:
                raise RuntimeError("legal_action_features must be rank-3: [batch, max_legal_actions, feature_dim].")
            if legal_action_features.shape[0] != legal_ids.shape[0] or legal_action_features.shape[1] != legal_ids.shape[1]:
                raise RuntimeError(
                    f"legal_action_features shape mismatch: got={tuple(legal_action_features.shape)} "
                    f"expected_prefix=({legal_ids.shape[0]}, {legal_ids.shape[1]})"
                )
            if int(legal_action_features.shape[2]) != int(self.legal_action_feature_dim):
                raise RuntimeError(
                    f"legal_action_features last dim mismatch: got={legal_action_features.shape[2]} "
                    f"expected={self.legal_action_feature_dim}"
                )
            feat_emb = self.action_feature_encoder(legal_action_features.float())
            repeated_state = h.unsqueeze(1).expand(-1, legal_ids.shape[1], -1)
            logits = self.action_scorer(torch.cat([repeated_state, legal_emb, feat_emb], dim=-1)).squeeze(-1)
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

    if "legal_global_ids_padded" in infos and "legal_action_valid_mask" in infos:
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

    if "legal_global_ids" not in infos:
        raise RuntimeError("Infos missing legal_global_ids_padded/legal_action_valid_mask and legal_global_ids.")

    raw_ids = infos["legal_global_ids"]
    ids_mask = infos.get("_legal_global_ids", None)
    ids_out = np.zeros((num_envs, max_legal_actions), dtype=np.int64)
    valid_out = np.zeros((num_envs, max_legal_actions), dtype=np.bool_)

    def _copy_sparse_row(i, ids_like):
        ids_arr = np.asarray(ids_like, dtype=np.int64).reshape(-1)
        if ids_arr.shape[0] > max_legal_actions:
            raise RuntimeError(
                f"legal_action_count={ids_arr.shape[0]} exceeded max_legal_actions={max_legal_actions} at env={i}"
            )
        if ids_arr.size > 0:
            if np.unique(ids_arr).size != ids_arr.size:
                raise RuntimeError(f"Duplicate legal_global_ids detected at env={i}.")
            if np.any(ids_arr < 0) or np.any(ids_arr >= action_dim):
                raise RuntimeError(f"Out-of-range legal global ID detected at env={i}.")
            ids_out[i, : ids_arr.shape[0]] = ids_arr
            valid_out[i, : ids_arr.shape[0]] = True

    try:
        if ids_mask is None:
            if len(raw_ids) == num_envs:
                for i in range(num_envs):
                    _copy_sparse_row(i, raw_ids[i])
            else:
                for i in range(num_envs):
                    _copy_sparse_row(i, raw_ids)
        else:
            mask_arr = np.asarray(ids_mask, dtype=bool).reshape(-1)
            if len(mask_arr) != num_envs or len(raw_ids) != num_envs:
                raise RuntimeError("Invalid _legal_global_ids validity mask in vector infos.")
            for i in range(num_envs):
                if mask_arr[i]:
                    _copy_sparse_row(i, raw_ids[i])
    except TypeError:
        for i in range(num_envs):
            _copy_sparse_row(i, raw_ids)

    return (
        torch.tensor(ids_out, dtype=torch.long, device=device),
        torch.tensor(valid_out, dtype=torch.bool, device=device),
    )


def _extract_vector_legal_feature_tensors(infos, num_envs, max_legal_actions, feature_dim, device):
    if infos is None:
        raise RuntimeError("Missing infos payload; cannot extract legal action feature tensors.")
    feat_out = np.zeros((num_envs, max_legal_actions, feature_dim), dtype=np.float32)

    if "legal_action_features_padded" in infos:
        raw_features = np.asarray(infos["legal_action_features_padded"])
        features_mask = infos.get("_legal_action_features_padded", None)

        def _copy_dense_row(i, src_feat):
            feat_row = np.asarray(src_feat, dtype=np.float32)
            if feat_row.ndim != 2:
                raise RuntimeError(f"legal feature tensor must be rank-2 at env={i}; got shape={feat_row.shape}")
            if feat_row.shape[0] != max_legal_actions or feat_row.shape[1] != feature_dim:
                raise RuntimeError(
                    f"legal feature shape mismatch at env={i}: got={feat_row.shape}, expected=({max_legal_actions}, {feature_dim})"
                )
            if not np.all(np.isfinite(feat_row)):
                raise RuntimeError(f"Non-finite values in legal_action_features_padded at env={i}.")
            feat_out[i] = feat_row

        if raw_features.ndim == 3 and raw_features.shape[0] == num_envs:
            for i in range(num_envs):
                _copy_dense_row(i, raw_features[i])
        else:
            if features_mask is None:
                raise RuntimeError(f"Cannot align legal feature tensors: features_shape={raw_features.shape}")
            mask_arr = np.asarray(features_mask, dtype=bool).reshape(-1)
            if mask_arr.shape[0] != num_envs:
                raise RuntimeError("Invalid _legal_action_features_padded validity mask in vector infos.")
            for i in range(num_envs):
                if mask_arr[i]:
                    _copy_dense_row(i, raw_features[i])
                else:
                    raise RuntimeError(f"Missing legal feature tensor row for env={i} in legal_features actor mode.")
        return torch.tensor(feat_out, dtype=torch.float32, device=device)

    if "legal_action_features" not in infos:
        raise RuntimeError("Infos missing legal_action_features_padded and legal_action_features for legal_features actor.")

    raw_features = infos["legal_action_features"]
    features_mask = infos.get("_legal_action_features", None)

    def _copy_sparse_row(i, src_feat):
        feat_row = np.asarray(src_feat, dtype=np.float32)
        if feat_row.ndim == 1 and feat_row.size == 0:
            return
        if feat_row.ndim != 2:
            raise RuntimeError(f"legal_action_features must be rank-2 at env={i}; got shape={feat_row.shape}")
        if feat_row.shape[1] != feature_dim:
            raise RuntimeError(
                f"legal_action_features width mismatch at env={i}: got={feat_row.shape[1]}, expected={feature_dim}"
            )
        if feat_row.shape[0] > max_legal_actions:
            raise RuntimeError(
                f"legal_action_features rows exceeded max_legal_actions at env={i}: "
                f"{feat_row.shape[0]} > {max_legal_actions}"
            )
        if not np.all(np.isfinite(feat_row)):
            raise RuntimeError(f"Non-finite values in legal_action_features at env={i}.")
        feat_out[i, : feat_row.shape[0], :] = feat_row

    try:
        if features_mask is None:
            if len(raw_features) == num_envs:
                for i in range(num_envs):
                    _copy_sparse_row(i, raw_features[i])
            else:
                for i in range(num_envs):
                    _copy_sparse_row(i, raw_features)
        else:
            mask_arr = np.asarray(features_mask, dtype=bool).reshape(-1)
            if len(mask_arr) != num_envs or len(raw_features) != num_envs:
                raise RuntimeError("Invalid _legal_action_features validity mask in vector infos.")
            for i in range(num_envs):
                if mask_arr[i]:
                    _copy_sparse_row(i, raw_features[i])
    except TypeError:
        for i in range(num_envs):
            _copy_sparse_row(i, raw_features)
    return torch.tensor(feat_out, dtype=torch.float32, device=device)


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


def _hash_file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _build_action_validator_fingerprint(
    env_id: str,
    states: int,
    seed: int,
    actor_mode: str,
    max_legal_actions: int,
    legal_action_feature_dim: int,
):
    tracked_files = []
    tracked_files.append(os.path.abspath(__file__))
    tracked_files.append(os.path.join(_repo_root, "pol_env", "Tribes", "py", "register_env.py"))
    tracked_files.append(os.path.join(_repo_root, "pol_env", "Tribes", "src", "core", "game", "PythonEnv.java"))

    java_actions_root = os.path.join(_repo_root, "pol_env", "Tribes", "src", "core", "actions")
    if os.path.isdir(java_actions_root):
        for root, _dirs, files in os.walk(java_actions_root):
            for fn in files:
                if fn.lower().endswith(".java"):
                    tracked_files.append(os.path.join(root, fn))
    tracked_files = sorted(os.path.abspath(p) for p in tracked_files if os.path.isfile(p))

    file_hashes = {}
    for p in tracked_files:
        rel = os.path.relpath(p, _repo_root)
        try:
            file_hashes[rel] = _hash_file_sha256(p)
        except Exception as e:
            file_hashes[rel] = f"ERROR:{e}"

    payload = {
        "validator_version": "action_validator_cache_v1",
        "env_id": str(env_id),
        "states": int(states),
        "seed": int(seed),
        "actor_mode": str(actor_mode),
        "max_legal_actions": int(max_legal_actions),
        "legal_action_feature_dim": int(legal_action_feature_dim),
        "file_hashes": file_hashes,
    }
    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    fingerprint = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
    return fingerprint, payload


def _validate_action_interface(
    env_id: str,
    states: int,
    seed: int,
    actor_mode: str,
    max_legal_actions: int,
    legal_action_feature_dim: int,
    cache_enabled: bool = True,
    force_revalidate: bool = False,
):
    fingerprint, fingerprint_payload = _build_action_validator_fingerprint(
        env_id=env_id,
        states=states,
        seed=seed,
        actor_mode=actor_mode,
        max_legal_actions=max_legal_actions,
        legal_action_feature_dim=legal_action_feature_dim,
    )
    cache_dir = os.path.join(_repo_root, ".cache", "action_validator")
    cache_path = os.path.join(cache_dir, f"{fingerprint}.json")
    if bool(cache_enabled) and not bool(force_revalidate) and os.path.isfile(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if bool(cached.get("passed", False)):
                print(
                    f"[ACTION_VALIDATOR] Cache hit: reusing successful strict validation "
                    f"over {int(cached.get('checked_states', 0))} decision states."
                )
                print(
                    "[ACTION_VALIDATOR] Cached situation coverage seen:"
                    f" spawn_warrior={bool(cached.get('saw_spawn_warrior', False))},"
                    f" resource_animal={bool(cached.get('saw_resource_animal', False))},"
                    f" capture_village={bool(cached.get('saw_capture_village', False))},"
                    f" research_forestry={bool(cached.get('saw_research_forestry', False))},"
                    f" clear_forest={bool(cached.get('saw_clear_forest', False))}"
                )
                return
        except Exception:
            pass

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
        if bool(cache_enabled):
            try:
                os.makedirs(cache_dir, exist_ok=True)
                cache_record = {
                    "passed": True,
                    "checked_states": int(checked),
                    "saw_spawn_warrior": bool(saw_spawn_warrior),
                    "saw_resource_animal": bool(saw_resource_animal),
                    "saw_capture_village": bool(saw_capture_village),
                    "saw_research_forestry": bool(saw_research_forestry),
                    "saw_clear_forest": bool(saw_clear_forest),
                    "created_at_unix": float(time.time()),
                    "fingerprint": fingerprint,
                    "fingerprint_payload": fingerprint_payload,
                }
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(cache_record, f, indent=2, sort_keys=True)
            except Exception as e:
                print(f"[ACTION_VALIDATOR] WARNING: failed to write cache file {cache_path}: {e}")
    finally:
        env.close()


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.actor_mode = str(args.actor_mode).strip().lower()
    if args.actor_mode not in ("legal_only", "legal_features", "dense_debug"):
        raise RuntimeError(f"Unsupported --actor-mode {args.actor_mode}. Expected legal_only, legal_features, or dense_debug.")
    if int(args.max_legal_actions) <= 0:
        raise RuntimeError("--max-legal-actions must be > 0.")
    if int(args.legal_action_feature_dim) <= 0:
        raise RuntimeError("--legal-action-feature-dim must be > 0.")
    if int(args.step_diagnostics_log_every) <= 0:
        raise RuntimeError("--step-diagnostics-log-every must be > 0.")
    os.environ["POLYVISION_MAX_LEGAL_ACTIONS"] = str(int(args.max_legal_actions))
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    run_dir = os.path.join("runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    profile_enabled = _env_flag_bool("POLYVISION_PROFILE_SPS", default=False)
    profile_every_n_steps = max(1, _env_flag_int("POLYVISION_PROFILE_EVERY_N_STEPS", default=1000))
    profile_write_json = _env_flag_bool("POLYVISION_PROFILE_WRITE_JSON", default=True)
    profile_output_dir = os.environ.get("POLYVISION_PROFILE_OUTPUT_DIR", os.path.join("outputs", "sps_profiles"))
    os.environ["POLYVISION_PROFILE_SPS"] = "1" if profile_enabled else "0"
    os.environ["POLYVISION_PROFILE_EVERY_N_STEPS"] = str(profile_every_n_steps)
    profiler = SPSProfiler(
        enabled=profile_enabled,
        every_n_env_steps=profile_every_n_steps,
        write_json=profile_write_json,
        output_dir=profile_output_dir,
    )
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

    _LOG_SCALAR_BLOCKLIST = {"reward/terminal_spt_bonus"}

    def log_scalar(name, value, step):
        if name in _LOG_SCALAR_BLOCKLIST:
            return
        t0_log = time.perf_counter()
        writer.add_scalar(name, value, step)
        if wandb_run is not None:
            wandb_run.log({"global_step": int(step), name: float(value)})
        profiler.add("trainer/logging_scalar", time.perf_counter() - t0_log)

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
        _validate_action_interface(
            args.env_id,
            args.validation_states,
            args.validation_seed,
            args.actor_mode,
            int(args.max_legal_actions),
            int(args.legal_action_feature_dim),
            cache_enabled=bool(args.validation_cache_enabled),
            force_revalidate=bool(args.force_revalidate_action_interface),
        )

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

    agent = Agent(
        envs,
        actor_mode=args.actor_mode,
        max_legal_actions=args.max_legal_actions,
        legal_action_feature_dim=args.legal_action_feature_dim,
    ).to(device)
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
    legal_action_features_buf = None
    if args.actor_mode == "legal_features":
        legal_action_features_buf = torch.zeros(
            (args.num_steps, args.num_envs, args.max_legal_actions, args.legal_action_feature_dim),
            dtype=torch.float32,
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
    profiler.start_perf = time.perf_counter()
    start_time = time.time()
    t0_env_reset = time.perf_counter()
    next_obs, reset_infos = envs.reset(seed=args.seed)
    profiler.add("trainer/envs_reset_wall", time.perf_counter() - t0_env_reset)
    t0_reset_tensor = time.perf_counter()
    next_obs = torch.Tensor(next_obs).to(device)
    profiler.add("trainer/reset_tensor_conversion", time.perf_counter() - t0_reset_tensor)
    profiler.ingest_env_profile_infos(reset_infos, args.num_envs)
    next_done = torch.zeros(args.num_envs).to(device)
    next_action_mask = None
    next_legal_global_ids = None
    next_legal_action_valid_mask = None
    next_legal_action_features = None
    if args.actor_mode == "dense_debug":
        t0_extract_mask = time.perf_counter()
        next_action_mask = _extract_vector_action_mask(
            reset_infos,
            args.num_envs,
            envs.single_action_space.n,
            device,
        )
        profiler.add("trainer/reset_info_extract", time.perf_counter() - t0_extract_mask)
        if next_action_mask.shape[-1] != envs.single_action_space.n:
            raise RuntimeError(
                f"action_mask.shape[-1]={next_action_mask.shape[-1]} does not match action space n={envs.single_action_space.n}"
            )
    else:
        t0_extract_legal = time.perf_counter()
        next_legal_global_ids, next_legal_action_valid_mask = _extract_vector_legal_tensors(
            reset_infos,
            args.num_envs,
            int(args.max_legal_actions),
            envs.single_action_space.n,
            device,
        )
        profiler.add("trainer/reset_info_extract", time.perf_counter() - t0_extract_legal)
        if args.actor_mode == "legal_features":
            t0_extract_features = time.perf_counter()
            next_legal_action_features = _extract_vector_legal_feature_tensors(
                reset_infos,
                args.num_envs,
                int(args.max_legal_actions),
                int(args.legal_action_feature_dim),
                device,
            )
            profiler.add("trainer/reset_feature_extract", time.perf_counter() - t0_extract_features)
    if args.actor_mode == "dense_debug":
        if agent.actor[-1].out_features != envs.single_action_space.n:
            raise RuntimeError(
                f"ppo_policy_output_dim={agent.actor[-1].out_features} does not match env.action_space.n={envs.single_action_space.n}"
            )
    else:
        if agent.action_embedding is None:
            raise RuntimeError("legal action modes require action_embedding to be initialized.")
        if int(agent.action_embedding.num_embeddings) != int(envs.single_action_space.n):
            raise RuntimeError(
                f"action_embedding.num_embeddings={agent.action_embedding.num_embeddings} "
                f"does not match env.action_space.n={envs.single_action_space.n}"
            )
        if args.actor_mode == "legal_features":
            if agent.action_feature_encoder is None or agent.action_scorer is None:
                raise RuntimeError("legal_features mode requires action_feature_encoder and action_scorer.")
            if int(agent.legal_action_feature_dim) != int(args.legal_action_feature_dim):
                raise RuntimeError(
                    f"agent.legal_action_feature_dim={agent.legal_action_feature_dim} "
                    f"does not match args.legal_action_feature_dim={args.legal_action_feature_dim}"
                )
            feature_dim_values = _extract_vector_field(reset_infos, "legal_action_feature_dim", args.num_envs, default_value=None)
            reported_feature_dim = next((int(v) for v in feature_dim_values if v is not None), None)
            if reported_feature_dim is not None and reported_feature_dim != int(args.legal_action_feature_dim):
                raise RuntimeError(
                    f"Reported legal_action_feature_dim={reported_feature_dim} "
                    f"does not match --legal-action-feature-dim={int(args.legal_action_feature_dim)}"
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
        "legal_action_feature_dim": int(args.legal_action_feature_dim),
        "legal_action_feature_version": None,
    }
    for k in [
        "catalog_version",
        "canonicalizer_version",
        "map_width",
        "map_height",
        "global_action_space_n",
        "action_offset_table_hash",
        "max_legal_actions",
        "legal_action_feature_version",
    ]:
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
        t0_rollout_iter = time.perf_counter()
        # Per-iteration diagnostics for dashboard clarity.
        iter_valid_actions_sum = 0.0
        iter_valid_actions_count = 0
        iter_non_endturn_count = 0
        iter_action_count = 0
        iter_delta_spt_sum = 0.0
        iter_delta_spt_count = 0
        tm_missed_move_onto_visible_village_num = 0.0
        tm_move_onto_visible_village_available_den = 0.0
        tm_ignored_capture_num = 0.0
        tm_capture_available_den = 0.0
        tm_end_turn_with_capture_available_num = 0.0
        tm_end_turn_with_capture_available_den = 0.0
        tm_end_turn_with_level_up_num = 0.0
        tm_level_up_available_den = 0.0
        tm_missed_city_upgrade_completion_num = 0.0
        tm_completion_gather_available_den = 0.0
        tm_move_off_neutral_village_before_capture_num = 0.0
        tm_unit_on_neutral_village_capture_illegal_den = 0.0
        tm_end_turn_with_useful_move_num = 0.0
        tm_useful_move_available_den = 0.0

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
                if args.actor_mode == "legal_features":
                    legal_action_features_buf[step] = next_legal_action_features

            # ALGO LOGIC: action logic
            t0_policy = time.perf_counter()
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
                        legal_action_features=next_legal_action_features if args.actor_mode == "legal_features" else None,
                    )
                    slot_valid = next_legal_action_valid_mask.gather(1, action_slot.unsqueeze(1)).squeeze(1)
                    if not bool(torch.all(slot_valid)):
                        raise RuntimeError("Illegal selected_slot sampled in rollout (invalid/padded slot).")
                    selected_global_from_slot = next_legal_global_ids.gather(1, action_slot.unsqueeze(1)).squeeze(1)
                    if not bool(torch.equal(selected_global_from_slot.long(), action.long())):
                        raise RuntimeError("selected_global_id mismatch with selected_slot mapping during rollout.")
                values[step] = value.flatten()
            profiler.add("trainer/policy_forward_action_selection", time.perf_counter() - t0_policy)
            actions[step] = action
            if action_slot is not None:
                selected_slots[step] = action_slot.long()
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            t0_action_numpy = time.perf_counter()
            action_np = action.cpu().numpy()
            profiler.add("trainer/action_to_numpy", time.perf_counter() - t0_action_numpy)
            t0_env_step = time.perf_counter()
            next_obs, reward, terminations, truncations, infos = envs.step(action_np)
            profiler.add("trainer/envs_step_wall", time.perf_counter() - t0_env_step)
            profiler.ingest_env_profile_infos(infos, args.num_envs)
            next_done = np.logical_or(terminations, truncations)
            t0_step_tensor = time.perf_counter()
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
            profiler.add("trainer/step_tensor_conversion", time.perf_counter() - t0_step_tensor)
            if args.actor_mode == "dense_debug":
                t0_extract_mask = time.perf_counter()
                next_action_mask = _extract_vector_action_mask(
                    infos,
                    args.num_envs,
                    envs.single_action_space.n,
                    device,
                )
                profiler.add("trainer/step_info_extract", time.perf_counter() - t0_extract_mask)
            else:
                t0_extract_legal = time.perf_counter()
                next_legal_global_ids, next_legal_action_valid_mask = _extract_vector_legal_tensors(
                    infos,
                    args.num_envs,
                    int(args.max_legal_actions),
                    envs.single_action_space.n,
                    device,
                )
                profiler.add("trainer/step_info_extract", time.perf_counter() - t0_extract_legal)
                if args.actor_mode == "legal_features":
                    t0_extract_features = time.perf_counter()
                    next_legal_action_features = _extract_vector_legal_feature_tensors(
                        infos,
                        args.num_envs,
                        int(args.max_legal_actions),
                        int(args.legal_action_feature_dim),
                        device,
                    )
                    profiler.add("trainer/step_feature_extract", time.perf_counter() - t0_extract_features)
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

            # Tactical mistake counters (always-on). These are 0/1 per env step.
            tm_missed_move_onto_visible_village_num += sum(
                float(v) for v in _extract_vector_field(infos, "tm_missed_move_onto_visible_village_num", args.num_envs, default_value=0) if v is not None
            )
            tm_move_onto_visible_village_available_den += sum(
                float(v) for v in _extract_vector_field(infos, "tm_move_onto_visible_village_available_den", args.num_envs, default_value=0) if v is not None
            )
            tm_ignored_capture_num += sum(
                float(v) for v in _extract_vector_field(infos, "tm_ignored_capture_num", args.num_envs, default_value=0) if v is not None
            )
            tm_capture_available_den += sum(
                float(v) for v in _extract_vector_field(infos, "tm_capture_available_den", args.num_envs, default_value=0) if v is not None
            )
            tm_end_turn_with_capture_available_num += sum(
                float(v)
                for v in _extract_vector_field(
                    infos, "tm_end_turn_with_capture_available_num", args.num_envs, default_value=0
                )
                if v is not None
            )
            tm_end_turn_with_capture_available_den += sum(
                float(v)
                for v in _extract_vector_field(
                    infos, "tm_end_turn_with_capture_available_den", args.num_envs, default_value=0
                )
                if v is not None
            )
            tm_end_turn_with_level_up_num += sum(
                float(v) for v in _extract_vector_field(infos, "tm_end_turn_with_level_up_num", args.num_envs, default_value=0) if v is not None
            )
            tm_level_up_available_den += sum(
                float(v) for v in _extract_vector_field(infos, "tm_level_up_available_den", args.num_envs, default_value=0) if v is not None
            )
            tm_missed_city_upgrade_completion_num += sum(
                float(v) for v in _extract_vector_field(infos, "tm_missed_city_upgrade_completion_num", args.num_envs, default_value=0) if v is not None
            )
            tm_completion_gather_available_den += sum(
                float(v) for v in _extract_vector_field(infos, "tm_completion_gather_available_den", args.num_envs, default_value=0) if v is not None
            )
            tm_move_off_neutral_village_before_capture_num += sum(
                float(v) for v in _extract_vector_field(infos, "tm_move_off_neutral_village_before_capture_num", args.num_envs, default_value=0) if v is not None
            )
            tm_unit_on_neutral_village_capture_illegal_den += sum(
                float(v) for v in _extract_vector_field(infos, "tm_unit_on_neutral_village_capture_illegal_den", args.num_envs, default_value=0) if v is not None
            )
            tm_end_turn_with_useful_move_num += sum(
                float(v) for v in _extract_vector_field(infos, "tm_end_turn_with_useful_move_num", args.num_envs, default_value=0) if v is not None
            )
            tm_useful_move_available_den += sum(
                float(v) for v in _extract_vector_field(infos, "tm_useful_move_available_den", args.num_envs, default_value=0) if v is not None
            )

            # Always-on episode-end research telemetry (even when step diagnostics are disabled).
            if not args.enable_step_diagnostics:
                forestry_t10_values = []
                organization_t10_values = []
                techs_t10_values = []
                forestry_turn_values = []
                organization_turn_values = []
                done_episode_count = 0
                for env_idx in range(args.num_envs):
                    if not (truncations[env_idx] or terminations[env_idx]):
                        continue
                    done_episode_count += 1

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

                    forestry_researched_value = _extract_done_metric("forestry_researched")
                    organization_researched_value = _extract_done_metric("organization_researched")
                    techs_researched_value = _extract_done_metric("techs_researched")
                    forestry_turn_value = _extract_done_metric("turn_forestry_researched")
                    organization_turn_value = _extract_done_metric("turn_organization_researched")

                    if "final_info" in infos and len(infos["final_info"]) > env_idx:
                        finfo = infos["final_info"][env_idx]
                        if finfo is not None:
                            if forestry_researched_value is None and "forestry_researched" in finfo:
                                forestry_researched_value = finfo["forestry_researched"]
                            if organization_researched_value is None and "organization_researched" in finfo:
                                organization_researched_value = finfo["organization_researched"]
                            if techs_researched_value is None and "techs_researched" in finfo:
                                techs_researched_value = finfo["techs_researched"]
                            if forestry_turn_value is None and "turn_forestry_researched" in finfo:
                                forestry_turn_value = finfo["turn_forestry_researched"]
                            if organization_turn_value is None and "turn_organization_researched" in finfo:
                                organization_turn_value = finfo["turn_organization_researched"]

                    if forestry_researched_value is not None:
                        forestry_t10_values.append(float(forestry_researched_value))
                    if organization_researched_value is not None:
                        organization_t10_values.append(float(organization_researched_value))
                    if techs_researched_value is not None:
                        techs_t10_values.append(float(techs_researched_value))
                    if forestry_turn_value is not None and float(forestry_turn_value) >= 0:
                        forestry_turn_values.append(float(forestry_turn_value))
                    if organization_turn_value is not None and float(organization_turn_value) >= 0:
                        organization_turn_values.append(float(organization_turn_value))

                if done_episode_count > 0:
                    log_scalar(
                        "research/episode_end_techs_researched_t10",
                        float(np.mean(techs_t10_values)) if techs_t10_values else 0.0,
                        global_step,
                    )
                    log_scalar(
                        "research/episode_end_forestry_researched_t10_rate",
                        float(np.mean(forestry_t10_values)) if forestry_t10_values else 0.0,
                        global_step,
                    )
                    log_scalar(
                        "research/episode_end_organization_researched_t10_rate",
                        float(np.mean(organization_t10_values)) if organization_t10_values else 0.0,
                        global_step,
                    )
                    log_scalar(
                        "research/avg_turn_forestry_researched",
                        float(np.mean(forestry_turn_values)) if forestry_turn_values else 0.0,
                        global_step,
                    )
                    log_scalar(
                        "research/avg_turn_organization_researched",
                        float(np.mean(organization_turn_values)) if organization_turn_values else 0.0,
                        global_step,
                    )

            if args.enable_step_diagnostics:
                t0_step_diag = time.perf_counter()
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
                env_step_index = global_step // args.num_envs
                should_log_step_diag_scalars = (env_step_index % int(args.step_diagnostics_log_every)) == 0
                if should_log_step_diag_scalars:
                    for key, chart_name in diagnostic_series:
                        vals = _extract_vector_field(infos, key, args.num_envs, default_value=None)
                        clean = [float(v) for v in vals if v is not None]
                        if len(clean) > 0:
                            log_scalar(chart_name, float(np.mean(clean)), global_step)

                # Custom Phase-1 telemetry:
                # Log SPT for envs that just ended (typically via truncation at turn horizon).
                techs_t10_values = []
                forestry_t10_values = []
                organization_t10_values = []
                forestry_turn_values = []
                organization_turn_values = []
                first_visible_turn_values = []
                second_city_turn_values = []
                animals_harvested_t10_values = []
                fruit_harvested_t10_values = []
                lumber_huts_built_t10_values = []
                sawmills_built_t10_values = []
                forests_cleared_t10_values = []
                village_capture_pct_t10_values = []
                terminal_spt_bonus_values = []
                final_spt_over_10_values = []
                final_spt_over_15_values = []
                done_episode_count = 0
                for env_idx in range(args.num_envs):
                    if truncations[env_idx] or terminations[env_idx]:
                        done_episode_count += 1
                        spt_value = None
                        city_count_value = None
                        fog_cleared_total_value = None
                        avg_city_level_value = None
                        techs_researched_value = None
                        forestry_researched_value = None
                        organization_researched_value = None
                        first_visible_turn_value = None
                        second_city_turn_value = None
                        forestry_turn_value = None
                        organization_turn_value = None
                        animals_harvested_t10_value = None
                        fruit_harvested_t10_value = None
                        lumber_huts_built_t10_value = None
                        sawmills_built_t10_value = None
                        forests_cleared_t10_value = None
                        village_capture_pct_t10_value = None
                        terminal_spt_bonus_value = None

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
                        forestry_turn_value = _extract_done_metric("turn_forestry_researched")
                        organization_turn_value = _extract_done_metric("turn_organization_researched")
                        animals_harvested_t10_value = _extract_done_metric("animals_harvested_t10")
                        fruit_harvested_t10_value = _extract_done_metric("fruit_harvested_t10")
                        lumber_huts_built_t10_value = _extract_done_metric("lumber_huts_built_t10")
                        sawmills_built_t10_value = _extract_done_metric("sawmills_built_t10")
                        forests_cleared_t10_value = _extract_done_metric("forests_cleared_t10")
                        village_capture_pct_t10_value = _extract_done_metric("village_capture_pct_t10")
                        terminal_spt_bonus_value = _extract_done_metric("terminal_spt_bonus")

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
                                if forestry_turn_value is None and "turn_forestry_researched" in finfo:
                                    forestry_turn_value = finfo["turn_forestry_researched"]
                                if organization_turn_value is None and "turn_organization_researched" in finfo:
                                    organization_turn_value = finfo["turn_organization_researched"]
                                if animals_harvested_t10_value is None and "animals_harvested_t10" in finfo:
                                    animals_harvested_t10_value = finfo["animals_harvested_t10"]
                                if fruit_harvested_t10_value is None and "fruit_harvested_t10" in finfo:
                                    fruit_harvested_t10_value = finfo["fruit_harvested_t10"]
                                if lumber_huts_built_t10_value is None and "lumber_huts_built_t10" in finfo:
                                    lumber_huts_built_t10_value = finfo["lumber_huts_built_t10"]
                                if sawmills_built_t10_value is None and "sawmills_built_t10" in finfo:
                                    sawmills_built_t10_value = finfo["sawmills_built_t10"]
                                if forests_cleared_t10_value is None and "forests_cleared_t10" in finfo:
                                    forests_cleared_t10_value = finfo["forests_cleared_t10"]
                                if village_capture_pct_t10_value is None and "village_capture_pct_t10" in finfo:
                                    village_capture_pct_t10_value = finfo["village_capture_pct_t10"]
                                if terminal_spt_bonus_value is None and "terminal_spt_bonus" in finfo:
                                    terminal_spt_bonus_value = finfo["terminal_spt_bonus"]

                        if spt_value is not None:
                            log_scalar("charts/custom_spt_return", float(spt_value), global_step)
                            log_scalar("charts/episode_end_spt", float(spt_value), global_step)
                            log_scalar("charts/final_spt_t10", float(spt_value), global_step)
                            final_spt_over_10_values.append(max(0.0, float(spt_value) - 10.0))
                            final_spt_over_15_values.append(max(0.0, float(spt_value) - 15.0))
                        if city_count_value is not None:
                            log_scalar("charts/episode_end_village_count_t10", float(city_count_value), global_step)
                        if fog_cleared_total_value is not None:
                            log_scalar("charts/episode_end_fog_tiles_cleared_t10", float(fog_cleared_total_value), global_step)
                        if avg_city_level_value is not None:
                            log_scalar("charts/episode_end_avg_city_level_t10", float(avg_city_level_value), global_step)
                        if techs_researched_value is not None:
                            techs_t10_values.append(float(techs_researched_value))
                        if forestry_researched_value is not None:
                            forestry_t10_values.append(float(forestry_researched_value))
                        if organization_researched_value is not None:
                            organization_t10_values.append(float(organization_researched_value))
                        if first_visible_turn_value is not None and float(first_visible_turn_value) >= 0:
                            first_visible_turn_values.append(float(first_visible_turn_value))
                        if second_city_turn_value is not None and float(second_city_turn_value) >= 0:
                            second_city_turn_values.append(float(second_city_turn_value))
                        if forestry_turn_value is not None and float(forestry_turn_value) >= 0:
                            forestry_turn_values.append(float(forestry_turn_value))
                        if organization_turn_value is not None and float(organization_turn_value) >= 0:
                            organization_turn_values.append(float(organization_turn_value))
                        if animals_harvested_t10_value is not None:
                            animals_harvested_t10_values.append(float(animals_harvested_t10_value))
                        if fruit_harvested_t10_value is not None:
                            fruit_harvested_t10_values.append(float(fruit_harvested_t10_value))
                        if lumber_huts_built_t10_value is not None:
                            lumber_huts_built_t10_values.append(float(lumber_huts_built_t10_value))
                        if sawmills_built_t10_value is not None:
                            sawmills_built_t10_values.append(float(sawmills_built_t10_value))
                        if forests_cleared_t10_value is not None:
                            forests_cleared_t10_values.append(float(forests_cleared_t10_value))
                        if village_capture_pct_t10_value is not None:
                            village_capture_pct_t10_values.append(float(village_capture_pct_t10_value))
                        if terminal_spt_bonus_value is not None:
                            terminal_spt_bonus_values.append(float(terminal_spt_bonus_value))
                if done_episode_count > 0:
                    log_scalar(
                        "research/episode_end_techs_researched_t10",
                        float(np.mean(techs_t10_values)) if techs_t10_values else 0.0,
                        global_step,
                    )
                    log_scalar(
                        "research/episode_end_forestry_researched_t10_rate",
                        float(np.mean(forestry_t10_values)) if forestry_t10_values else 0.0,
                        global_step,
                    )
                    log_scalar(
                        "research/episode_end_organization_researched_t10_rate",
                        float(np.mean(organization_t10_values)) if organization_t10_values else 0.0,
                        global_step,
                    )
                    log_scalar(
                        "reward/terminal_spt_bonus",
                        float(np.mean(terminal_spt_bonus_values)) if terminal_spt_bonus_values else 0.0,
                        global_step,
                    )
                    log_scalar(
                        "charts/final_spt_over_10",
                        float(np.mean(final_spt_over_10_values)) if final_spt_over_10_values else 0.0,
                        global_step,
                    )
                    log_scalar(
                        "charts/final_spt_over_15",
                        float(np.mean(final_spt_over_15_values)) if final_spt_over_15_values else 0.0,
                        global_step,
                    )
                    log_scalar(
                        "research/avg_turn_forestry_researched",
                        float(np.mean(forestry_turn_values)) if forestry_turn_values else 0.0,
                        global_step,
                    )
                    log_scalar(
                        "research/avg_turn_organization_researched",
                        float(np.mean(organization_turn_values)) if organization_turn_values else 0.0,
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
                if animals_harvested_t10_values:
                    log_scalar(
                        "economy/episode_end_animals_harvested_t10",
                        float(np.mean(animals_harvested_t10_values)),
                        global_step,
                    )
                if fruit_harvested_t10_values:
                    log_scalar(
                        "economy/episode_end_fruit_harvested_t10",
                        float(np.mean(fruit_harvested_t10_values)),
                        global_step,
                    )
                if lumber_huts_built_t10_values:
                    log_scalar(
                        "economy/episode_end_lumber_huts_built_t10",
                        float(np.mean(lumber_huts_built_t10_values)),
                        global_step,
                    )
                if sawmills_built_t10_values:
                    log_scalar(
                        "economy/episode_end_sawmills_built_t10",
                        float(np.mean(sawmills_built_t10_values)),
                        global_step,
                    )
                if forests_cleared_t10_values:
                    log_scalar(
                        "economy/episode_end_forests_cleared_t10",
                        float(np.mean(forests_cleared_t10_values)),
                        global_step,
                    )
                if village_capture_pct_t10_values:
                    log_scalar(
                        "economy/episode_end_village_capture_pct_t10",
                        float(np.mean(village_capture_pct_t10_values)),
                        global_step,
                    )
                profiler.add("trainer/step_diagnostics", time.perf_counter() - t0_step_diag)

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
                            t0_ckpt = time.perf_counter()
                            torch.save(agent.state_dict(), checkpoint_path)
                            meta_path = checkpoint_path + ".action_interface.json"
                            try:
                                with open(meta_path, "w", encoding="utf-8") as f:
                                    json.dump(action_interface_meta, f, indent=2, sort_keys=True, default=str)
                            except Exception as e:
                                print(f"warning: failed to save action interface metadata: {e}")
                            print(f"checkpoint_saved={checkpoint_path}")
                            profiler.add("trainer/checkpoint_save", time.perf_counter() - t0_ckpt)

        profiler.add("trainer/rollout_collection", time.perf_counter() - t0_rollout_iter)

        # bootstrap value if not done
        t0_update_iter = time.perf_counter()
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
        b_legal_action_features = None
        if args.actor_mode == "legal_features":
            b_legal_action_features = legal_action_features_buf.reshape(
                (-1, int(args.max_legal_actions), int(args.legal_action_feature_dim))
            )
        if args.actor_mode == "dense_debug":
            b_action_masks = action_masks.reshape((-1, envs.single_action_space.n))

        if args.actor_mode in ("legal_only", "legal_features"):
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
                    legal_action_features=b_legal_action_features if args.actor_mode == "legal_features" else None,
                    selected_slot=b_selected_slots,
                )
                max_abs_diff = float(torch.max(torch.abs(recomputed_old_logprob - b_logprobs)).item())
                if max_abs_diff > float(args.old_logprob_recompute_tol):
                    raise RuntimeError(
                        f"Pre-update old_logprob invariant failed: max_abs_diff={max_abs_diff:.8f} "
                        f"> tol={float(args.old_logprob_recompute_tol):.8f}"
                    )

        # Optimizing the policy and value network
        t0_ppo_update = time.perf_counter()
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
                    mb_legal_features = b_legal_action_features[mb_inds] if args.actor_mode == "legal_features" else None
                    slot_global_ids = mb_legal_ids.gather(1, mb_slots.unsqueeze(1)).squeeze(1)
                    if not bool(torch.equal(slot_global_ids.long(), b_actions.long().view(-1)[mb_inds])):
                        raise RuntimeError("Minibatch invariant failed: legal_global_ids_padded[selected_slot] != selected_global_id.")
                    _, _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        b_obs[mb_inds],
                        legal_global_ids=mb_legal_ids,
                        legal_action_valid_mask=mb_legal_valid,
                        legal_action_features=mb_legal_features,
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
        profiler.add("trainer/ppo_update", time.perf_counter() - t0_ppo_update)

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
        log_scalar(
            "tactical_mistakes/missed_move_onto_visible_village_rate",
            (tm_missed_move_onto_visible_village_num / tm_move_onto_visible_village_available_den)
            if tm_move_onto_visible_village_available_den > 0
            else 0.0,
            global_step,
        )
        log_scalar(
            "tactical_mistakes/ignored_capture_rate",
            (tm_ignored_capture_num / tm_capture_available_den)
            if tm_capture_available_den > 0
            else 0.0,
            global_step,
        )
        log_scalar(
            "tactical_mistakes/end_turn_with_capture_available_rate",
            (tm_end_turn_with_capture_available_num / tm_end_turn_with_capture_available_den)
            if tm_end_turn_with_capture_available_den > 0
            else 0.0,
            global_step,
        )
        log_scalar(
            "tactical_mistakes/end_turn_with_level_up_available_rate",
            (tm_end_turn_with_level_up_num / tm_level_up_available_den)
            if tm_level_up_available_den > 0
            else 0.0,
            global_step,
        )
        log_scalar(
            "tactical_mistakes/missed_city_upgrade_completion_rate",
            (tm_missed_city_upgrade_completion_num / tm_completion_gather_available_den)
            if tm_completion_gather_available_den > 0
            else 0.0,
            global_step,
        )
        log_scalar(
            "tactical_mistakes/move_off_neutral_village_before_capture_rate",
            (tm_move_off_neutral_village_before_capture_num / tm_unit_on_neutral_village_capture_illegal_den)
            if tm_unit_on_neutral_village_capture_illegal_den > 0
            else 0.0,
            global_step,
        )
        log_scalar(
            "tactical_mistakes/end_turn_with_useful_move_available_rate",
            (tm_end_turn_with_useful_move_num / tm_useful_move_available_den)
            if tm_useful_move_available_den > 0
            else 0.0,
            global_step,
        )
        print("SPS:", int(global_step / (time.time() - start_time)))
        log_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        profiler.add("trainer/update_and_logging_iteration", time.perf_counter() - t0_update_iter)
        profiler.maybe_report(global_step, args.num_envs)

    if args.save_model:
        t0_final_save = time.perf_counter()
        model_path = args.model_path or os.path.join(run_dir, f"{args.exp_name}.cleanrl_model")
        torch.save(agent.state_dict(), model_path)
        meta_path = model_path + ".action_interface.json"
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(action_interface_meta, f, indent=2, sort_keys=True, default=str)
        except Exception as e:
            print(f"warning: failed to save action interface metadata: {e}")
        print(f"model_saved={model_path}")
        profiler.add("trainer/final_model_save", time.perf_counter() - t0_final_save)

    if profiler.enabled:
        profile_summary = profiler.build_summary(global_step)
        profile_summary["run_name"] = run_name
        profile_summary["run_dir"] = run_dir
        profile_summary["actor_mode"] = str(args.actor_mode)
        profile_summary["num_envs"] = int(args.num_envs)
        profile_summary["num_steps"] = int(args.num_steps)
        profile_summary["total_timesteps"] = int(args.total_timesteps)
        profile_summary["env_id"] = str(args.env_id)
        profile_summary["enable_step_diagnostics"] = bool(args.enable_step_diagnostics)
        profile_summary["wandb_track_enabled"] = bool(args.track)
        profile_summary["detected_info_mode"] = str(detected_info_mode)
        profile_summary["legacy_sps_formula"] = int(global_step / max(1e-9, (time.time() - start_time)))
        print(
            "[SPS_PROFILE] FINAL "
            f"sps={profile_summary['sps']:.2f} "
            f"global_step={profile_summary['global_step']} "
            f"wall_s={profile_summary['total_wall_s']:.3f}"
        )
        if profiler.write_json:
            try:
                os.makedirs(profiler.output_dir, exist_ok=True)
                out_name = f"sps_profile_{run_name}.json"
                out_path = os.path.join(profiler.output_dir, out_name)
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(profile_summary, f, indent=2, sort_keys=True, default=str)
                print(f"[SPS_PROFILE] wrote_json={out_path}")
            except Exception as e:
                print(f"[SPS_PROFILE] warning: failed to write json summary: {e}")

    envs.close()
    writer.close()
