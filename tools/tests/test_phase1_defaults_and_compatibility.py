import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import gymnasium as gym
import torch

from pol_env.Tribes.py.register_env import TribesGymWrapper
from pol_env.Tribes.py.environment_contract import (
    CheckpointCompatibilityError,
    environment_compatibility_metadata,
    read_checkpoint_metadata,
    validate_checkpoint_compatibility,
)
from py_rl.cleanrl.cleanrl.ppo import Agent, Args


REPO_ROOT = Path(__file__).resolve().parents[2]


class TestRewardDefaults(unittest.TestCase):
    def test_default_terminal_spt_reward_enabled(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            wrapper = object.__new__(TribesGymWrapper)
            enabled = wrapper._parse_bool_env(
                "POLYVISION_TERMINAL_SPT_REWARD_ENABLED",
                default=TribesGymWrapper.TERMINAL_SPT_REWARD_ENABLED_DEFAULT,
            )
            self.assertTrue(TribesGymWrapper.TERMINAL_SPT_REWARD_ENABLED_DEFAULT)
            self.assertTrue(enabled)

    def test_explicit_terminal_spt_reward_disabled(self):
        for disable_val in ("0", "false", "no", "off", "FALSE", "Off"):
            with self.subTest(disable_val=disable_val):
                with mock.patch.dict(os.environ, {"POLYVISION_TERMINAL_SPT_REWARD_ENABLED": disable_val}, clear=True):
                    wrapper = object.__new__(TribesGymWrapper)
                    enabled = wrapper._parse_bool_env(
                        "POLYVISION_TERMINAL_SPT_REWARD_ENABLED",
                        default=TribesGymWrapper.TERMINAL_SPT_REWARD_ENABLED_DEFAULT,
                    )
                    self.assertFalse(enabled)

    def test_explicit_terminal_spt_reward_enabled(self):
        for enable_val in ("1", "true", "yes", "on", "TRUE", "On"):
            with self.subTest(enable_val=enable_val):
                with mock.patch.dict(os.environ, {"POLYVISION_TERMINAL_SPT_REWARD_ENABLED": enable_val}, clear=True):
                    wrapper = object.__new__(TribesGymWrapper)
                    enabled = wrapper._parse_bool_env(
                        "POLYVISION_TERMINAL_SPT_REWARD_ENABLED",
                        default=TribesGymWrapper.TERMINAL_SPT_REWARD_ENABLED_DEFAULT,
                    )
                    self.assertTrue(enabled)

    def test_default_terminal_spt_weights(self):
        self.assertEqual(TribesGymWrapper.TERMINAL_SPT_BASE_WEIGHT_DEFAULT, 1.0)
        self.assertEqual(TribesGymWrapper.TERMINAL_SPT_OVER_10_WEIGHT_DEFAULT, 2.0)
        self.assertEqual(TribesGymWrapper.TERMINAL_SPT_OVER_15_WEIGHT_DEFAULT, 3.0)


class TestPPOActorDefaults(unittest.TestCase):
    def _make_dummy_env_adapter(self, action_n: int = 63913, obs_dim: int = 5335):
        return SimpleNamespace(
            single_observation_space=gym.spaces.Box(low=-1e5, high=1e5, shape=(obs_dim,)),
            single_action_space=gym.spaces.Discrete(action_n),
        )

    def test_args_default_actor_mode(self):
        args = Args()
        self.assertEqual(args.actor_mode, "legal_features")

    def test_args_explicit_actor_modes(self):
        args_legal_only = Args(actor_mode="legal_only")
        self.assertEqual(args_legal_only.actor_mode, "legal_only")

        args_dense_debug = Args(actor_mode="dense_debug")
        self.assertEqual(args_dense_debug.actor_mode, "dense_debug")

    def test_agent_default_actor_mode(self):
        adapter = self._make_dummy_env_adapter()
        agent = Agent(adapter)
        self.assertEqual(agent.actor_mode, "legal_features")

    def test_agent_explicit_actor_modes(self):
        adapter = self._make_dummy_env_adapter()
        agent_legal_only = Agent(adapter, actor_mode="legal_only")
        self.assertEqual(agent_legal_only.actor_mode, "legal_only")

        agent_dense_debug = Agent(adapter, actor_mode="dense_debug")
        self.assertEqual(agent_dense_debug.actor_mode, "dense_debug")


class TestCheckpointMetadataCompatibility(unittest.TestCase):
    def _make_dummy_env_adapter(self, action_n: int = 63913, obs_dim: int = 6424):
        return SimpleNamespace(
            single_observation_space=gym.spaces.Box(low=-1e5, high=1e5, shape=(obs_dim,)),
            single_action_space=gym.spaces.Discrete(action_n),
        )

    def test_frozen_reference_checkpoint_is_rejected_by_current_v5_environment(self):
        ckpt_path = (
            REPO_ROOT
            / "runs"
            / "Tribes-v0__Phase1-Scientific-Train-V3-Seed3-TerminalSPT__3__1787602198"
            / "model_checkpoint_16000000.cleanrl_model"
        )
        if not ckpt_path.is_file():
            self.skipTest(f"Frozen checkpoint not found at {ckpt_path}")

        meta = read_checkpoint_metadata(str(ckpt_path))
        self.assertEqual(meta.get("actor_mode"), "legal_features")
        self.assertEqual(int(meta.get("observation_dim")), 505)
        self.assertEqual(meta.get("phase1_environment_version"), "v3_corrected_turn_economy")

        # Current v5 environment metadata has observation_dim=6424 and phase1_environment_version=v5_human_information_parity
        wrapper = object.__new__(TribesGymWrapper)
        wrapper._catalog = SimpleNamespace(width=11, height=11)
        wrapper.observation_space = gym.spaces.Box(low=-1e5, high=1e5, shape=(6424,))
        wrapper.action_space = gym.spaces.Discrete(63913)
        wrapper._catalog_fingerprint = meta["action_catalog_fingerprint"]
        wrapper._max_legal_actions = 256
        wrapper.PHASE1_ENVIRONMENT_VERSION = "v5_human_information_parity"
        wrapper.LEGAL_ACTION_FEATURE_VERSION = "v1_4_parity_spatial_and_cost"
        wrapper.ACTION_FEATURE_DIM = 47
        wrapper.CATALOG_VERSION = meta["catalog_version"]
        wrapper.CANONICALIZER_VERSION = meta["canonicalizer_version"]
        wrapper.PHASE1_OPENING_VERSION = meta["phase1_opening_version"]

        env_meta = environment_compatibility_metadata(
            wrapper,
            actor_mode=meta["actor_mode"],
            max_legal_actions=meta["max_legal_actions"],
        )

        with self.assertRaises(CheckpointCompatibilityError) as ctx:
            validate_checkpoint_compatibility(meta, env_meta)
        err = str(ctx.exception)
        self.assertIn("observation_dim", err)
        self.assertIn("phase1_environment_version", err)

    def test_historical_legal_only_checkpoint_metadata_retains_legal_only(self):
        ckpt_path = (
            REPO_ROOT
            / "runs"
            / "Tribes-v0__ppo__1__1778158665"
            / "model_checkpoint_500000.cleanrl_model"
        )
        if not ckpt_path.is_file():
            self.skipTest(f"Historical checkpoint not found at {ckpt_path}")

        meta = read_checkpoint_metadata(str(ckpt_path))
        # The metadata in the sidecar MUST be legal_only, untouched by changing the CLI default
        self.assertEqual(meta.get("actor_mode"), "legal_only")


if __name__ == "__main__":
    unittest.main()
