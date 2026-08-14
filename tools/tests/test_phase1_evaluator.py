import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.distributions import Categorical

from py_rl.cleanrl.cleanrl.ppo import Agent

from tools.phase1_eval_core import (
    ALL_POLICIES, PPOPolicy, RandomLegalPolicy, VisibleGreedyPolicy,
    aggregate_results, bootstrap_ci, build_schedule, load_verified_pool,
    paired_stats, validate_schedule,
)
from tools.evaluate_phase1 import REQUIRED_EPISODE_FIELDS

REPO_ROOT = Path(__file__).resolve().parents[2]


def info_fixture():
    feats = np.zeros((5, 42), dtype=np.float32)
    return {
        "legal_global_ids_padded": np.array([10, 20, 30, 0, 0]),
        "legal_action_valid_mask": np.array([1, 1, 1, 0, 0], dtype=bool),
        "legal_action_features_padded": feats,
        "city_count": 1,
    }


class FakeAgent:
    def __init__(self, logits): self.logits = torch.tensor([logits], dtype=torch.float32)
    def get_action_distribution(self, *args, **kwargs): return Categorical(logits=self.logits)


class Phase1PolicyTests(unittest.TestCase):
    def test_training_sampling_uses_same_authoritative_distribution(self):
        envs = SimpleNamespace(
            single_observation_space=SimpleNamespace(shape=(4,)),
            single_action_space=SimpleNamespace(n=32),
        )
        torch.manual_seed(2)
        agent = Agent(envs, actor_mode="legal_features", max_legal_actions=3, legal_action_feature_dim=42)
        x=torch.zeros((1,4)); ids=torch.tensor([[4,9,12]]); valid=torch.tensor([[1,1,1]],dtype=torch.bool)
        feats=torch.zeros((1,3,42))
        torch.manual_seed(99)
        expected=agent.get_action_distribution(x,legal_global_ids=ids,legal_action_valid_mask=valid,legal_action_features=feats).sample()
        torch.manual_seed(99)
        actual=agent.get_action_and_value(x,legal_global_ids=ids,legal_action_valid_mask=valid,legal_action_features=feats)[1]
        self.assertTrue(torch.equal(expected,actual))

    def test_argmax_chooses_highest_valid_slot_and_global_id(self):
        policy = PPOPolicy(FakeAgent([1, 9, 3, -1e8, -1e8]), torch.device("cpu"), False, None)
        gid, slot = policy.choose_action(np.zeros(4), info_fixture())
        self.assertEqual((gid, slot), (20, 1))

    def test_sampled_is_valid_and_reproducible(self):
        def sequence(seed):
            p = PPOPolicy(FakeAgent([1, 2, 3, -1e8, -1e8]), torch.device("cpu"), True, seed)
            return [p.choose_action(np.zeros(4), info_fixture())[0] for _ in range(20)]
        self.assertEqual(sequence(123), sequence(123))
        self.assertTrue(set(sequence(123)) <= {10, 20, 30})

    def test_random_legal_uses_exact_visible_candidate_set_and_seed(self):
        def sequence(seed):
            p = RandomLegalPolicy(seed)
            return [p.choose_action(None, info_fixture())[0] for _ in range(50)]
        self.assertEqual(sequence(7), sequence(7))
        self.assertEqual(set(sequence(7)), {10, 20, 30})

    def test_visible_greedy_warrior_uses_feature_not_private_state(self):
        info = info_fixture()
        info["legal_action_features_padded"][1, 14] = 1
        info["legal_action_features_padded"][1, 11] = 1
        gid, slot = VisibleGreedyPolicy().choose_action(np.zeros(4), info)
        self.assertEqual((gid, slot), (20, 1))


class Phase1ScheduleAndStatsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest, cls.maps, _ = load_verified_pool(REPO_ROOT, "validation")

    def test_full_validation_schedule_and_pairing(self):
        self.assertEqual(len(self.maps), 250)
        self.assertEqual(len({m.canonical_map_sha256 for m in self.maps}), 250)
        schedule = build_schedule(self.maps, ALL_POLICIES, 5, 42)
        self.assertEqual(len(schedule), 3000)
        validate_schedule(schedule, self.maps, ALL_POLICIES, 5)
        lookup = {(r.policy, r.map_identity, r.replicate_index): r for r in schedule}
        sample = self.maps[0].canonical_map_sha256
        self.assertEqual(lookup[("ppo_sampled", sample, 3)].episode_seed,
                         lookup[("random_legal", sample, 3)].episode_seed)
        pools = {r["pool"] for r in self.manifest["maps"] if r["canonical_map_sha256"] in {m.canonical_map_sha256 for m in self.maps}}
        self.assertEqual(pools, {"validation"})

    def test_statistics_map_aggregation_bootstrap_and_pairs(self):
        episodes=[]
        for policy, vals in {"ppo_sampled":{"a":[10,14],"b":[2,6]},"random_legal":{"a":[5,7],"b":[1,3]}}.items():
            for mid, xs in vals.items():
                for i,x in enumerate(xs): episodes.append({"policy":policy,"map_identity":mid,"map_filename":mid,"final_spt_t10":x})
        summary, per_map = aggregate_results(episodes, 9)
        self.assertEqual(summary["ppo_sampled"]["map_level"]["mean"], 8.0)
        self.assertEqual(bootstrap_ci([1,2,3], 10), bootstrap_ci([1,2,3], 10))
        pair=paired_stats(per_map,"ppo_sampled","random_legal",9)
        self.assertEqual((pair["wins"],pair["ties"],pair["losses"]),(2,0,0))
        self.assertEqual(pair["mean_difference"],4.0)

    def test_schedule_policy_seed_is_recorded(self):
        rows=build_schedule(self.maps[:1],["ppo_sampled"],2,5)
        self.assertTrue(all(isinstance(r.policy_seed,int) for r in rows))
        self.assertNotEqual(rows[0].policy_seed,rows[1].policy_seed)

    def test_required_episode_serialization_schema(self):
        row = {key: None for key in REQUIRED_EPISODE_FIELDS}
        row.update({"contract_valid": True, "map_path": r"C:\repo\validation\map.csv"})
        encoded = json.dumps(row, sort_keys=True)
        row = json.loads(encoded)
        self.assertFalse(REQUIRED_EPISODE_FIELDS - set(row))
        self.assertTrue(row["contract_valid"])
        self.assertEqual(row["map_path"].lower().count("\\validation\\"), 1)


if __name__ == "__main__": unittest.main()
