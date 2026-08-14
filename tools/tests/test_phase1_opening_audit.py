import json
import os
import unittest
from pathlib import Path

from tools.audit_phase1_opening import PHASES, aggregate, classify, expectations, load_manifest_maps, unit_rows

REPO_ROOT=Path(__file__).resolve().parents[2]


def trace_row(name,success=True,attempted=True,exc=None,pre=None,post=None,note=None):
    return {"step_name":name,"success":success,"attempted":attempted,"exception_type":exc,
            "pre":pre,"post":post,"note":note}


class OpeningAuditPureTests(unittest.TestCase):
    def test_trace_phase_contract_and_spawn_classification(self):
        capital=[3,5]; one={"capital":capital,"units":[{"unit_id":2,"x":3,"y":5}]}
        trace=[trace_row(p) for p in PHASES]
        trace[6]=trace_row("turn1_second_warrior_spawn",False,False,pre=one,post=one,note="action_unavailable_or_unmatched")
        handoff={"owned_unit_count":1,"capital_coordinate":capital,"owned_units":[{"unit_id":2,"x":3,"y":5,"on_capital":True}]}
        outcome,flags=classify(trace,handoff)
        self.assertEqual(outcome,"ONE_UNIT_SPAWN_UNAVAILABLE")
        self.assertIn("ONE_UNIT_SPAWN_UNAVAILABLE",flags)
        self.assertFalse(expectations(trace,handoff)["capital_free_before_spawn"])

    def test_swallowed_exception_is_classified_without_reset_failure(self):
        trace=[trace_row(p) for p in PHASES]
        trace[3]=trace_row("turn0_starting_warrior_move",False,True,exc="RuntimeError",note="exception_swallowed")
        outcome,flags=classify(trace,{"owned_unit_count":1})
        self.assertEqual(outcome,"MOVE_EXCEPTION_SWALLOWED")
        self.assertIn("ONE_UNIT_MOVE_FAILED",flags)

    def test_unit_count_and_capital_occupancy(self):
        obs={"unit":{"2":{"tribeId":0,"type":0,"x":3,"y":5},"4":{"tribeId":0,"type":0,"x":2,"y":4},"9":{"tribeId":1,"type":0,"x":1,"y":1}}}
        units=unit_rows(obs,[3,5])
        self.assertEqual(len(units),2); self.assertTrue(units[0]["on_capital"]); self.assertFalse(units[1]["on_capital"])

    def test_manifest_pool_mapping(self):
        manifest,rows,_=load_manifest_maps("human_benchmark")
        self.assertEqual(len(rows),17); self.assertTrue(all(r["pool"]=="human_benchmark" for r in rows))
        self.assertEqual(manifest["pool_counts"]["train"],5000)

    def test_summary_aggregation(self):
        base={"trace":[],"expectations":{"turn0_move_executed":True,"turn1_move_executed":True,"second_warrior_spawn_executed":True,"capital_free_before_spawn":True},"outcome_flags":[]}
        rows=[{**base,"pool":"train","opening_outcome":"EXPECTED_TWO_UNIT_OPENING","handoff":{"owned_unit_count":2,"owned_units":[]}},
              {**base,"pool":"validation","opening_outcome":"ONE_UNIT_SPAWN_UNAVAILABLE","expectations":{**base["expectations"],"second_warrior_spawn_executed":False,"capital_free_before_spawn":False},"handoff":{"owned_unit_count":1,"owned_units":[{"believed_original_starting_unit":True,"on_capital":True}]}}]
        summary=aggregate(rows)
        self.assertEqual(summary["total_maps"],2); self.assertEqual(summary["overall"]["unit_count_distribution"],{"2":1,"1":1})

    def test_audit_source_has_no_policy_or_model_evaluation(self):
        source=(REPO_ROOT/"tools/audit_phase1_opening.py").read_text(encoding="utf-8")
        self.assertNotIn("from py_rl",source); self.assertNotIn("torch.load",source)
        self.assertIn('"runs_policy":False',source); self.assertIn('"writes_model_scores":False',source)

    def test_corrected_opening_is_masked_and_fail_closed(self):
        source=(REPO_ROOT/"pol_env/Tribes/py/register_env.py").read_text(encoding="utf-8")
        self.assertIn('excluded_destination=opening_capital',source)
        self.assertIn('expected legal second-warrior spawn after Turn-1 movement',source)
        self.assertIn('raise opening_error("turn2_handoff"',source)
        self.assertNotIn('note="exception_swallowed")\n                return local_obs',source)


class OpeningAuditLiveTests(unittest.TestCase):
    def test_recorder_none_preserves_handoff_behavior(self):
        os.environ["POLYVISION_SOLO_NO_OPPONENT_MODE"]="1"
        os.environ["POLYVISION_LEVEL_POOL_GLOB"]="levels/phase1_pool_bardur_real/human_benchmark/*.csv"
        from pol_env.Tribes.py.register_env import TribesGymWrapper
        path=str((REPO_ROOT/"pol_env/Tribes/levels/phase1_pool_bardur_real/human_benchmark/map_003696.csv").resolve())
        states=[]
        for instrumented in (False,True):
            env=TribesGymWrapper(); env._level_pool=[path]; env._level_pool_size=1
            if instrumented:
                class R:
                    def __init__(self): self.rows=[]
                    def record(self,row): self.rows.append(row)
                recorder=R(); env._opening_audit_recorder=recorder
            try:
                env.reset(seed=42); states.append(json.dumps(env.tribes_env._last_obs,sort_keys=True))
                if instrumented: self.assertEqual([r["step_name"] for r in recorder.rows],list(PHASES))
            finally: env.close()
        self.assertEqual(states[0],states[1])

    def test_map_003696_capital_regression_is_fixed(self):
        os.environ["POLYVISION_SOLO_NO_OPPONENT_MODE"]="1"
        from pol_env.Tribes.py.register_env import TribesGymWrapper
        path=str((REPO_ROOT/"pol_env/Tribes/levels/phase1_pool_bardur_real/human_benchmark/map_003696.csv").resolve())
        class R:
            def __init__(self): self.rows=[]
            def record(self,row): self.rows.append(row)
        recorder=R(); env=TribesGymWrapper(); env._level_pool=[path]; env._level_pool_size=1
        env._opening_audit_recorder=recorder
        try:
            _,info=env.reset(seed=42)
            steps={row["step_name"]:row for row in recorder.rows}
            self.assertEqual(steps["turn0_starting_warrior_move"]["post"]["units"][0]["x"],2)
            self.assertEqual(steps["turn0_starting_warrior_move"]["post"]["units"][0]["y"],4)
            turn1_unit=steps["turn1_starting_warrior_move"]["post"]["units"][0]
            self.assertNotEqual((turn1_unit["x"],turn1_unit["y"]),(3,5))
            self.assertTrue(steps["turn1_second_warrior_spawn"]["success"])
            final_units=steps["turn1_end_turn"]["post"]["units"]
            self.assertEqual(len(final_units),2)
            self.assertIn((3,5),{(u["x"],u["y"]) for u in final_units})
            self.assertEqual(info["phase1_opening_version"],"v2_guaranteed_two_unit")
        finally: env.close()


if __name__=="__main__": unittest.main()
