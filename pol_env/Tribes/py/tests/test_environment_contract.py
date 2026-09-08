import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from pol_env.Tribes.py.environment_contract import (
    CheckpointCompatibilityError,
    MapGeometryError,
    compute_level_pool_identity,
    observation_layout,
    validate_checkpoint_compatibility,
    validate_fixed_square_geometry,
)
from pol_env.Tribes.py.register_env import TribesGymWrapper
from py_rl.cleanrl.cleanrl.ppo import _build_action_validator_fingerprint


REPO_ROOT = Path(__file__).resolve().parents[4]
TRIBES_ROOT = REPO_ROOT / "pol_env" / "Tribes"


def compatibility_metadata(width=11, height=11, **overrides):
    values = {
        "map_width": width,
        "map_height": height,
        "observation_dim": observation_layout(width, height).expected_obs_dim,
        "action_space_n": 63913 if (width, height) == (11, 11) else 89305,
        "action_catalog_fingerprint": f"catalog-{width}x{height}",
        "actor_mode": "legal_features",
        "legal_action_feature_version": "v1_4_parity_spatial_and_cost",
        "legal_action_feature_dim": 47,
        "catalog_version": "flat-v1",
        "canonicalizer_version": "flat-v1-structured",
        "phase1_opening_version": "v2_guaranteed_two_unit",
        "phase1_environment_version": "v5_human_information_parity",
        "max_legal_actions": 128,
    }
    values.update(overrides)
    return values


class ObservationLayoutTests(unittest.TestCase):
    def test_dimension_derived_layouts(self):
        layout_11 = observation_layout(11, 11)
        self.assertEqual(layout_11.expected_obs_dim, 6424)
        self.assertEqual(layout_11.terrain_start, 0)
        self.assertEqual(layout_11.terrain_end, 121)
        self.assertEqual(layout_11.unit_types_start, 121)
        self.assertEqual(layout_11.unit_types_end, 1573)
        self.assertEqual(layout_11.city_territory_start, 1573)
        self.assertEqual(layout_11.city_territory_end, 2662)
        self.assertEqual(layout_11.unit_home_city_start, 2662)
        self.assertEqual(layout_11.unit_home_city_end, 3751)
        self.assertEqual(layout_11.road_start, 3751)
        self.assertEqual(layout_11.road_end, 3872)
        self.assertEqual(layout_11.buildings_start, 3872)
        self.assertEqual(layout_11.buildings_end, 6171)
        self.assertEqual(layout_11.resource_start, 6171)
        self.assertEqual(layout_11.resource_end, 6292)
        self.assertEqual(layout_11.legacy_scalar_start, 6292)
        self.assertEqual(layout_11.legacy_scalar_end, 6298)
        self.assertEqual(layout_11.economy_scalar_start, 6298)
        self.assertEqual(layout_11.economy_scalar_end, 6310)
        self.assertEqual(layout_11.tech_vector_start, 6310)
        self.assertEqual(layout_11.tech_vector_end, 6334)
        self.assertEqual(layout_11.city_block_start, 6334)
        self.assertEqual(layout_11.city_block_end, 6424)


class GeometryContractTests(unittest.TestCase):
    def test_uniform_geometries_are_accepted(self):
        self.assertEqual(validate_fixed_square_geometry(12, 12, 12, 12), (12, 12))
        self.assertEqual(validate_fixed_square_geometry(11, 11, 11, 11), (11, 11))

    def test_mixed_geometries_fail_in_both_directions(self):
        for loaded, expected in [((11, 11), (12, 12)), ((12, 12), (11, 11))]:
            with self.subTest(loaded=loaded, expected=expected):
                with self.assertRaisesRegex(MapGeometryError, "Mixed-dimension level pools are unsupported"):
                    validate_fixed_square_geometry(*loaded, *expected)

    def test_rectangular_geometry_fails(self):
        with self.assertRaisesRegex(MapGeometryError, "Rectangular maps are unsupported"):
            validate_fixed_square_geometry(12, 11)

    def test_wrapper_loaded_observation_geometry_uses_same_contract(self):
        wrapper = object.__new__(TribesGymWrapper)
        wrapper._catalog = SimpleNamespace(width=12, height=12)
        obs = {"board": {"terrain": [[0] * 11 for _ in range(11)]}}
        dims = wrapper._board_dimensions_from_obs(obs)
        with self.assertRaisesRegex(MapGeometryError, "environment geometry: 12x12"):
            validate_fixed_square_geometry(*dims, wrapper._catalog.width, wrapper._catalog.height)


class BootstrapFailureTests(unittest.TestCase):
    def test_unexpected_bootstrap_failure_is_chained_and_fails_closed(self):
        bootstrap_failure = RuntimeError("distinctive reset bootstrap failure")
        tribes_env = SimpleNamespace(reset=mock.Mock(side_effect=bootstrap_failure))

        with mock.patch(
            "pol_env.Tribes.py.register_env.make_default_env", return_value=tribes_env
        ), mock.patch.object(
            TribesGymWrapper, "_resolve_level_pool", return_value=["bootstrap.csv"]
        ), mock.patch.object(
            TribesGymWrapper, "_resolve_map_profile", return_value="BARDUR"
        ), mock.patch.object(
            TribesGymWrapper, "_validate_level_file_is_square"
        ):
            with self.assertRaisesRegex(
                RuntimeError, "establishing the real PolyVision interface"
            ) as raised:
                TribesGymWrapper()

        self.assertIs(raised.exception.__cause__, bootstrap_failure)
        tribes_env.reset.assert_called_once()


class CheckpointCompatibilityTests(unittest.TestCase):
    def test_same_environment_is_accepted(self):
        meta = compatibility_metadata()
        validate_checkpoint_compatibility(meta, dict(meta))

    def test_geometry_mismatch_is_rejected(self):
        with self.assertRaisesRegex(CheckpointCompatibilityError, "map_width"):
            validate_checkpoint_compatibility(compatibility_metadata(12, 12), compatibility_metadata(11, 11))

    def test_catalog_feature_version_and_actor_mismatches_are_rejected(self):
        cases = {
            "action_space_n": 123,
            "action_catalog_fingerprint": "different-catalog",
            "legal_action_feature_version": "features-v2",
            "legal_action_feature_dim": 41,
            "actor_mode": "legal_only",
        }
        for field, value in cases.items():
            with self.subTest(field=field):
                with self.assertRaisesRegex(CheckpointCompatibilityError, field):
                    validate_checkpoint_compatibility(
                        compatibility_metadata(),
                        compatibility_metadata(**{field: value}),
                    )

    def test_missing_required_metadata_is_rejected(self):
        checkpoint = compatibility_metadata()
        checkpoint.pop("observation_dim")
        with self.assertRaisesRegex(CheckpointCompatibilityError, "metadata is insufficient"):
            validate_checkpoint_compatibility(checkpoint, compatibility_metadata())

    def test_historical_checkpoint_without_opening_version_is_rejected(self):
        checkpoint = compatibility_metadata()
        checkpoint.pop("phase1_opening_version")
        with self.assertRaisesRegex(CheckpointCompatibilityError, "phase1_opening_version"):
            validate_checkpoint_compatibility(checkpoint, compatibility_metadata())

    def test_historical_checkpoint_without_environment_version_is_rejected(self):
        checkpoint = compatibility_metadata()
        checkpoint.pop("phase1_environment_version")
        with self.assertRaisesRegex(CheckpointCompatibilityError, "phase1_environment_version"):
            validate_checkpoint_compatibility(checkpoint, compatibility_metadata())

    def test_opening_contract_mismatch_is_rejected(self):
        with self.assertRaisesRegex(CheckpointCompatibilityError, "phase1_opening_version"):
            validate_checkpoint_compatibility(
                compatibility_metadata(phase1_opening_version="v1_mixed_capital_regression"),
                compatibility_metadata(),
            )

    def test_environment_version_mismatch_is_rejected(self):
        with self.assertRaisesRegex(CheckpointCompatibilityError, "phase1_environment_version"):
            validate_checkpoint_compatibility(
                compatibility_metadata(phase1_environment_version="v2_broken_turn_economy"),
                compatibility_metadata(),
            )


class ValidatorFingerprintTests(unittest.TestCase):
    def _fingerprint(self, metadata, pool_identity, pool_files):
        return _build_action_validator_fingerprint(
            env_id="Tribes-v0",
            states=100,
            seed=7,
            actor_mode="legal_features",
            max_legal_actions=128,
            legal_action_feature_dim=42,
            environment_metadata=metadata,
            pool_identity=pool_identity,
            pool_files=pool_files,
        )[0]

    def test_geometry_changes_fingerprint(self):
        self.assertNotEqual(
            self._fingerprint(compatibility_metadata(11, 11), "pool", []),
            self._fingerprint(compatibility_metadata(12, 12), "pool", []),
        )

    def test_pool_identity_is_stable_and_content_sensitive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = root / "a.csv"
            b = root / "b.csv"
            a.write_bytes(b".:\n")
            b.write_bytes(b"f:\n")
            identity_1, files_1 = compute_level_pool_identity([str(b), str(a)], relative_to=str(root))
            identity_2, files_2 = compute_level_pool_identity([str(a), str(b)], relative_to=str(root))
            self.assertEqual((identity_1, files_1), (identity_2, files_2))

            other_identity, other_files = compute_level_pool_identity([str(a)], relative_to=str(root))
            self.assertNotEqual(
                self._fingerprint(compatibility_metadata(), identity_1, files_1),
                self._fingerprint(compatibility_metadata(), other_identity, other_files),
            )

            b.write_bytes(b"m:o\n")
            identity_3, files_3 = compute_level_pool_identity([str(a), str(b)], relative_to=str(root))
            self.assertNotEqual(identity_1, identity_3)
            self.assertNotEqual(
                self._fingerprint(compatibility_metadata(), identity_1, files_1),
                self._fingerprint(compatibility_metadata(), identity_3, files_3),
            )


class LevelPoolSelectionTests(unittest.TestCase):
    def test_default_selects_training_pool_only_and_explicit_overrides_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = root / "levels" / "phase1_pool_bardur_real" / "train"
            validation = root / "levels" / "phase1_pool_bardur_real" / "validation"
            train.mkdir(parents=True)
            validation.mkdir(parents=True)
            (train / "train.csv").write_text(".:\n", encoding="utf-8")
            (validation / "validation.csv").write_text(".:\n", encoding="utf-8")
            wrapper = object.__new__(TribesGymWrapper)
            with mock.patch.object(wrapper, "_tribes_root_dir", return_value=str(root)):
                with mock.patch.dict(os.environ, {}, clear=True):
                    self.assertEqual(
                        [Path(path).resolve() for path in wrapper._resolve_level_pool(wrapper.PHASE1_LEVEL_FILE)],
                        [(train / "train.csv").resolve()],
                    )
                with mock.patch.dict(
                    os.environ,
                    {"POLYVISION_LEVEL_POOL_GLOB": "levels/phase1_pool_bardur_real/validation/*.csv"},
                    clear=True,
                ):
                    self.assertEqual(
                        [Path(path).resolve() for path in wrapper._resolve_level_pool(wrapper.PHASE1_LEVEL_FILE)],
                        [(validation / "validation.csv").resolve()],
                    )

    def test_explicit_empty_override_never_falls_back_to_training(self):
        wrapper = object.__new__(TribesGymWrapper)
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(wrapper, "_tribes_root_dir", return_value=tmp):
                with mock.patch.dict(
                    os.environ,
                    {"POLYVISION_LEVEL_POOL_GLOB": "levels/phase1_pool_bardur_real/test/*.csv"},
                    clear=True,
                ):
                    with self.assertRaisesRegex(FileNotFoundError, "refusing to fall back"):
                        wrapper._resolve_level_pool(wrapper.PHASE1_LEVEL_FILE)


class LiveMixedPoolRegressionTests(unittest.TestCase):
    def _fixture_paths(self):
        synthetic = TRIBES_ROOT / "levels" / "phase1_pool_bardur_solo" / "phase1_12x12_pool_000.csv"
        genuine = TRIBES_ROOT / "levels" / "phase1_pool_bardur_real" / "train" / "map_000001.csv"
        if not synthetic.is_file() or not genuine.is_file():
            self.skipTest("Local 12x12 and 11x11 pool fixtures are required for the live regression test")
        return synthetic, genuine

    def _assert_second_reset_fails(self, first: Path, second: Path):
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "a.csv").write_bytes(first.read_bytes())
            (tmp_path / "b.csv").write_bytes(second.read_bytes())
            with mock.patch.dict(
                os.environ,
                {
                    "POLYVISION_LEVEL_POOL_GLOB": str(tmp_path / "*.csv"),
                    "POLYVISION_LEVEL_SELECTION_MODE": "round_robin",
                    "POLYVISION_SOLO_NO_OPPONENT_MODE": "1",
                },
                clear=False,
            ):
                wrapper = TribesGymWrapper()
                first_obs, _ = wrapper.reset(seed=42)
                self.assertTrue(wrapper.observation_space.contains(first_obs))
                with self.assertRaisesRegex(MapGeometryError, "Mixed-dimension level pools are unsupported"):
                    wrapper.reset()

    def test_12_then_11_fails_deliberately(self):
        synthetic, genuine = self._fixture_paths()
        self._assert_second_reset_fails(synthetic, genuine)

    def test_11_then_12_fails_deliberately(self):
        synthetic, genuine = self._fixture_paths()
        self._assert_second_reset_fails(genuine, synthetic)

    def test_rectangular_csv_fails_before_java_loading(self):
        synthetic, _ = self._fixture_paths()
        rows = synthetic.read_text(encoding="utf-8").splitlines()[:11]
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            rectangular = Path(tmp) / "rectangular.csv"
            rectangular.write_text("\n".join(rows) + "\n", encoding="utf-8")
            wrapper = object.__new__(TribesGymWrapper)
            with self.assertRaisesRegex(MapGeometryError, "Rectangular maps are unsupported"):
                wrapper._validate_level_file_is_square(str(rectangular))


if __name__ == "__main__":
    unittest.main()
