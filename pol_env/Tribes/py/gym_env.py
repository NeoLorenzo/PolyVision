import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from py4j.java_gateway import GatewayParameters, JavaGateway, launch_gateway


class TribesGymEnv:
    """Minimal Gym-like API for the Tribes Java engine via Py4J.

    Methods:
      - reset(level_file: str, seed: int, mode: str) -> observation(dict)
      - step(action_index: int) -> (observation, reward, done, info)
      - action_space_n -> int
      - close()
    """

    def __init__(self, classpath_out: str, json_jar: str, port: int = None) -> None:
        # Launch a JVM with the proper classpath if no external gateway is provided.
        if port is None:
            classpath = os.pathsep.join([classpath_out, json_jar])
            # Set working directory to the Tribes directory so Java can find terrainProbs.json
            tribes_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            port = launch_gateway(classpath=classpath, die_on_exit=True, cwd=tribes_dir)
        self._gateway = JavaGateway(gateway_parameters=GatewayParameters(port=port, auto_convert=True))
        self._jvm = self._gateway.jvm
        self._env = self._jvm.core.game.PythonEnv()
        self._solo_no_opponent_mode = str(
            os.environ.get("POLYVISION_SOLO_NO_OPPONENT_MODE", "0")
        ).strip().lower() in ("1", "true", "yes", "on")
        try:
            self._env.setSoloNoOpponentMode(bool(self._solo_no_opponent_mode))
        except Exception:
            # Backward-compatible fallback if Java bridge is older.
            pass
        self._profile_sps_enabled = str(
            os.environ.get("POLYVISION_PROFILE_SPS", "0")
        ).strip().lower() in ("1", "true", "yes", "on")
        self._batch_legal_action_fetch_enabled = str(
            os.environ.get("POLYVISION_BATCH_LEGAL_ACTION_FETCH", "0")
        ).strip().lower() in ("1", "true", "yes", "on")
        self._batch_legal_fetch_equiv_check_enabled = str(
            os.environ.get("POLYVISION_BATCH_LEGAL_FETCH_EQUIV_CHECK", "0")
        ).strip().lower() in ("1", "true", "yes", "on")
        try:
            self._batch_legal_fetch_equiv_check_every_n = max(
                1,
                int(str(os.environ.get("POLYVISION_BATCH_LEGAL_FETCH_EQUIV_CHECK_EVERY_N_STEPS", "50")).strip()),
            )
        except Exception:
            self._batch_legal_fetch_equiv_check_every_n = 50
        self._list_actions_call_count = 0
        self._last_obs = None
        self._last_spt = 0.0
        self._last_step_profile = {}
        self._last_list_actions_profile = {}

    def reset(self, level_file: str, seed: int = 42, mode: str = "SCORE") -> Dict[str, Any]:
        game_mode = getattr(self._jvm.core.Types.GAME_MODE, mode)
        self._env.initFromLevel(level_file, int(seed), game_mode)
        self._last_obs = json.loads(self._env.observationJson())
        self._last_spt = self._compute_spt_from_obs(self._last_obs, tribe_id=0)
        return self._last_obs

    @property
    def action_space_n(self) -> int:
        return int(self._env.actionCount())

    def step(self, action_index: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        profile = {}
        prev_spt = self._last_spt
        t_conv0 = time.perf_counter() if self._profile_sps_enabled else None
        java_action_index = int(action_index)
        if self._profile_sps_enabled:
            profile["java_action_serialize_s"] = float(time.perf_counter() - t_conv0)

        t_step0 = time.perf_counter() if self._profile_sps_enabled else None
        self._env.stepByIndex(java_action_index)
        if self._profile_sps_enabled:
            profile["java_step_call_s"] = float(time.perf_counter() - t_step0)

        t_obs_fetch0 = time.perf_counter() if self._profile_sps_enabled else None
        obs_json = self._env.observationJson()
        if self._profile_sps_enabled:
            profile["java_observation_fetch_s"] = float(time.perf_counter() - t_obs_fetch0)

        t_obs_parse0 = time.perf_counter() if self._profile_sps_enabled else None
        obs = json.loads(obs_json)
        if self._profile_sps_enabled:
            profile["java_response_parse_s"] = float(time.perf_counter() - t_obs_parse0)

        t_done0 = time.perf_counter() if self._profile_sps_enabled else None
        done = bool(self._env.isDone())
        if self._profile_sps_enabled:
            profile["java_done_fetch_s"] = float(time.perf_counter() - t_done0)

        t_spt0 = time.perf_counter() if self._profile_sps_enabled else None
        current_spt = self._compute_spt_from_obs(obs, tribe_id=0)
        if self._profile_sps_enabled:
            profile["java_spt_compute_s"] = float(time.perf_counter() - t_spt0)
        reward = float(current_spt - prev_spt)
        self._last_spt = current_spt

        t_scores0 = time.perf_counter() if self._profile_sps_enabled else None
        scores = list(self._env.getScores())
        if self._profile_sps_enabled:
            profile["java_scores_fetch_s"] = float(time.perf_counter() - t_scores0)
        tribe0_score = scores[0] if scores else 0

        t_tick0 = time.perf_counter() if self._profile_sps_enabled else None
        tick = int(self._env.getTick())
        if self._profile_sps_enabled:
            profile["java_tick_fetch_s"] = float(time.perf_counter() - t_tick0)

        t_active0 = time.perf_counter() if self._profile_sps_enabled else None
        active_tribe_id = int(self._env.getActiveTribeID())
        if self._profile_sps_enabled:
            profile["java_active_tribe_fetch_s"] = float(time.perf_counter() - t_active0)

        info = {
            "tick": tick,
            "activeTribeID": active_tribe_id,
            "scores": list(scores),
            "score": tribe0_score,
            "spt": current_spt,
            "delta_spt": float(current_spt - prev_spt),
        }
        self._last_obs = obs
        self._last_step_profile = profile
        return obs, reward, done, info

    def _compute_spt_from_obs(self, obs: Dict[str, Any], tribe_id: int = 0) -> float:
        city_map = obs.get("city", {})
        spt = 0.0
        if isinstance(city_map, dict):
            for city in city_map.values():
                if not isinstance(city, dict):
                    continue
                if int(city.get("tribeID", -1)) == int(tribe_id):
                    spt += float(city.get("production", 0))
        return spt

    def list_actions(self) -> List[dict]:
        self._list_actions_call_count += 1
        use_batch = bool(self._batch_legal_action_fetch_enabled)
        profile = {}
        if use_batch:
            parsed = self._list_actions_batch(profile if self._profile_sps_enabled else None)
        else:
            parsed = self._list_actions_legacy(profile if self._profile_sps_enabled else None)

        if (
            self._batch_legal_fetch_equiv_check_enabled
            and (self._list_actions_call_count % int(self._batch_legal_fetch_equiv_check_every_n) == 0)
        ):
            legacy_actions = self._list_actions_legacy(None)
            batch_actions = self._list_actions_batch(None)
            if legacy_actions != batch_actions:
                raise RuntimeError(
                    "BATCH_LEGAL_FETCH_EQUIV: raw action mismatch "
                    f"legacy_count={len(legacy_actions)} batch_count={len(batch_actions)}"
                )

        if self._profile_sps_enabled:
            self._last_list_actions_profile = profile
        return parsed

    def _list_actions_legacy(self, profile: Optional[Dict[str, Any]]) -> List[dict]:
        if not self._profile_sps_enabled and profile is None:
            return [json.loads(s) for s in list(self._env.listActionsJson())]

        p = profile if isinstance(profile, dict) else {}
        t_java0 = time.perf_counter() if self._profile_sps_enabled or profile is not None else None
        java_actions = self._env.listActionsJson()
        if t_java0 is not None:
            p["java_compute_bridge_s"] = float(time.perf_counter() - t_java0)

        t_list0 = time.perf_counter() if self._profile_sps_enabled or profile is not None else None
        action_strings = list(java_actions)
        if t_list0 is not None:
            p["python_list_materialize_s"] = float(time.perf_counter() - t_list0)

        t_parse0 = time.perf_counter() if self._profile_sps_enabled or profile is not None else None
        parsed = []
        total_chars = 0
        for s in action_strings:
            if isinstance(s, str):
                total_chars += len(s)
            parsed.append(json.loads(s))
        if t_parse0 is not None:
            p["python_json_parse_s"] = float(time.perf_counter() - t_parse0)
        p["raw_action_count"] = int(len(action_strings))
        p["raw_action_total_chars"] = int(total_chars)
        if "java_compute_bridge_s" in p and "python_list_materialize_s" in p and "python_json_parse_s" in p:
            p["list_actions_total_s"] = float(
                p["java_compute_bridge_s"] + p["python_list_materialize_s"] + p["python_json_parse_s"]
            )
        return parsed

    def _list_actions_batch(self, profile: Optional[Dict[str, Any]]) -> List[dict]:
        p = profile if isinstance(profile, dict) else {}
        t_java0 = time.perf_counter() if self._profile_sps_enabled or profile is not None else None
        batch_payload = self._env.listActionsJsonBatch()
        if t_java0 is not None:
            p["java_compute_bridge_s"] = float(time.perf_counter() - t_java0)

        # In batch mode there is no per-element bridge iteration/materialization.
        p["python_list_materialize_s"] = 0.0
        if isinstance(batch_payload, str):
            p["raw_action_total_chars"] = int(len(batch_payload))
        else:
            p["raw_action_total_chars"] = 0

        t_parse0 = time.perf_counter() if self._profile_sps_enabled or profile is not None else None
        parsed = json.loads(batch_payload) if isinstance(batch_payload, str) and batch_payload else []
        if t_parse0 is not None:
            p["python_json_parse_s"] = float(time.perf_counter() - t_parse0)
        p["raw_action_count"] = int(len(parsed)) if isinstance(parsed, list) else 0
        if "java_compute_bridge_s" in p and "python_json_parse_s" in p:
            p["list_actions_total_s"] = float(
                p["java_compute_bridge_s"] + p["python_list_materialize_s"] + p["python_json_parse_s"]
            )
        if not isinstance(parsed, list):
            raise RuntimeError(
                f"BATCH_LEGAL_FETCH: expected JSON array from listActionsJsonBatch(), got {type(parsed)}"
            )
        return parsed

    def get_observation(self, full_visibility: bool = False) -> Dict[str, Any]:
        """Fetch a fresh observation from Java.

        full_visibility=True is diagnostic-only and bypasses fog-of-war.
        """
        if full_visibility:
            return json.loads(self._env.observationJsonFull())
        return json.loads(self._env.observationJson())

    def render(self, mode: str = "ansi") -> Optional[Union[str, "Image.Image"]]:
        """Render the current state.

        - mode="human": print a compact textual view and return None
        - mode="ansi": return a string with the textual view
        - mode="rgb_image": return a Pillow Image (H, W, 3)
        - mode="java": open/update the Java Swing GUI viewer
        """
        if self._last_obs is None:
            self._last_obs = json.loads(self._env.observationJson())
        obs = self._last_obs

        board = obs.get("board", {})
        terrain = board.get("terrain")
        unit_id = board.get("unitID")
        city_id = board.get("cityID")
        resource = board.get("resource")
        size = len(terrain) if terrain else 0

        # Terrain key mapping (must match core.Types.TERRAIN keys)
        terr_to_char = {
            0: ".",  # PLAIN
            1: "s",  # SHALLOW_WATER
            2: "d",  # DEEP_WATER
            3: "m",  # MOUNTAIN
            4: "v",  # VILLAGE
            5: "c",  # CITY
            6: "f",  # FOREST
        }

        # Resource icon paths and cache
        icon_files = {
            0: "img/resource/fish2.png",
            1: "img/resource/fruit2.png",
            2: "img/resource/animal2.png",
            3: "img/resource/whale2.png",
            5: "img/resource/ore2.png",
            6: "img/resource/crops2.png",
            7: "img/resource/ruins2.png",
        }
        icon_cache = {}
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        try:
            scores_py = list(self._env.getScores())
        except Exception:
            scores_py = []
        header = f"tick={obs.get('tick', 0)} activeTribeID={obs.get('activeTribeID', -1)} scores={scores_py} actions={self._env.actionCount()}"
        acts = self.list_actions()
        show = ", ".join(a.get("repr", a.get("type", "?")) for a in acts[:6])
        lines: List[str] = []
        lines.append(header)
        lines.append(f"sample_actions=[{show}{', ...' if len(acts) > 6 else ''}]")

        if size:
            for i in range(size):
                row_chars = []
                ti = terrain[i]
                ui = unit_id[i]
                ci = city_id[i]
                for j in range(size):
                    ch = terr_to_char.get(int(ti[j]), ".")
                    has_city = int(ci[j]) != -1
                    has_unit = int(ui[j]) != 0
                    if has_city and has_unit:
                        ch = "X"
                    elif has_city:
                        ch = "C"
                    elif has_unit:
                        ch = "U"
                    row_chars.append(ch)
                lines.append("".join(row_chars))

        text = "\n".join(lines)
        if mode == "human":
            print(text)
            return None
        if mode == "ansi":
            return text

        if mode == "java":
            self._env.openGui()
            self._env.renderGui()
            return None

        if mode == "rgb_image" or mode == "rgb_array":
            try:
                from PIL import Image, ImageDraw  # type: ignore
            except Exception:
                raise RuntimeError("Pillow (PIL) is required for rgb_image rendering. Install pillow.")

            if not size:
                return None

            colors = {
                0: (210, 200, 180),
                1: (100, 170, 230),
                2: (60, 120, 200),
                3: (120, 120, 120),
                4: (230, 200, 80),
                5: (200, 80, 80),
                6: (60, 140, 80),
            }
            scale = max(8, 512 // max(1, size))
            img = Image.new("RGB", (size * scale, size * scale), (0, 0, 0))
            draw = ImageDraw.Draw(img)

            for i in range(size):
                for j in range(size):
                    col = colors.get(int(terrain[i][j]), (0, 0, 0))
                    x0, y0 = j * scale, i * scale
                    x1, y1 = x0 + scale, y0 + scale
                    draw.rectangle([x0, y0, x1, y1], fill=col)

                    if resource and i < len(resource) and j < len(resource[i]):
                        rkey = int(resource[i][j])
                        icon_rel = icon_files.get(rkey)
                        has_city = int(city_id[i][j]) != -1
                        has_unit = int(unit_id[i][j]) != 0
                        if icon_rel and not (has_city or has_unit):
                            try:
                                raw = icon_cache.get(("raw", rkey))
                                if raw is None:
                                    raw = Image.open(os.path.join(base_dir, icon_rel)).convert("RGBA")
                                    icon_cache[("raw", rkey)] = raw
                                margin = max(2, scale // 6)
                                size_px = max(6, scale - 2 * margin)
                                icon = icon_cache.get((rkey, size_px))
                                if icon is None:
                                    icon = raw.resize((size_px, size_px), Image.LANCZOS)
                                    icon_cache[(rkey, size_px)] = icon
                                px = x0 + (scale - icon.width) // 2
                                py = y0 + (scale - icon.height) // 2
                                img.paste(icon, (px, py), icon)
                            except Exception:
                                pass

                    uid = int(unit_id[i][j])
                    if uid != 0:
                        r = max(2, scale // 6)
                        cx, cy = x0 + scale // 2, y0 + scale // 2
                        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(0, 0, 0))
            if mode == "rgb_image":
                return img
            return img.getdata()

        raise ValueError(f"Unsupported render mode: {mode}")

    def close(self) -> None:
        try:
            self._gateway.shutdown()
        except Exception:
            pass


def make_default_env() -> TribesGymEnv:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(base_dir, "out")
    json_jar = os.path.join(base_dir, "lib", "json.jar")
    return TribesGymEnv(classpath_out=out_dir, json_jar=json_jar)
