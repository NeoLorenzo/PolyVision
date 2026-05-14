import argparse
import json
import os
import random
import sys
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch

# repo import setup (matches ppo.py style)
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

try:
    import pol_env.Tribes.py.register_env as register_env  # noqa: F401
except Exception:
    pass

from py_rl.cleanrl.cleanrl.ppo import Agent, _extract_vector_legal_feature_tensors, _extract_vector_legal_tensors


@dataclass
class MoveQualityStats:
    move_states: int = 0
    chosen_is_move: int = 0
    chosen_above_avg_reveal: int = 0
    chosen_above_avg_adj_delta: int = 0
    chosen_better_village_delta: int = 0
    chosen_backtrack: int = 0
    avg_legal_moves_per_state: float = 0.0
    reveal_margin_mean: float = 0.0
    adj_delta_margin_mean: float = 0.0
    village_delta_margin_mean: float = 0.0


def make_sync_env(seed: int):
    def thunk():
        return gym.make("Tribes-v0")

    envs = gym.vector.SyncVectorEnv([thunk])
    obs, infos = envs.reset(seed=seed)
    return envs, obs, infos


def _infer_feature_dim(infos, envs):
    fd_raw = infos.get("legal_action_feature_dim", None)
    if isinstance(fd_raw, (list, tuple, np.ndarray)):
        vals = np.asarray(fd_raw).reshape(-1)
        for v in vals:
            try:
                return int(v)
            except Exception:
                continue
    elif fd_raw is not None:
        try:
            return int(fd_raw)
        except Exception:
            pass
    try:
        return int(envs.envs[0].unwrapped.ACTION_FEATURE_DIM)
    except Exception:
        return 22


def _extract_tensors(infos, envs, device, max_legal_actions=1024, feature_dim=None):
    if feature_dim is None:
        feature_dim = _infer_feature_dim(infos, envs)
    ids, valid = _extract_vector_legal_tensors(
        infos,
        num_envs=1,
        max_legal_actions=max_legal_actions,
        action_dim=envs.single_action_space.n,
        device=device,
    )
    feats = _extract_vector_legal_feature_tensors(
        infos,
        num_envs=1,
        max_legal_actions=max_legal_actions,
        feature_dim=feature_dim,
        device=device,
    )
    return ids, valid, feats


def _slot_logits(agent, obs_t, ids_t, valid_t, feats_t):
    with torch.no_grad():
        h = agent.state_encoder(obs_t)
        legal_emb = agent.action_embedding(ids_t.long())
        if agent.actor_mode == "legal_only":
            logits = torch.einsum("bd,bkd->bk", h, legal_emb)
        else:
            feat_emb = agent.action_feature_encoder(feats_t.float())
            repeated_state = h.unsqueeze(1).expand(-1, ids_t.shape[1], -1)
            logits = agent.action_scorer(torch.cat([repeated_state, legal_emb, feat_emb], dim=-1)).squeeze(-1)
        logits = logits.masked_fill(~valid_t.bool(), -1e8)
    return logits


def test_shape_alignment(envs, infos, verbose_slots=8):
    uw = envs.envs[0].unwrapped
    feats_raw = np.asarray(infos["legal_action_features_padded"], dtype=np.float32)
    ids_raw = np.asarray(infos["legal_global_ids_padded"], dtype=np.int64)
    valid_raw = np.asarray(infos["legal_action_valid_mask"], dtype=bool)
    feats = feats_raw[0] if feats_raw.ndim == 3 else feats_raw
    ids = ids_raw[0] if ids_raw.ndim == 2 else ids_raw
    valid = valid_raw[0] if valid_raw.ndim == 2 else valid_raw

    fd_raw = infos.get("legal_action_feature_dim", -1)
    fv_raw = infos.get("legal_action_feature_version", "")
    if isinstance(fd_raw, (list, tuple, np.ndarray)):
        feature_dim_reported = int(np.asarray(fd_raw).reshape(-1)[0])
    else:
        feature_dim_reported = int(fd_raw)
    if isinstance(fv_raw, (list, tuple, np.ndarray)):
        feature_version = str(np.asarray(fv_raw).reshape(-1)[0])
    else:
        feature_version = str(fv_raw)

    assert feats.shape == (int(uw._max_legal_actions), int(uw.ACTION_FEATURE_DIM)), (
        f"feature shape mismatch: {feats.shape} != {(uw._max_legal_actions, uw.ACTION_FEATURE_DIM)}"
    )
    assert feats.dtype == np.float32, f"feature dtype mismatch: {feats.dtype}"
    assert feature_dim_reported == int(uw.ACTION_FEATURE_DIM), (
        f"reported feature dim mismatch: {feature_dim_reported} != {uw.ACTION_FEATURE_DIM}"
    )
    assert np.allclose(feats[~valid], 0.0), "invalid/padded slot features are not zero"

    # alignment check: slot k -> gid -> raw action -> recomputed feature vector
    mapping = uw._current_legal_id_to_raw_index
    legal_actions = uw._current_legal_actions
    obs = uw.tribes_env._last_obs
    checked = 0
    for k in np.where(valid)[0]:
        gid = int(ids[k])
        assert gid in mapping, f"slot {k} gid {gid} missing from legal_id_to_raw_index"
        raw_idx = int(mapping[gid])
        assert 0 <= raw_idx < len(legal_actions), f"slot {k} gid {gid} raw_idx out of range: {raw_idx}"
        expected = uw._compute_legal_action_feature_vector(legal_actions[raw_idx], obs)
        if not np.allclose(expected, feats[k], atol=1e-6):
            raise AssertionError(f"slot {k} feature mismatch vs recompute")
        checked += 1
        if checked >= 64:
            break

    print(
        f"[TEST1] PASS shape/alignment: shape={feats.shape} dtype={feats.dtype} "
        f"feature_dim={feature_dim_reported} version={feature_version} checked_slots={checked}"
    )

    # per-user request: one state slot dump
    print("[TEST1] slot dump (first valid slots):")
    shown = 0
    for k in np.where(valid)[0]:
        gid = int(ids[k])
        raw_idx = mapping.get(gid, None)
        if raw_idx is None:
            continue
        action = legal_actions[int(raw_idx)]
        a_type = str(action.get("type", "UNKNOWN")).upper()
        summary = str(action.get("repr", ""))[:120]
        print(
            json.dumps(
                {
                    "slot_id": int(k),
                    "global_action_id": gid,
                    "raw_action_type": a_type,
                    "raw_action_summary": summary,
                    "valid_mask": bool(valid[k]),
                    "feature_vector": np.round(feats[k], 4).tolist(),
                },
                ensure_ascii=True,
            )
        )
        shown += 1
        if shown >= int(verbose_slots):
            break


def test_actor_uses_features(envs, obs, infos, model_path=None, seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cpu")
    feature_dim = _infer_feature_dim(infos, envs)
    ids_t, valid_t, feats_t = _extract_tensors(infos, envs, device, feature_dim=feature_dim)
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)

    agent = Agent(
        envs,
        actor_mode="legal_features",
        max_legal_actions=1024,
        legal_action_feature_dim=int(feature_dim),
    ).to(device)
    loaded = False
    if model_path:
        sd = torch.load(model_path, map_location=device)
        agent.load_state_dict(sd, strict=False)
        loaded = True

    logits_base = _slot_logits(agent, obs_t, ids_t, valid_t, feats_t)
    logits_zero = _slot_logits(agent, obs_t, ids_t, valid_t, torch.zeros_like(feats_t))
    diff_zero = torch.mean(torch.abs(logits_base - logits_zero)).item()
    max_diff_zero = torch.max(torch.abs(logits_base - logits_zero)).item()

    perturbed = feats_t.clone()
    valid_slots = torch.where(valid_t[0])[0]
    if valid_slots.numel() > 0:
        k = int(valid_slots[0].item())
        perturbed[0, k, 1] = perturbed[0, k, 1] + 1.0  # newly_revealed_tiles feature
    logits_perturbed = _slot_logits(agent, obs_t, ids_t, valid_t, perturbed)
    diff_perturb = torch.mean(torch.abs(logits_base - logits_perturbed)).item()
    max_diff_perturb = torch.max(torch.abs(logits_base - logits_perturbed)).item()

    assert max_diff_zero > 1e-9, "logits unchanged when zeroing features; features appear unused"
    assert max_diff_perturb > 1e-9, "logits unchanged when perturbing one feature; features appear unused"

    print(
        f"[TEST2] PASS feature influence: loaded_model={loaded} "
        f"mean|base-zero|={diff_zero:.6f} max|base-zero|={max_diff_zero:.6f} "
        f"mean|base-perturb|={diff_perturb:.6f} max|base-perturb|={max_diff_perturb:.6f}"
    )


def _choose_action(agent, obs_t, ids_t, valid_t, feats_t):
    with torch.no_grad():
        action, slot, _, _, _ = agent.get_action_and_value(
            obs_t,
            legal_global_ids=ids_t,
            legal_action_valid_mask=valid_t,
            legal_action_features=feats_t if agent.actor_mode == "legal_features" else None,
        )
    return int(action[0].item()), int(slot[0].item())


def test_selected_vs_average_move_features(model_path=None, steps=400, seed=7):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cpu")

    envs, obs, infos = make_sync_env(seed=seed)
    try:
        feature_dim = _infer_feature_dim(infos, envs)
        ids_t, valid_t, feats_t = _extract_tensors(infos, envs, device, feature_dim=feature_dim)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device)

        agent = Agent(
            envs,
            actor_mode="legal_features",
            max_legal_actions=1024,
            legal_action_feature_dim=int(feature_dim),
        ).to(device)
        loaded = False
        if model_path:
            sd = torch.load(model_path, map_location=device)
            agent.load_state_dict(sd, strict=False)
            loaded = True

        stats = MoveQualityStats()
        total_legal_moves = 0.0
        reveal_margins = []
        adj_delta_margins = []
        village_delta_margins = []

        for _ in range(int(steps)):
            valid_np = valid_t[0].cpu().numpy().astype(bool)
            feats_np = feats_t[0].cpu().numpy()
            move_mask = feats_np[:, 0] > 0.5
            legal_move_mask = np.logical_and(valid_np, move_mask)
            legal_move_idxs = np.where(legal_move_mask)[0]
            if legal_move_idxs.size > 0:
                stats.move_states += 1
                total_legal_moves += float(legal_move_idxs.size)

            action, slot = _choose_action(agent, obs_t, ids_t, valid_t, feats_t)
            chosen_feat = feats_np[slot]
            chosen_is_move = bool(chosen_feat[0] > 0.5)
            if chosen_is_move:
                stats.chosen_is_move += 1

            if legal_move_idxs.size > 0 and chosen_is_move:
                legal_move_feats = feats_np[legal_move_idxs]
                avg_reveal = float(np.mean(legal_move_feats[:, 1]))
                avg_adj_delta = float(np.mean(legal_move_feats[:, 3]))
                avg_village_delta = float(np.mean(legal_move_feats[:, 7]))
                c_reveal = float(chosen_feat[1])
                c_adj_delta = float(chosen_feat[3])
                c_village_delta = float(chosen_feat[7])

                reveal_margins.append(c_reveal - avg_reveal)
                adj_delta_margins.append(c_adj_delta - avg_adj_delta)
                village_delta_margins.append(c_village_delta - avg_village_delta)
                if c_reveal > avg_reveal:
                    stats.chosen_above_avg_reveal += 1
                if c_adj_delta > avg_adj_delta:
                    stats.chosen_above_avg_adj_delta += 1
                if c_village_delta > avg_village_delta:
                    stats.chosen_better_village_delta += 1
                if float(chosen_feat[8]) > 0.5:
                    stats.chosen_backtrack += 1

            obs, _rew, term, trunc, infos = envs.step(np.array([action], dtype=np.int64))
            if bool(term[0]) or bool(trunc[0]):
                obs, infos = envs.reset()
            ids_t, valid_t, feats_t = _extract_tensors(infos, envs, device)
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)

        if stats.move_states > 0:
            stats.avg_legal_moves_per_state = float(total_legal_moves / stats.move_states)
        if reveal_margins:
            stats.reveal_margin_mean = float(np.mean(reveal_margins))
            stats.adj_delta_margin_mean = float(np.mean(adj_delta_margins))
            stats.village_delta_margin_mean = float(np.mean(village_delta_margins))

        print(
            "[TEST3] selected-vs-avg MOVE quality "
            f"(loaded_model={loaded}, steps={steps}): "
            f"move_states={stats.move_states}, "
            f"avg_legal_moves_per_state={stats.avg_legal_moves_per_state:.2f}, "
            f"chosen_is_move={stats.chosen_is_move}, "
            f"chosen_above_avg_reveal={stats.chosen_above_avg_reveal}, "
            f"chosen_above_avg_adj_delta={stats.chosen_above_avg_adj_delta}, "
            f"chosen_better_village_delta={stats.chosen_better_village_delta}, "
            f"chosen_backtrack={stats.chosen_backtrack}, "
            f"reveal_margin_mean={stats.reveal_margin_mean:.4f}, "
            f"adj_delta_margin_mean={stats.adj_delta_margin_mean:.4f}, "
            f"village_delta_margin_mean={stats.village_delta_margin_mean:.4f}"
        )
    finally:
        envs.close()


def test_feature_scale_distribution(seed=13, states=200):
    envs, obs, infos = make_sync_env(seed=seed)
    try:
        all_rows = []
        move_rows = []
        legal_count_hist = []
        move_count_hist = []
        for _ in range(int(states)):
            feats_raw = np.asarray(infos["legal_action_features_padded"], dtype=np.float32)
            valid_raw = np.asarray(infos["legal_action_valid_mask"], dtype=bool)
            feats = feats_raw[0] if feats_raw.ndim == 3 else feats_raw
            valid = valid_raw[0] if valid_raw.ndim == 2 else valid_raw
            rows = feats[valid]
            if rows.size > 0:
                all_rows.append(rows)
                move_mask = rows[:, 0] > 0.5
                move_rows.append(rows[move_mask])
                legal_count_hist.append(int(rows.shape[0]))
                move_count_hist.append(int(np.sum(move_mask)))

            # random legal action to advance
            ids_raw = np.asarray(infos["legal_global_ids_padded"], dtype=np.int64)
            ids = ids_raw[0] if ids_raw.ndim == 2 else ids_raw
            valid_slots = np.where(valid)[0]
            chosen_slot = int(np.random.choice(valid_slots))
            action = int(ids[chosen_slot])
            obs, _rew, term, trunc, infos = envs.step(np.array([action], dtype=np.int64))
            if bool(term[0]) or bool(trunc[0]):
                obs, infos = envs.reset()

        if not all_rows:
            raise RuntimeError("No valid legal rows collected.")
        all_concat = np.concatenate(all_rows, axis=0)
        move_concat = np.concatenate([r for r in move_rows if r.size > 0], axis=0) if any(r.size > 0 for r in move_rows) else np.zeros((0, all_concat.shape[1]), dtype=np.float32)
        mins = np.min(all_concat, axis=0)
        maxs = np.max(all_concat, axis=0)
        means = np.mean(all_concat, axis=0)
        print(
            "[TEST4] feature scale summary: "
            f"states={states}, legal_rows={all_concat.shape[0]}, move_rows={move_concat.shape[0]}, "
            f"legal_count_mean={np.mean(legal_count_hist):.2f}, legal_count_p50={np.median(legal_count_hist):.1f}, "
            f"move_count_mean={np.mean(move_count_hist):.2f}, move_count_p50={np.median(move_count_hist):.1f}"
        )
        # print concise table for key movement indices and flags
        idx_names = [
            (0, "is_move"),
            (1, "new_reveal_norm"),
            (3, "adj_fog_delta_norm"),
            (4, "is_zero_reveal_move"),
            (7, "village_dist_delta_norm"),
            (8, "is_immediate_backtrack"),
            (10, "capital_dist_delta_norm"),
            (12, "is_end_turn"),
            (13, "is_capture"),
            (21, "is_other"),
        ]
        for idx, name in idx_names:
            print(
                f"[TEST4] idx={idx:02d} {name:28s} "
                f"min={mins[idx]: .4f} max={maxs[idx]: .4f} mean={means[idx]: .4f}"
            )
    finally:
        envs.close()


def main():
    parser = argparse.ArgumentParser(description="Diagnostics for legal_features actor and legal-action feature tensor wiring.")
    parser.add_argument("--model-path", type=str, default=None, help="Optional checkpoint (.cleanrl_model) for legal_features agent.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--steps", type=int, default=500, help="Rollout steps for selected-vs-average test.")
    parser.add_argument("--states", type=int, default=250, help="Number of sampled states for scale distribution test.")
    parser.add_argument("--verbose-slots", type=int, default=8, help="How many slot dump rows to print in Test 1.")
    args = parser.parse_args()

    envs, obs, infos = make_sync_env(seed=args.seed)
    try:
        test_shape_alignment(envs, infos, verbose_slots=args.verbose_slots)
        test_actor_uses_features(envs, obs, infos, model_path=args.model_path, seed=args.seed)
    finally:
        envs.close()

    test_selected_vs_average_move_features(model_path=args.model_path, steps=args.steps, seed=args.seed + 1)
    test_feature_scale_distribution(seed=args.seed + 2, states=args.states)
    print("[DONE] diagnostics complete")


if __name__ == "__main__":
    main()
