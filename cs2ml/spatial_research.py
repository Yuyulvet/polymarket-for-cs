"""Fixed-time Inferno spatial ablation. Offline exploratory evidence, never orders."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from demoparser2 import DemoParser
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from .inround import snapshot_problem
from .inround_model import attach_chronology, forward_splits, hierarchical_weights
from .inplay_map_research import _verified_rounds
from .map1_data import path_key, roster_key

VERSION = "inferno_spatial_fixed_time_v1"
OFFSETS = (15, 30, 45)
FIELDS = ["X", "Y", "Z", "health", "armor_value", "current_equip_value",
          "is_alive", "team_name", "steamid", "last_place_name", "active_weapon_name"]
VARIANTS = ("macro", "coarse_geometry", "regional_deployment")


def snapshot_features(snapshot, row, elapsed):
    """No future event, future attacker identity or round outcome enters features."""
    s = snapshot.copy()
    s["team_name"] = s.team_name.replace({"TERRORIST": "T"})
    problem = snapshot_problem(s)
    if problem:
        raise ValueError(problem)
    for side, column in (("CT", "ct_roster"), ("T", "t_roster")):
        if roster_key(s.loc[s.team_name.eq(side), "steamid"].tolist()) != row[column]:
            raise ValueError("roster_mismatch")
    if not s.is_alive.all() or not s.health.eq(100).all():
        raise ValueError("already_damaged_or_dead")
    if not np.isfinite(s[["X", "Y", "Z", "armor_value"]].to_numpy(float)).all():
        raise ValueError("invalid_spatial_or_armor")
    if s.last_place_name.isna().any() or s.last_place_name.astype(str).str.strip().eq("").any():
        raise ValueError("missing_region")
    sign = 1 if row["ct_is_a"] else -1
    macro = {"elapsed": float(elapsed), "round_num": float(row["round_num"]),
             "score_gap": sign * (row["score_a_before"] - row["score_b_before"]),
             "pistol": float(row["round_num"] in (1, 13))}
    coarse, deployment = {}, {}
    sides = {side: s[s.team_name.eq(side)].sort_values("steamid") for side in ("CT", "T")}
    for side, players in sides.items():
        # Equipment distributions and held weapons available to both variants.
        for field in ("current_equip_value", "armor_value"):
            for i, value in enumerate(sorted(players[field].astype(float))):
                macro[f"{side}_{field}_{i}"] = value
        for weapon, count in Counter(players.active_weapon_name.fillna("unknown")).items():
            macro[f"{side}_held_{weapon}"] = float(count)
        coords = players[["X", "Y"]].to_numpy(float)
        distances = np.linalg.norm(coords[:, None] - coords[None, :], axis=2)
        coarse[f"{side}_spread"] = float(distances[np.triu_indices(5, 1)].mean())
        coarse[f"{side}_max_pair"] = float(distances.max())
        places = players.last_place_name.astype(str)
        coarse[f"{side}_n_places"] = float(places.nunique())
        for place, count in Counter(places).items():
            deployment[f"{side}_place_{place}"] = float(count)
        # Fixed coordinate grid, not a grid tuned against test outcomes.
        cells = [f"{int(x // 800)}_{int(y // 800)}" for x, y in coords]
        for cell, count in Counter(cells).items():
            deployment[f"{side}_cell_{cell}"] = float(count)
    ct, tt = sides["CT"], sides["T"]
    cxy, txy = ct[["X", "Y"]].to_numpy(float), tt[["X", "Y"]].to_numpy(float)
    cross = np.linalg.norm(txy[:, None] - cxy[None, :], axis=2)
    coarse["nearest_enemy_mean"] = float(cross.min(axis=1).mean())
    coarse["nearest_enemy_min"] = float(cross.min())
    friends = np.linalg.norm(txy[:, None] - txy[None, :], axis=2)
    for radius in (600, 1000):
        nearby_ct, nearby_t = cross <= radius, friends <= radius
        local_gap = nearby_t.sum(axis=1) - nearby_ct.sum(axis=1)
        local_equip = (nearby_t @ tt.current_equip_value.to_numpy(float)
                       - nearby_ct @ ct.current_equip_value.to_numpy(float))
        for name, values in (("ct", nearby_ct.sum(axis=1)), ("t", nearby_t.sum(axis=1)),
                             ("numbers_gap", local_gap), ("equipment_gap", local_equip)):
            for stat, func in (("min", np.min), ("max", np.max), ("mean", np.mean)):
                deployment[f"local_{radius}_{name}_{stat}"] = float(func(values))
    return macro, coarse, deployment


def extract_one(task):
    path, rows, winner_roster, plant_ticks, cache_dir = task
    source = Path(path).stat()
    signature = f"{VERSION}|{path_key(path)}|{source.st_size}|{source.st_mtime_ns}|{rows}|{plant_ticks}|{winner_roster}"
    cache = Path(cache_dir) / (hashlib.sha256(signature.encode()).hexdigest() + ".json")
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    checkpoints = [(r, dt, int(r["round_start_tick"]) + dt * 64) for r in rows for dt in OFFSETS]
    ticks = sorted({tick for r, dt, tick in checkpoints if tick < r["round_end_tick"]})
    snapshots = DemoParser(path).parse_ticks(FIELDS, ticks=ticks)
    by_tick = {int(t): group for t, group in snapshots.groupby("tick")}
    kept, rejected = [], Counter()
    for row, elapsed, tick in checkpoints:
        if tick >= row["round_end_tick"]:
            rejected["round_already_ended"] += 1
            continue
        if any(row["round_start_tick"] <= p <= tick for p in plant_ticks):
            rejected["already_planted"] += 1
            continue
        if tick not in by_tick:
            rejected["missing_tick"] += 1
            continue
        try:
            macro, coarse, deployment = snapshot_features(by_tick[tick], row, elapsed)
        except ValueError as exc:
            rejected[str(exc)] += 1
            continue
        kept.append({"demo_path": path, "round_num": int(row["round_num"]), "elapsed": elapsed,
                     "state_tick": tick, "label_ct_win": int(row["winner_side"] == "CT"),
                     "label_ct_map_win": int(winner_roster == row["ct_roster"]),
                     "macro": macro, "coarse": coarse, "deployment": deployment})
    result = {"rows": kept, "rejected": dict(rejected), "demo_path": path}
    cache.write_text(json.dumps(result), encoding="utf-8")
    return result


def features(records, variant):
    return [{**r["macro"], **(r["coarse"] if variant != "macro" else {}),
             **(r["deployment"] if variant == "regional_deployment" else {})} for r in records]


def metrics(frame, probability, target):
    y, p = frame[target].to_numpy(int), np.clip(probability, 1e-8, 1-1e-8)
    w = hierarchical_weights(frame)
    return {"brier": float(np.average((y-p)**2, weights=w)),
            "log_loss": float(np.average(-y*np.log(p)-(1-y)*np.log(1-p), weights=w)),
            "accuracy": float(np.average((p >= .5) == y, weights=w)),
            "auc": float(roc_auc_score(y, p, sample_weight=w)) if len(set(y)) == 2 else None}


def paired_delta(frame, baseline, candidate, target):
    y = frame[target].to_numpy(int)
    w = hierarchical_weights(frame)
    d = pd.DataFrame({"series": frame.match_id.to_numpy(), "weight": w,
                      "diff": w*((candidate-y)**2 - (baseline-y)**2)})
    g = d.groupby("series")[["diff", "weight"]].sum()
    values = (g["diff"] / g.weight).to_numpy()
    rng = np.random.default_rng(20260918)
    boot = rng.choice(values, (3000, len(values)), replace=True).mean(axis=1)
    return {"candidate_minus_baseline_brier": float(values.mean()),
            "series_bootstrap_95pct": np.quantile(boot, [.025, .975]).tolist(),
            "series": len(values), "negative_favors_candidate": True}


def evaluate(data):
    records = data.to_dict("records")
    splits = list(forward_splits(data, n_splits=4, min_train_series=15))
    predictions = {f"{target}__{v}": np.full(len(data), np.nan)
                   for target in ("label_ct_win", "label_ct_map_win") for v in VARIANTS}
    folds = []
    with threadpool_limits(limits=2):
        for i, (train, test) in enumerate(splits):
            tr, te = data.iloc[train], data.iloc[test]
            weights = hierarchical_weights(tr)
            fold = {"train_series": int(tr.match_id.nunique()), "test_series": int(te.match_id.nunique()),
                    "train_rows": len(tr), "test_rows": len(te),
                    "train_max_available": str(tr.available_at.max()), "test_min_start": str(te.start_at.min()),
                    "test_max_start": str(te.start_at.max()), "overlap": len(set(tr.match_id) & set(te.match_id)),
                    "metrics": {}}
            for variant in VARIANTS:
                dictionaries = features(records, variant)
                encoder = DictVectorizer(sparse=False)
                xtrain = encoder.fit_transform([dictionaries[j] for j in train])
                xtest = encoder.transform([dictionaries[j] for j in test])
                for target in ("label_ct_win", "label_ct_map_win"):
                    model = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=7,
                        min_samples_leaf=30, learning_rate=.05, l2_regularization=5,
                        early_stopping=False, random_state=20260918)
                    model.fit(xtrain, tr[target], sample_weight=weights/weights.mean())
                    pred = model.predict_proba(xtest)[:, 1]
                    key = f"{target}__{variant}"
                    predictions[key][test] = pred
                    fold["metrics"][key] = metrics(te, pred, target)
            folds.append(fold)
            print(f"evaluated fold {i+1}/{len(splits)}", flush=True)
    valid = np.isfinite(predictions["label_ct_win__macro"])
    scored = data.loc[valid].copy()
    if scored.empty:
        return {"status": "insufficient_data", "folds": folds}, scored
    report = {"status": "exploratory_not_acceptance", "folds": folds,
              "scored_series": int(scored.match_id.nunique()), "scored_maps": int(scored.demo_path.nunique()),
              "scored_rounds": len(scored[["demo_path", "round_num"]].drop_duplicates()),
              "scored_snapshots": len(scored), "metrics": {}, "comparisons": {}, "by_elapsed": {}}
    for key, pred in predictions.items():
        scored[key] = pred[valid]
        target = key.split("__")[0]
        report["metrics"][key] = metrics(scored, pred[valid], target)
    for target in ("label_ct_win", "label_ct_map_win"):
        for base in ("macro", "coarse_geometry"):
            report["comparisons"][f"{target}__regional_vs_{base}"] = paired_delta(
                scored, scored[f"{target}__{base}"].to_numpy(),
                scored[f"{target}__regional_deployment"].to_numpy(), target)
    for elapsed, group in scored.groupby("elapsed"):
        report["by_elapsed"][str(elapsed)] = {"rows": len(group), "series": int(group.match_id.nunique()),
            "metrics": {key: metrics(group, group[key].to_numpy(), key.split("__")[0]) for key in predictions}}
    return report, scored.drop(columns=["macro", "coarse", "deployment"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data/map1/rebuild/full-20260916"))
    p.add_argument("--output", type=Path, default=Path("reports/inferno_spatial_20260918_v1"))
    p.add_argument("--workers", type=int, default=3)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cache = args.output / "extraction"
    cache.mkdir(exist_ok=True)
    rounds = pd.read_parquet(args.data_dir / "round_states.parquet")
    labels = pd.read_parquet(args.data_dir / "map_labels.parquet")
    events = pd.read_parquet(args.data_dir / "inround_events.parquet")
    labels["path_key"] = labels.demo_path.map(path_key)
    label_by_path = labels.set_index("path_key").to_dict("index")
    tasks, rejected_maps = [], {}
    for path, group in rounds[rounds.map_name.eq("de_inferno")].groupby("demo_path"):
        try:
            label = label_by_path[path_key(path)]
            clean, _, _, _ = _verified_rounds(group, label)
            if not Path(path).exists():
                raise ValueError("missing_demo")
            plants = events.loc[events.demo_path.eq(path) & events.event_type.eq("bomb_planted"), "event_tick"].astype(int).tolist()
            tasks.append((path, clean.to_dict("records"), label["winner_roster"], plants, str(cache)))
        except (ValueError, KeyError) as exc:
            rejected_maps[path] = str(exc)
    protocol = {"version": VERSION, "map": "Inferno", "offsets_seconds": OFFSETS,
                "eligible_maps": len(tasks), "rejected_maps": rejected_maps,
                "cohort": "Fixed checkpoints, all ten alive at 100HP, no planted bomb; same rows in every variant.",
                "learner": "HGB 100 trees, 7 leaves, min_leaf30, lr .05, L2=5; no tuning, no random early-stop split",
                "validation": "4 forward series blocks; labels available strictly before test; equal series/round/snapshot weights",
                "primary": "Round winner Brier; regional versus macro and coarse geometry",
                "secondary": "Map winner Brier and checkpoint strata; exploratory, not independent confirmations",
                "limitations": ["Full HP is not proof that no shots have been fired.",
                    "Euclidean distance ignores walls, visibility, utility and true rotate paths.",
                    "Regional occupancy does not reveal intended future attack destination.",
                    "No live receipt time, execution costs, market prices or profitability validation.",
                    "Previously researched history is not an untouched holdout; confidence intervals are exploratory."]}
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    all_rows, exclusions, failures = [], Counter(), {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(extract_one, task): task[0] for task in tasks}
        for i, job in enumerate(as_completed(jobs)):
            try:
                result = job.result()
                all_rows.extend(result["rows"])
                exclusions.update(result["rejected"])
            except Exception as exc:
                failures[jobs[job]] = repr(exc)
            print(f"extracted {i+1}/{len(tasks)} maps; kept {len(all_rows)} snapshots; failures {len(failures)}", flush=True)
    if not all_rows:
        raise RuntimeError(f"No retained rows: {failures}")
    data, chronology = attach_chronology(pd.DataFrame(all_rows), pd.read_parquet("data/map1/history.parquet"))
    data = data.sort_values(["start_at", "demo_path", "round_num", "elapsed"]).reset_index(drop=True)
    report, scored = evaluate(data)
    report.update(protocol=protocol, exclusions=dict(exclusions), failures=failures, chronology=chronology,
                  total_retained_snapshots=len(data), retained_rounds=len(data[["demo_path", "round_num"]].drop_duplicates()))
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    scored.to_parquet(args.output / "predictions.parquet", index=False)
    print(json.dumps({k: report.get(k) for k in ("status", "scored_series", "scored_rounds", "metrics", "comparisons")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
