"""Frozen next-action diagnostics, with explicit separation from policy-win-rate claims."""

from pathlib import Path

import numpy as np
from rift_common.actions import ACTION_NAMES, GROUPS, ROLES, Action
from rift_common.artifacts import atomic_json, file_digest, read_json
from rift_common.coaching import DESTINATIONS
from rift_common.dataset import load_dataset
from rift_common.metrics import behavior_metrics


def coordination_metrics(arrays, selected, predictions):
    """Pair only measured same-match/team/time rows; never interpolate missing macro rows."""
    groups = arrays["groups"][selected]
    keys = list(
        zip(
            arrays["match_ids"][selected], arrays["teams"][selected], arrays["timestamps"][selected]
        )
    )
    macros = {
        key: int(action)
        for key, group, action in zip(keys, groups, predictions)
        if group == GROUPS.index("TEAM")
    }
    counts = {
        name: 0
        for name in (
            "role_rows",
            "missing_macro_row",
            "unsupported_pair",
            "forced_respawn",
            "aligned",
            "independent",
            "withheld_conflict",
        )
    }
    for key, group, action in zip(keys, groups, predictions):
        if GROUPS[group] not in ROLES:
            continue
        counts["role_rows"] += 1
        if key not in macros:
            counts["missing_macro_row"] += 1
            continue
        macro = macros[key]
        if action < 0 or macro < 0:
            label = "unsupported_pair"
        elif action == Action.WAIT_RESPAWN or macro == Action.WAIT_RESPAWN:
            label = "forced_respawn"
        elif action == macro:
            label = "aligned"
        elif action in DESTINATIONS and macro in DESTINATIONS:
            label = "withheld_conflict"
        else:
            label = "independent"
        counts[label] += 1
    return {
        **counts,
        "simultaneous_supported_nonconflicting_pairs": counts["aligned"] + counts["independent"],
        "meaning": "Raw routing proposals with instantaneous symbolic abstention rule; no persistence simulation or strategic compatibility proof",
    }


def describe_dataset(path):
    arrays, metadata = load_dataset(path)
    counts = np.zeros((len(GROUPS), len(ACTION_NAMES)), dtype=int)
    np.add.at(counts, (arrays["groups"], arrays["actions"]), 1)
    return {
        "matches": len(set(arrays["match_ids"])),
        "episodes": len(set(arrays["episode_ids"])),
        "transitions": len(arrays["actions"]),
        "duration_seconds": {
            "min": float(arrays["durations"].min()),
            "median": float(np.median(arrays["durations"])),
            "max": float(arrays["durations"].max()),
        },
        "action_names": ACTION_NAMES,
        "counts_by_group": {group: counts[i].tolist() for i, group in enumerate(GROUPS)},
        "behavior_evidence": metadata["behavior_evidence"],
        "identified_transitions": int(
            arrays.get("action_identified", np.ones(len(arrays["actions"]), dtype=bool)).sum()
        ),
        "excluded": metadata["excluded"],
        "state_schema": metadata["state_schema"],
        "action_schema": metadata["action_schema"],
    }


def evaluate_policy(
    dataset,
    bundle,
    output,
    *,
    samples=500,
    new_matches=False,
    stress=False,
    match_ids=None,
    project_state=False,
):
    from offline_rift.inference.coach import load_policy

    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    arrays, metadata = load_dataset(dataset)
    policy = load_policy(bundle)
    projection = None
    if project_state:
        from rift_common.items import ItemCatalog
        from rift_common.state import StateEncoder

        source_names = metadata["feature_names"]
        target_names = policy.metadata["feature_names"]
        canonical = StateEncoder(
            ItemCatalog(metadata["item_catalog"]), schema="decision-state-v1"
        ).names
        if (
            not new_matches
            or metadata["state_schema"] != "decision-state-v2"
            or policy.metadata["state_schema"] != "decision-state-v1"
            or target_names != canonical
            or source_names[: len(canonical)] != canonical
        ):
            raise ValueError(
                "Only explicit new-match v2-to-v1 canonical baseline projection is supported"
            )
        arrays = {
            **arrays,
            **{
                key: arrays[key][:, : len(canonical)]
                for key in ("observations", "next_observations")
            },
        }
        projection = {
            "from": metadata["state_schema"],
            "to": policy.metadata["state_schema"],
            "dropped_features": source_names[len(canonical) :],
        }
        metadata = {**metadata, "feature_names": target_names}
    report = policy.metadata["training_report"]
    training, validation = set(report["training_matches"]), set(report["validation_matches"])
    if not training or not validation or training & validation:
        raise ValueError("Invalid or incompatible model match split")
    corpus = set(arrays["match_ids"])
    requested = None if match_ids is None else set(match_ids)
    if requested is not None and (
        not new_matches
        or not requested
        or not requested <= corpus
        or requested & (training | validation)
    ):
        raise ValueError(
            "Requested evaluation IDs must be nonempty, present and absent from training/development; use new_matches"
        )
    if new_matches:
        evaluated_matches = corpus - (training | validation)
        if requested is not None:
            evaluated_matches &= requested
        if not evaluated_matches:
            raise ValueError("No new matches outside the frozen training/development corpus")
    else:
        if training | validation | set(report.get("holdout_matches", [])) != corpus:
            raise ValueError("Invalid or incompatible model match split")
        if metadata["provenance"]["fingerprint"] != policy.metadata["provenance"]["fingerprint"]:
            raise ValueError("Dataset differs from the frozen model corpus")
        expected_digest = policy.metadata.get("dataset_sha256")
        if expected_digest is not None and file_digest(dataset) != expected_digest:
            raise ValueError("Frozen dataset integrity mismatch")
        evaluated_matches = validation
    if any(
        metadata[key] != policy.metadata[key]
        for key in ("feature_names", "reward", "time_unit_seconds")
    ):
        raise ValueError("Evaluation dataset contract differs from frozen policy")
    mask = np.isin(arrays["match_ids"], list(evaluated_matches))
    predicted, scores = policy.predict_batch(
        arrays["observations"][mask], arrays["groups"][mask], arrays["action_masks"][mask]
    )
    truth = arrays["actions"][mask]
    metrics = behavior_metrics(truth, predicted, len(ACTION_NAMES))
    if not new_matches and metrics != report["validation"]:
        raise ValueError("Serving predictions do not reproduce training evaluation")
    matches, groups = np.unique(arrays["match_ids"][mask], return_inverse=True)
    matrices = np.zeros((len(matches), len(ACTION_NAMES), len(ACTION_NAMES) + 1), dtype=int)
    np.add.at(matrices, (groups, truth, np.where(predicted < 0, len(ACTION_NAMES), predicted)), 1)
    if samples < 1:
        raise ValueError("Positive bootstrap samples required")
    rng = np.random.default_rng(20261105)
    estimates = []
    for _ in range(samples):
        matrix = matrices[rng.integers(len(matches), size=len(matches))].sum(0)
        supported = matrix.sum(1)
        proposed = matrix[:, : len(ACTION_NAMES)].sum(0)
        diagonal = matrix[:, : len(ACTION_NAMES)].diagonal()
        f1 = np.divide(
            2 * diagonal,
            supported + proposed,
            out=np.zeros(len(ACTION_NAMES)),
            where=supported + proposed > 0,
        )
        estimates.append(f1[(supported + proposed) > 0].mean())
    result = {
        "dataset_sha256": file_digest(dataset),
        "bundle": str(Path(bundle).resolve()),
        "bundle_sha256": file_digest(
            Path(bundle)
            / ("coach.json" if (Path(bundle) / "coach.json").exists() else "bundle.json")
        ),
        "algorithm": policy.algorithm,
        "state_projection": projection,
        "evaluation_mode": "new_matches" if new_matches else "frozen_validation",
        "validation_matches": sorted(validation),
        "evaluated_matches": sorted(evaluated_matches),
        "requested_match_ids": sorted(requested) if requested is not None else None,
        "excluded_previously_seen_matches": sorted(corpus & (training | validation))
        if new_matches
        else [],
        "evaluated_transitions": int(mask.sum()),
        "majority_baseline": float(np.mean(truth == policy.counts.sum(0).argmax())),
        "retention_coverage": float(np.mean(predicted >= 0)),
        "metrics": metrics,
        "identified_metrics": behavior_metrics(
            truth[
                arrays.get("action_identified", np.ones(len(arrays["actions"]), dtype=bool))[mask]
            ],
            predicted[
                arrays.get("action_identified", np.ones(len(arrays["actions"]), dtype=bool))[mask]
            ],
            len(ACTION_NAMES),
        )
        if arrays.get("action_identified", np.ones(len(arrays["actions"]), dtype=bool))[mask].any()
        else None,
        "per_group": {
            group: {
                **behavior_metrics(
                    truth[arrays["groups"][mask] == i],
                    predicted[arrays["groups"][mask] == i],
                    len(ACTION_NAMES),
                ),
                "training_action_support": policy.counts[i].tolist(),
                "identified": behavior_metrics(
                    truth[
                        (arrays["groups"][mask] == i)
                        & arrays.get(
                            "action_identified", np.ones(len(arrays["actions"]), dtype=bool)
                        )[mask]
                    ],
                    predicted[
                        (arrays["groups"][mask] == i)
                        & arrays.get(
                            "action_identified", np.ones(len(arrays["actions"]), dtype=bool)
                        )[mask]
                    ],
                    len(ACTION_NAMES),
                )
                if np.any(
                    (arrays["groups"][mask] == i)
                    & arrays.get("action_identified", np.ones(len(arrays["actions"]), dtype=bool))[
                        mask
                    ]
                )
                else None,
                "retention_coverage": float(np.mean(predicted[arrays["groups"][mask] == i] >= 0)),
                "majority_baseline": float(
                    np.mean(truth[arrays["groups"][mask] == i] == policy.counts[i].argmax())
                ),
                "largest_prediction_fraction": float(
                    np.bincount(
                        predicted[(arrays["groups"][mask] == i) & (predicted >= 0)],
                        minlength=len(ACTION_NAMES),
                    ).max()
                    / max(1, np.sum((arrays["groups"][mask] == i) & (predicted >= 0)))
                ),
            }
            for i, group in enumerate(GROUPS)
            if np.any(arrays["groups"][mask] == i)
        },
        "score_quantiles": np.quantile(
            scores[np.isfinite(scores)], [0, 0.01, 0.5, 0.99, 1]
        ).tolist()
        if np.isfinite(scores).any()
        else [],
        "score_meaning": "per-policy scores; child methods may use different units"
        if hasattr(policy, "policies")
        else "Q estimate"
        if policy.algorithm == "cql"
        else "policy logit",
        "behavior_macro_f1_match_bootstrap_95": np.quantile(estimates, [0.025, 0.975]).tolist(),
        "bootstrap_samples": samples,
        "policy_winrate": None,
        "policy_improvement": None,
        "limitations": [
            "Behavior agreement and bootstrap intervals are diagnostics of inferred actions, not strategic utility",
            "New match IDs exclude this bundle's training/development corpus; prior human or other-model consultation is not identifiable"
            if new_matches
            else "These are previously consulted retrospective matches, not a new prospective test",
            "No simulator or authenticated live-game outcome study has been performed",
            "Shared team rewards and partial observations do not identify individual causal credit",
        ],
    }
    if stress:
        from rift_common.actions import feasible_actions
        from rift_common.observation_masks import COLD_START_FIELDS, HEALTH_FIELDS

        result["availability_stress"] = {}
        for scenario, fields in {
            "unknown_health": HEALTH_FIELDS,
            "cold_history": COLD_START_FIELDS,
        }.items():
            observed = arrays["observations"][mask].copy()
            observed[:, [metadata["feature_names"].index(field) for field in fields]] = 0
            omitted_masks = feasible_actions(observed, metadata["feature_names"])
            still_feasible = (predicted >= 0) & omitted_masks[
                np.arange(len(predicted)), np.maximum(predicted, 0)
            ]
            changed, _ = policy.predict_batch(
                observed,
                arrays["groups"][mask],
                omitted_masks,
            )
            voluntary = still_feasible & (predicted != Action.WAIT_RESPAWN)
            group_results = {}
            for i, group in enumerate(GROUPS):
                rows = arrays["groups"][mask] == i
                if not rows.any():
                    continue
                eligible = rows & still_feasible
                choices = rows & voluntary
                group_results[group] = {
                    "changed_fraction": float(np.mean(changed[rows] != predicted[rows])),
                    "retention": float(np.mean(changed[rows] >= 0)),
                    "original_choice_made_infeasible_fraction": float(
                        np.mean(((predicted >= 0) & ~still_feasible)[rows])
                    ),
                    "changed_when_original_still_feasible": float(
                        np.mean(changed[eligible] != predicted[eligible])
                    )
                    if eligible.any()
                    else None,
                    "voluntary_still_feasible_rows": int(choices.sum()),
                    "voluntary_changed_when_original_still_feasible": float(
                        np.mean(changed[choices] != predicted[choices])
                    )
                    if choices.any()
                    else None,
                }
            result["availability_stress"][scenario] = group_results
    if hasattr(policy, "policies"):
        result["coordination"] = coordination_metrics(arrays, mask, predicted)
        result["score_quantiles"] = []
        for scope, child in policy.policies.items():
            if scope in result["per_group"]:
                result["per_group"][scope]["algorithm"] = child.algorithm
    identified = arrays["action_identified"][mask]
    for i, group in enumerate(GROUPS):
        rows = (arrays["groups"][mask] == i) & identified
        if group not in result["per_group"]:
            continue
        result["per_group"][group]["identified_rows"] = int(rows.sum())
        result["per_group"][group]["identified_majority_agreement"] = (
            float(np.mean(truth[rows] == policy.counts[i].argmax())) if rows.any() else None
        )
        result["per_group"][group]["identified_match_bootstrap"] = identified_bootstrap(
            truth[rows], predicted[rows], arrays["match_ids"][mask][rows], samples=samples
        )
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        output / "predictions.npz",
        truth=truth,
        predicted=predicted,
        match_ids=arrays["match_ids"][mask],
        groups=arrays["groups"][mask],
        teams=arrays["teams"][mask],
        timestamps=arrays["timestamps"][mask],
        actors=arrays["actors"][mask],
        identified=arrays["action_identified"][mask],
        scores=scores,
    )
    atomic_json(output / "report.json", result)
    return result


def identified_bootstrap(truth, predicted, matches, *, samples=500):
    """Resample whole matches, keeping correlated perspectives together."""
    if not len(truth):
        return None
    identifiers, inverse = np.unique(matches, return_inverse=True)
    matrices = np.zeros((len(identifiers), len(ACTION_NAMES), len(ACTION_NAMES) + 1), dtype=int)
    np.add.at(matrices, (inverse, truth, np.where(predicted < 0, len(ACTION_NAMES), predicted)), 1)
    rng = np.random.default_rng(20261110)
    estimates = []
    for _ in range(samples):
        matrix = matrices[rng.integers(len(identifiers), size=len(identifiers))].sum(0)
        denominator = matrix.sum(1) + matrix[:, :-1].sum(0)
        f1 = np.divide(
            2 * matrix[:, :-1].diagonal(),
            denominator,
            out=np.zeros(len(ACTION_NAMES)),
            where=denominator > 0,
        )
        estimates.append(float(f1[denominator > 0].mean()))
    return {
        "macro_f1_95": np.quantile(estimates, [0.025, 0.975]).tolist(),
        "matches_with_identified_evidence": len(identifiers),
        "samples": samples,
        "unit": "match; recurring players may remain dependent",
    }


def compare_holdout(previous, candidate, destination, *, samples=1000):
    """Compare fixed products on exactly the same unseen rows with paired match bootstrap."""
    paths = {"previous": Path(previous), "candidate": Path(candidate)}
    if samples < 1:
        raise ValueError("Positive paired bootstrap samples required")
    reports = {name: read_json(path / "report.json") for name, path in paths.items()}
    predictions = {}
    for name, path in paths.items():
        with np.load(path / "predictions.npz", allow_pickle=False) as archive:
            predictions[name] = dict(archive)
    first, second = predictions["previous"], predictions["candidate"]
    keys = ("truth", "match_ids", "groups", "teams", "timestamps", "actors", "identified")
    if (
        any(not np.array_equal(first[k], second[k]) for k in keys)
        or reports["previous"]["dataset_sha256"] != reports["candidate"]["dataset_sha256"]
    ):
        raise ValueError("Holdout comparisons require exactly matching measured rows and data")
    if any(r["evaluation_mode"] != "new_matches" for r in reports.values()):
        raise ValueError("Holdout comparison requires unseen matches")
    result = {
        "models": {
            name: {"bundle": report["bundle"], "bundle_sha256": report["bundle_sha256"]}
            for name, report in reports.items()
        },
        "per_scope": [],
        "paired_differences": [],
        "matches": sorted(set(first["match_ids"])),
        "samples": samples,
        "meaning": "Fixed-product comparison on unseen European matches. Paired bootstrap resamples whole matches; recurring players can remain dependent. F1 measures recorded-activity agreement, not strategic quality or win rate. Multiple development changes prevent isolated causal attribution.",
    }
    identifiers, inverse = np.unique(first["match_ids"], return_inverse=True)
    for scope in ("TEAM", *ROLES):
        rows = (first["groups"] == GROUPS.index(scope)) & first["identified"]
        matrices = {}
        for name, predicted in predictions.items():
            matrices[name] = np.zeros((len(identifiers), len(Action), len(Action) + 1), dtype=int)
            np.add.at(
                matrices[name],
                (
                    inverse[rows],
                    first["truth"][rows],
                    np.where(
                        predicted["predicted"][rows] < 0, len(Action), predicted["predicted"][rows]
                    ),
                ),
                1,
            )
            metrics = reports[name]["per_group"][scope]
            result["per_scope"].append(
                {
                    "model": name,
                    "scope": scope,
                    "identified_rows": int(rows.sum()),
                    "identified_f1": (metrics["identified"] or {}).get("macro_f1"),
                    "identified_agreement": (metrics["identified"] or {}).get("behavior_agreement"),
                    "coverage": metrics["retention_coverage"],
                    "identified_majority_agreement": metrics["identified_majority_agreement"],
                    "bootstrap": metrics["identified_match_bootstrap"],
                }
            )
        rng = np.random.default_rng(20261110)
        differences = []
        for _ in range(samples):
            sampled = rng.integers(len(identifiers), size=len(identifiers))
            estimates = {}
            for name, matrix in matrices.items():
                aggregate = matrix[sampled].sum(0)
                denominator = aggregate.sum(1) + aggregate[:, :-1].sum(0)
                if not (denominator > 0).any():
                    break
                estimates[name] = np.divide(
                    2 * aggregate[:, :-1].diagonal(),
                    denominator,
                    out=np.zeros(len(Action)),
                    where=denominator > 0,
                )[denominator > 0].mean()
            if len(estimates) == 2:
                differences.append(estimates["candidate"] - estimates["previous"])
        result["paired_differences"].append(
            {
                "scope": scope,
                "point_difference": reports["candidate"]["per_group"][scope]["identified"][
                    "macro_f1"
                ]
                - reports["previous"]["per_group"][scope]["identified"]["macro_f1"]
                if rows.any()
                else None,
                "f1_difference_95": np.quantile(differences, [0.025, 0.975]).tolist()
                if differences
                else None,
                "valid_resamples": len(differences),
                "identified_rows": int(rows.sum()),
            }
        )
    atomic_json(Path(destination), result)
    return result
