"""Reproducible expansion pilot from verified cached participants, using workflow collection."""

from __future__ import annotations

import copy
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROLES = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")


def select_cohort(records, excluded, *, per_role=2, seed=20261104):
    """Uniform accounts within dominant-role strata, not proportional to appearances.

    These are verified payload PUUIDs, not guessed professional identities. This is
    snowball expansion of an elite KR frame, not a representative ranked-player sample.
    """
    from rift_common.pipeline.data_pipeline import REGIONS

    if per_role < 1:
        raise ValueError("Players per role must be positive")
    roles, sources, regions, displays = defaultdict(Counter), {}, {}, {}
    for _, payload in records:
        match = payload["match"]
        match_id = match["metadata"]["matchId"]
        region = REGIONS.get(match_id.split("_")[0].lower())
        if region is None:
            raise ValueError(f"Unknown routing prefix: {match_id}")
        for player in match["info"]["participants"]:
            puuid, role = player.get("puuid"), player.get("teamPosition")
            if not puuid or puuid in excluded or role not in ROLES:
                continue
            roles[puuid][role] += 1
            sources.setdefault(puuid, match_id)
            regions[puuid] = region
            displays[puuid] = (
                f"{player.get('riotIdGameName', '')}#{player.get('riotIdTagline', '')}"
            )
    strata = defaultdict(list)
    for puuid, counts in roles.items():
        dominant = max(ROLES, key=lambda role: counts[role])
        strata[dominant].append(puuid)
    rng = np.random.default_rng(seed)
    selected, sizes = [], {}
    for role in ROLES:
        candidates = sorted(strata[role])
        sizes[role] = len(candidates)
        for puuid in rng.permutation(candidates)[:per_role]:
            selected.append(
                {
                    "puuid": str(puuid),
                    "riot_id": displays[puuid],
                    "region": regions[puuid],
                    "sampling_role": role,
                    "verified_in_match": sources[puuid],
                    "selection_probability_within_frame_role": min(per_role, len(candidates))
                    / len(candidates),
                }
            )
    if not selected:
        raise ValueError("No eligible cached participants")
    return selected, sizes


def prepare_cohort(workflow, output, *, per_role=2, count=25, seed=20261104):
    if count < 1:
        raise ValueError("Match count must be positive")
    records = list(workflow.raw_records())
    selected, sizes = select_cohort(
        records, {player["puuid"] for player in workflow.players()}, per_role=per_role, seed=seed
    )
    return _write_cohort(
        workflow,
        output,
        selected,
        {
            "eligible_accounts_by_role": sizes,
            "sampling_frame": "cached configured-account co-participants; role-stratified accounts",
            "limitations": [
                "elite KR ego-network bias remains",
                "rank not verified",
                "roles are historical dominant roles, not guaranteed current roles",
                "selection probabilities apply only inside the observed participant frame",
            ],
        },
        records,
        count=count,
        seed=seed,
    )


def prepare_ranked_cohort(
    workflow, output, roster_dir, *, per_tier=15, count=20, seed=20261105, exclude_configs=()
):
    """Uniform PUUID sampling within frozen official KR top-tier roster responses."""
    from offline_rift.workflow import ResearchWorkflow
    from rift_common.artifacts import read_json

    if per_tier < 1 or count < 1:
        raise ValueError("Tier sample and history counts must be positive")
    excluded = {p["puuid"] for p in workflow.players()}
    for config in exclude_configs:
        excluded.update(p["puuid"] for p in ResearchWorkflow(config).players())
    selected, sizes, hashes = [], {}, {}
    import hashlib

    rng = np.random.default_rng(seed)
    for tier in ("MASTER", "GRANDMASTER", "CHALLENGER"):
        path = Path(roster_dir) / f"{tier.lower()}.json"
        entries = read_json(path)
        if not isinstance(entries, list) or any(not p.get("puuid") for p in entries):
            raise ValueError("Roster requires verified PUUIDs")
        candidates = sorted({p["puuid"] for p in entries} - excluded)
        if len(candidates) < per_tier:
            raise ValueError(f"Insufficient returned roster accounts for {tier}")
        sizes[tier] = len(candidates)
        hashes[tier] = hashlib.sha256(path.read_bytes()).hexdigest()
        for puuid in rng.permutation(candidates)[:per_tier]:
            selected.append(
                {
                    "puuid": str(puuid),
                    "riot_id": "",
                    "region": "asia",
                    "sampling_tier": tier,
                    "platform": "kr",
                    "selection_probability_within_returned_tier_roster": per_tier / len(candidates),
                }
            )
        excluded.update(p["puuid"] for p in selected)
    records = [(p, read_json(p)) for p in sorted(workflow.path("raw").glob("*.json"))]
    return _write_cohort(
        workflow,
        output,
        selected,
        {
            "eligible_accounts_by_current_tier": sizes,
            "roster_sha256": hashes,
            "roster_directory": str(Path(roster_dir).resolve()),
            "sampling_frame": "frozen official KR League-V4 returned rosters; uniform accounts by current tier",
            "limitations": [
                "KR top-tier coverage only",
                "current tier is not historical match rank",
                "returned roster may be API-truncated; probabilities are conditional on returned frame",
                "equal-tier allocation is not a population-frequency sample",
                "all-participant-disjointness is not guaranteed",
            ],
        },
        records,
        count=count,
        seed=seed,
    )


def _write_cohort(workflow, output, selected, frame, records, *, count, seed):
    from rift_common.artifacts import atomic_json

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = copy.deepcopy(workflow.config)
    config["name"] = "diversity-pilot"
    config["player"] = selected[0]
    config["additional_players"] = selected[1:]
    config["collection"]["count"] = count
    config["paths"] = {
        "raw": str(workflow.path("raw").resolve()),
        "catalogs": str(workflow.path("catalogs").resolve()),
        "artifacts": str(output / "artifacts"),
        "models": str(output / "models"),
    }
    manifest = {
        "status": "prepared",
        "seed": seed,
        "players": selected,
        "requested_matches_per_player": count,
        "source_config": str(workflow.config_path),
        "seed_match_ids": sorted(
            {payload["match"]["metadata"]["matchId"] for _, payload in records}
        ),
        **frame,
        "batches": [],
    }
    atomic_json(output / "workflow.json", config)
    atomic_json(output / "cohort.json", manifest)
    return manifest


def collect_cohort(config_path):
    """Resume the exact saved cohort; delegate retries/cache/progress to ResearchWorkflow."""
    from offline_rift.workflow import ResearchWorkflow
    from rift_common.artifacts import atomic_json, read_json

    workflow = ResearchWorkflow(config_path)
    manifest_path = Path(config_path).resolve().parent / "cohort.json"
    manifest = read_json(manifest_path)
    if not os.environ.get("RIOT_API_KEY"):
        manifest["status"] = "blocked_missing_credentials"
        atomic_json(manifest_path, manifest)
        return manifest
    manifest["status"] = "collecting"
    atomic_json(manifest_path, manifest)
    completed = {batch["puuid"] for batch in manifest["batches"]}
    try:
        for player in workflow.players():
            if player["puuid"] in completed:
                continue
            batch = workflow.collect(player=player["puuid"])
            manifest["batches"].append(
                {
                    "puuid": player["puuid"],
                    "batch_id": batch["batch_id"],
                    "downloaded": batch["downloaded"],
                    "reused": batch["reused"],
                }
            )
            atomic_json(manifest_path, manifest)
        manifest["status"] = "complete"
    except BaseException as error:
        manifest["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        manifest["error"] = str(error)
        raise
    finally:
        manifest["new_unique_downloads"] = len(
            {match for batch in manifest["batches"] for match in batch["downloaded"]}
        )
        atomic_json(manifest_path, manifest)
    return manifest


def add_cohort_arguments(parser):
    parser.add_argument("--output", type=Path, help="New pilot directory; never overwritten")
    parser.add_argument("--resume", type=Path, help="Existing pilot workflow.json; do not resample")
    parser.add_argument("--players-per-role", type=int, default=2)
    parser.add_argument("--count", type=int, default=25, help="Ranked matches per sampled account")
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--collect", action="store_true", help="Collect with RIOT_API_KEY")
    parser.add_argument(
        "--roster-dir",
        type=Path,
        help="Frozen official KR MASTER/GRANDMASTER/CHALLENGER JSON rosters",
    )
    parser.add_argument("--accounts-per-tier", type=int, default=15)
    parser.add_argument("--exclude-config", type=Path, action="append", default=[])


def run_cohort(args, config_path):
    from offline_rift.workflow import ResearchWorkflow

    if bool(args.output) == bool(args.resume):
        raise ValueError("Supply exactly one of --output or --resume")
    if args.resume:
        if not args.collect:
            raise ValueError("--resume requires --collect")
        pilot_config = args.resume
    else:
        if args.roster_dir:
            prepare_ranked_cohort(
                ResearchWorkflow(config_path),
                args.output,
                args.roster_dir,
                per_tier=args.accounts_per_tier,
                count=args.count,
                seed=args.seed,
                exclude_configs=args.exclude_config,
            )
        else:
            prepare_cohort(
                ResearchWorkflow(config_path),
                args.output,
                per_role=args.players_per_role,
                count=args.count,
                seed=args.seed,
            )
        pilot_config = args.output / "workflow.json"
    if args.collect:
        result = collect_cohort(pilot_config)
        print(
            f"Cohort status: {result['status']}; new unique downloads: "
            f"{result.get('new_unique_downloads', 0)}"
        )
    print(f"Pilot configuration: {pilot_config}")
