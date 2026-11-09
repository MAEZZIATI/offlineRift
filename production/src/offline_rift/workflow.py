"""One workflow for collection, versioned next-action datasets, training and live use."""

import argparse
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
from rift_common import actions, behavior, dataset, items, objectives, state, timeline
from rift_common.artifacts import atomic_json, file_digest, read_json, run_id
from rift_common.dataset import DatasetBuilder, load_dataset

from offline_rift.catalogs import CatalogCache, patch_of
from offline_rift.collection import Collector, RiotClient


class Workflow:
    def __init__(self, path="workflow.json"):
        self.config_path = Path(path).resolve()
        self.base = self.config_path.parent
        self.config = read_json(self.config_path)
        if self.config.get("version") != 2:
            raise ValueError(
                "Use a version-2 next-action configuration; legacy formation workflows are incompatible"
            )
        players = self.config["players"]
        if (
            not players
            or len({p["puuid"] for p in players}) != len(players)
            or not any(p["puuid"] == self.config["primary_puuid"] for p in players)
        ):
            raise ValueError("Configure distinct verified players and a primary PUUID")

    def path(self, name):
        return self.base / self.config["paths"][name]

    @property
    def state_path(self):
        return self.path("artifacts") / "latest.json"

    def state(self):
        return read_json(self.state_path) if self.state_path.exists() else {}

    def verify(self, *, model=None):
        """Read-only artifact/replay checks, not a certification of strategic utility."""
        import torch
        from rift_common.actions import ACTION_NAMES, GROUPS
        from rift_common.metrics import behavior_metrics

        from offline_rift.inference.coach import load_policy

        stored = self.state()
        directory = model or stored.get("model", {}).get("path")
        if not directory:
            raise ValueError("No next-action model; use rift train")
        directory = Path(directory)
        policy = load_policy(directory)
        run = read_json(directory / "run.json")
        if run.get("status") != "complete":
            raise ValueError("Model run is not complete")
        data = Path(run["dataset_path"])
        digest = file_digest(data)
        if digest != run["dataset_sha256"] or digest != policy.metadata.get(
            "dataset_sha256", digest
        ):
            raise ValueError("Frozen dataset integrity mismatch")
        arrays, metadata = load_dataset(data)
        if metadata["provenance"] != policy.metadata["provenance"]:
            raise ValueError("Frozen dataset provenance mismatch")
        report = policy.metadata["training_report"]
        training, validation = set(report["training_matches"]), set(report["validation_matches"])
        if (
            not training
            or not validation
            or training & validation
            or training | validation | set(report.get("holdout_matches", []))
            != set(arrays["match_ids"])
        ):
            raise ValueError("Invalid frozen match split")
        selected = np.isin(arrays["match_ids"], list(validation))
        observed, groups, masks = (
            arrays[key][selected] for key in ("observations", "groups", "action_masks")
        )
        torch.set_num_threads(2)
        predicted, _ = policy.predict_batch(observed, groups, masks)
        metrics = behavior_metrics(arrays["actions"][selected], predicted, len(ACTION_NAMES))
        if metrics != report["validation"]:
            raise ValueError("Serving predictions do not reproduce training evaluation")
        per_group = {}
        for i, group in enumerate(GROUPS):
            group_rows = groups == i
            if not group_rows.any():
                continue
            choices = predicted[group_rows]
            counts = np.bincount(choices[choices >= 0], minlength=len(ACTION_NAMES))
            per_group[group] = {
                "training_action_support": policy.counts[i].tolist(),
                "behavior_agreement": float(
                    np.mean(choices == arrays["actions"][selected][group_rows])
                ),
                "majority_baseline": float(
                    np.mean(arrays["actions"][selected][group_rows] == policy.counts[i].argmax())
                ),
                "prediction_counts": counts.tolist(),
                "retention_coverage": float(np.mean(choices >= 0)),
                "largest_prediction_fraction": float(counts.max() / max(1, counts.sum())),
            }
        elapsed = []
        for i in range(55):
            row = i % len(observed)
            start = time.perf_counter()
            policy.predict_batch(
                observed[row : row + 1], groups[row : row + 1], masks[row : row + 1]
            )
            if i >= 5:
                elapsed.append((time.perf_counter() - start) * 1000)
        return {
            "status": "verified",
            "model": str(directory.resolve()),
            "bundle_sha256": file_digest(
                directory / ("coach.json" if (directory / "coach.json").exists() else "bundle.json")
            ),
            "dataset_sha256": digest,
            "serving_replay_verified": True,
            "validation_matches": len(validation),
            "validation_transitions": int(selected.sum()),
            "metrics": metrics,
            "per_group": per_group,
            "single_state_latency_ms": {
                "samples": len(elapsed),
                "warmup": 5,
                "cpu_threads": torch.get_num_threads(),
                "p50": float(np.median(elapsed)),
                "p95": float(np.quantile(elapsed, 0.95)),
                "scope": "CPU encoded-state inference, excluding game transport and UI",
            },
            "stage": "experimental_next_action_policy",
            "production_strategy_validated": False,
            "limitations": [
                "Artifact integrity and replay parity do not establish useful or optimal advice",
                "Independent new-match and active-game acceptance remain necessary",
                "No prospective victory-rate improvement has been established",
            ],
        }

    def collect(self, *, count=100, start=0, player=None, all_players=False, api=None):
        if player and all_players:
            raise ValueError("Choose one player or all players")
        players = self.config["players"]
        if not all_players:
            identity = player or self.config["primary_puuid"]
            players = [p for p in players if identity in {p["puuid"], p.get("riot_id", "")}]
            if len(players) != 1:
                raise ValueError("Select a configured PUUID or Riot ID")
        collector = Collector(self.path("raw"), self.path("artifacts"))
        return [collector.collect(p, count=count, start=start, queue=420, api=api) for p in players]

    def activate(self, model):
        """Publish a verified experimental acceptance candidate, preserving pointer history."""
        checked = self.verify(model=model)
        directory = Path(model).resolve()
        metadata = read_json(directory / "metadata.json")
        latest = self.state()
        activated = {
            "path": str(directory),
            "fingerprint": checked["bundle_sha256"],
            "algorithm": metadata["algorithm"],
            "status": "complete",
            "dataset_id": "next-actions-" + metadata["provenance"]["fingerprint"][:16],
            "stage": "offline_six_policy_candidate"
            if (directory / "coach.json").exists()
            else "ranked_acceptance_candidate",
        }
        atomic_json(
            self.path("artifacts") / "activations" / (run_id() + ".json"),
            {
                "previous": latest.get("model"),
                "activated": activated,
                "serving_replay_verified": checked["serving_replay_verified"],
                "strategic_utility_validated": False,
            },
        )
        latest["model"] = activated
        atomic_json(self.state_path, latest)
        return activated

    def raw_records(self):
        identities = {p["puuid"] for p in self.config["players"]}
        seen = set()
        for path in sorted(self.path("raw").glob("*.json")):
            payload = read_json(path)
            match = payload["match"]
            info = match["info"]
            if info.get("queueId") != 420 or not any(
                p.get("puuid") in identities for p in info["participants"]
            ):
                continue
            match_id = match["metadata"]["matchId"]
            if match_id in seen:
                continue
            seen.add(match_id)
            yield path, payload

    def build(self, *, cache=None, match_ids=None):
        owns = cache is None
        cache = cache or CatalogCache(self.path("catalogs"))
        try:
            return self._build(cache, match_ids=match_ids)
        finally:
            if owns:
                cache.close()

    def _build(self, cache, *, match_ids=None):
        records = []
        patches = Counter()
        for path, payload in self.raw_records():
            if match_ids is not None and payload["match"]["metadata"]["matchId"] not in match_ids:
                continue
            version = payload["match"]["info"]["gameVersion"]
            objectives.ObjectiveRules.validate_patch(version)
            patch = patch_of(version)
            patches[patch] += 1
            records.append(
                (path, patch, payload["match"]["metadata"]["matchId"], file_digest(path))
            )
        if not records:
            raise ValueError("No cached matches for the configured players; use rift collect")
        catalogs = {patch: cache.get(patch) for patch in patches}
        signature = hashlib.sha256(
            json.dumps(
                {
                    "inputs": [(r[2], r[3]) for r in records],
                    "catalogs": {p: v[1:] for p, v in catalogs.items()},
                    "minimum_time": 90,
                },
                sort_keys=True,
            ).encode()
        )
        for module in (actions, behavior, dataset, items, objectives, state, timeline):
            signature.update(Path(module.__file__).read_bytes())
        signature.update(Path(objectives.__file__).with_name("objective_rules.json").read_bytes())
        fingerprint = signature.hexdigest()
        latest = self.state()
        old = latest.get("dataset", {})
        if old.get("fingerprint") == fingerprint and Path(old.get("path", "")).is_file():
            load_dataset(old["path"])
            return old
        dataset_id = "next-actions-" + fingerprint[:16]
        directory = self.path("artifacts") / "datasets" / dataset_id
        if directory.exists():
            directory = directory.parent / (dataset_id + "-" + run_id())
        directory.mkdir(parents=True, exist_ok=False)
        report = {
            "dataset_id": dataset_id,
            "fingerprint": fingerprint,
            "status": "building",
            "path": str((directory / "transitions.npz").resolve()),
            "patches": dict(patches),
            "accepted_matches": [],
            "skipped": [],
        }
        atomic_json(directory / "report.json", report)
        builder = DatasetBuilder()
        try:
            for path, patch, match_id, _ in records:
                payload = read_json(path)
                try:
                    builder.add_match(payload["match"], payload["timeline"], catalogs[patch][0])
                except (ValueError, KeyError, TypeError) as error:
                    report["skipped"].append({"match_id": match_id, "reason": str(error)})
                    continue
                report["accepted_matches"].append(match_id)
            latest_patch = max(catalogs, key=lambda p: tuple(map(int, p.split("."))))
            builder.save(
                directory / "transitions.npz",
                catalog=catalogs[latest_patch][0],
                provenance={
                    "fingerprint": fingerprint,
                    "input_sha256": {r[2]: r[3] for r in records},
                    "source_players": self.config["players"],
                    "catalog_versions": {p: v[1] for p, v in catalogs.items()},
                    "objective_rules": objectives.SPEC["version"],
                },
            )
            arrays, _ = load_dataset(directory / "transitions.npz")
            report.update(
                status="complete",
                matches=len(builder.matches),
                episodes=len(set(arrays["episode_ids"])),
                transitions=len(arrays["actions"]),
                action_counts=np.bincount(
                    arrays["actions"], minlength=len(actions.Action)
                ).tolist(),
                excluded=dict(builder.excluded),
                behavior_evidence=dict(builder.evidence),
            )
        except BaseException as error:
            report.update(
                status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                error=str(error),
            )
            raise
        finally:
            atomic_json(directory / "report.json", report)
        latest["dataset"] = report
        atomic_json(self.state_path, latest)
        return report

    def train(self, data, *, algorithm=None, steps=None, force=False):
        from offline_rift.inference.policy import NextActionPolicy
        from offline_rift.training.trainer import PolicyTrainer, TrainConfig

        settings = {**self.config.get("training", {})}
        if algorithm is not None:
            settings["algorithm"] = algorithm
        if steps is not None:
            settings["steps"] = steps
        config = TrainConfig(**settings)
        latest = self.state()
        old = latest.get("model", {})
        from rift_common import network, observation_masks

        from offline_rift.training import augmentation, iql, targets, trainer

        signature = hashlib.sha256(
            json.dumps(
                {"dataset": data["fingerprint"], "settings": settings}, sort_keys=True
            ).encode()
        )
        for module in (network, trainer, targets, augmentation, observation_masks):
            signature.update(Path(module.__file__).read_bytes())
        if config.algorithm == "iql":
            signature.update(Path(iql.__file__).read_bytes())
        fingerprint = signature.hexdigest()
        if (
            not force
            and old.get("fingerprint") == fingerprint
            and (Path(old["path"]) / "bundle.json").is_file()
        ):
            NextActionPolicy(old["path"])
            return old
        output = self.path("models") / run_id()
        result = PolicyTrainer(config).train(data["path"], output)
        # A completed optimizer run is an experimental candidate, not proof of policy utility.
        NextActionPolicy(output)
        model = {
            "path": str(output.resolve()),
            "fingerprint": fingerprint,
            "algorithm": config.algorithm,
            "status": result["status"],
            "dataset_id": data["dataset_id"],
            "stage": "experimental_next_action_policy",
        }
        latest["model"] = model
        atomic_json(self.state_path, latest)
        return model

    def add_player(self, riot_id, region, *, api=None):
        import os

        own = api is None
        api = api or RiotClient(os.environ.get("RIOT_API_KEY", ""), region)
        try:
            account = api.resolve_account(riot_id)
        finally:
            if own:
                api.close()
        if any(p["puuid"] == account["puuid"] for p in self.config["players"]):
            raise ValueError("Player already configured")
        player = {
            "puuid": account["puuid"],
            "riot_id": account["gameName"] + "#" + account["tagLine"],
            "region": region,
        }
        self.config["players"].append(player)
        atomic_json(self.config_path, self.config)
        return player


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("workflow.json"))
    parser.add_argument(
        "--env-file", type=Path, help="Load only RIOT_API_KEY from this dotenv file"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="Initialize a verified-player next-action project")
    init.add_argument("--puuid", required=True)
    init.add_argument("--region", required=True)
    init.add_argument("--name", default="rift")
    sub.add_parser("status", help="Show configured players and latest next-action artifacts")
    verify = sub.add_parser(
        "verify", help="Verify a completed model, frozen replay and local inference latency"
    )
    verify.add_argument("--model", type=Path)
    activate = sub.add_parser(
        "activate", help="Activate a verified experimental candidate and retain pointer history"
    )
    activate.add_argument("--model", type=Path, required=True)
    collect = sub.add_parser("collect", help="Collect real Match-V5 data, reusing cached matches")
    collect.add_argument("--count", type=int, default=100)
    collect.add_argument("--start", type=int, default=0)
    collect.add_argument("--player")
    collect.add_argument("--all-players", action="store_true")
    add = sub.add_parser("add-player")
    add.add_argument("--riot-id", required=True)
    add.add_argument("--region", required=True)
    sub.add_parser("build", help="Build explicit actor/role next-action transitions from cache")
    coach = sub.add_parser(
        "train-coach", help="Train and compare the macro coach and all five role specialists"
    )
    coach.add_argument("--steps", type=int, default=10000)
    coach.add_argument("--output", type=Path, required=True)
    coach.add_argument("--holdout-manifest", type=Path, required=True)
    for command in ("train", "run"):
        child = sub.add_parser(command, help="Build and train an experimental next-action policy")
        child.add_argument("--algorithm", choices=("cql", "bc", "iql"))
        child.add_argument("--steps", type=int)
        child.add_argument("--retrain", action="store_true")
    for command in ("ui", "live"):
        child = sub.add_parser(command)
        child.add_argument("--model", type=Path)
        child.add_argument("--team", type=int, choices=(100, 200), default=100)
        child.add_argument("--actor", default="active")
        child.add_argument("--role", choices=actions.ROLES)
        child.add_argument("--interval", type=float, default=10)
        child.add_argument("--transport", choices=("auto", "local", "windows"), default="auto")
        child.add_argument(
            "--capture-session",
            type=Path,
            help="Opt in to measured live input and advice recording",
        )
        if command == "ui":
            child.add_argument("--port", type=int, default=8765)
        else:
            child.add_argument("--once", action="store_true")
            child.add_argument(
                "--capture-availability",
                type=Path,
                help="Opt in to aggregate availability counts in a new directory",
            )
    args = parser.parse_args()
    try:
        if args.env_file is not None:
            from offline_rift.collection import load_api_key

            load_api_key(args.env_file)
        if args.command == "init":
            if args.config.exists():
                raise ValueError("Configuration already exists; no overwrite")
            from offline_rift.collection import REGIONS

            region = REGIONS.get(args.region.lower(), args.region.lower())
            if (
                region not in {"asia", "americas", "europe", "sea"}
                or not args.puuid
                or not re.fullmatch(r"[a-zA-Z0-9_-]+", args.name)
            ):
                raise ValueError(
                    "Valid verified PUUID, regional route and safe project name required"
                )
            atomic_json(
                args.config,
                {
                    "version": 2,
                    "name": args.name,
                    "primary_puuid": args.puuid,
                    "players": [{"puuid": args.puuid, "region": region, "riot_id": ""}],
                    "paths": {
                        "raw": "data/raw",
                        "catalogs": "data/catalogs",
                        "artifacts": f"data/projects/{args.name}/next-actions",
                        "models": f"models/next-actions/{args.name}",
                    },
                    "training": {},
                },
            )
            print("Initialized", args.config)
            return
        workflow = Workflow(args.config)
        if args.command == "status":
            stored = workflow.state()
            current_dataset = stored.get("dataset", {})
            print(
                json.dumps(
                    {
                        "project": workflow.config["name"],
                        "players": len(workflow.config["players"]),
                        "dataset": {
                            key: current_dataset[key]
                            for key in ("path", "matches", "episodes", "transitions", "status")
                            if key in current_dataset
                        },
                        "model": stored.get("model"),
                    },
                    indent=2,
                )
            )
        elif args.command == "verify":
            print(json.dumps(workflow.verify(model=args.model), indent=2))
        elif args.command == "activate":
            print(json.dumps(workflow.activate(args.model), indent=2))
        elif args.command == "collect":
            print(
                json.dumps(
                    workflow.collect(
                        count=args.count,
                        start=args.start,
                        player=args.player,
                        all_players=args.all_players,
                    ),
                    indent=2,
                )
            )
        elif args.command == "add-player":
            print(json.dumps(workflow.add_player(args.riot_id, args.region)))
        elif args.command == "train-coach":
            from offline_rift.training.coach import train_coach

            data = workflow.build()
            reserved = read_json(args.holdout_manifest).get("holdout_matches", [])
            result = train_coach(
                data["path"], args.output, steps=args.steps, holdout_matches=reserved
            )
            print(json.dumps({"status": result["status"], "bundle": result["bundle"]}))
        elif args.command in {"build", "train", "run"}:
            data = workflow.build()
            print(
                f"Dataset: {data['matches']} matches, {data['transitions']} option transitions",
                flush=True,
            )
            if args.command != "build":
                print(
                    "Policy:",
                    workflow.train(
                        data, algorithm=args.algorithm, steps=args.steps, force=args.retrain
                    )["path"],
                )
        else:
            model = args.model or workflow.state().get("model", {}).get("path")
            if not model:
                raise ValueError("No next-action model; use rift train")
            if args.command == "ui":
                from offline_rift.ui.server import run_ui

                run_ui(
                    model,
                    team=args.team,
                    actor=args.actor,
                    role=args.role,
                    interval=args.interval,
                    port=args.port,
                    transport=args.transport,
                    capture_session=args.capture_session,
                )
            else:
                from offline_rift.inference.live_client import LiveClientInference

                LiveClientInference(
                    model,
                    team=args.team,
                    actor=args.actor,
                    role=args.role,
                    interval_seconds=args.interval,
                    transport=args.transport,
                    capture_availability=args.capture_availability,
                    capture_session=args.capture_session,
                ).run(max_polls=1 if args.once else None)
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted; existing data and progress reports are retained.\n")
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")
