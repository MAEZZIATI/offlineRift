"""Resumable real-data collection, immutable dataset builds, and versioned training runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from offline_rift.pipeline.action_labeler import ACTION_NAMES, MDPDatasetBuilder, load_dataset
from offline_rift.pipeline.data_pipeline import REGIONS, RiotDataIngestor, TimelineProcessor
from offline_rift.pipeline.features import FeatureConfig, FeatureEncoder, ItemCatalog

LOG = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent


def atomic_json(path: Path, data: dict | list):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path):
    return json.loads(path.read_text())


def run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]


def patch_of(version: str) -> str:
    match = re.match(r"^(\d+)\.(\d+)\.", version)
    if not match:
        raise ValueError(f"Cannot determine patch from gameVersion={version!r}")
    return ".".join(match.groups())


class CatalogCache:
    """Resolve the newest published Data Dragon build within each match's patch."""

    def __init__(self, directory: Path, *, refresh: bool = False, session=None):
        self.directory, self.refresh = directory, refresh
        self.session = session or requests.Session()
        if session is None:
            retries = Retry(total=4, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
            self.session.mount("https://", HTTPAdapter(max_retries=retries))
        self.versions: list[str] | None = None
        self.resolved: dict[str, tuple[ItemCatalog, str, str]] = {}

    def _download(self, url):
        response = self.session.get(url, timeout=30)
        response.raise_for_status()
        return response.json()

    def get(self, patch: str) -> tuple[ItemCatalog, str, str]:
        if patch in self.resolved:
            return self.resolved[patch]
        path = self.directory / f"items-{patch}.json"
        if path.exists() and not self.refresh:
            data = read_json(path)
        else:
            if self.versions is None:
                self.versions = self._download(
                    "https://ddragon.leagueoflegends.com/api/versions.json"
                )
            versions = [v for v in self.versions if v.startswith(patch + ".")]
            if not versions:
                raise ValueError(f"No Data Dragon item catalog published for patch {patch}")
            version = max(versions, key=lambda v: tuple(int(part) for part in v.split(".")))
            LOG.info("Downloading patch %s item catalog (%s)", patch, version)
            data = self._download(
                f"https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/item.json"
            )
            if not data.get("data") or data.get("version") != version:
                raise ValueError(f"Invalid item catalog for {version}")
            atomic_json(path, data)
        if not data.get("data") or patch_of(data.get("version", "")) != patch:
            raise ValueError(f"Cached catalog {path} does not match patch {patch}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.resolved[patch] = ItemCatalog(data), data["version"], digest
        return self.resolved[patch]

    def close(self):
        self.session.close()


class ResearchWorkflow:
    def __init__(self, config_path: str | Path = "workflow.json"):
        self.config_path = Path(config_path).resolve()
        self.base = self.config_path.parent
        self.config = read_json(self.config_path)
        if self.config.get("version") != 1:
            raise ValueError("Unsupported workflow configuration version")

    def path(self, name: str) -> Path:
        return self.base / self.config["paths"][name]

    @property
    def state_path(self):
        return self.path("artifacts") / "latest.json"

    def state(self) -> dict:
        return read_json(self.state_path) if self.state_path.exists() else {}

    @classmethod
    def initialize(
        cls,
        config_path: Path,
        *,
        name: str,
        raw_dir: str = "data/raw",
        puuid: str | None = None,
        region: str | None = None,
    ):
        if config_path.exists():
            raise ValueError(f"{config_path} already exists; edit it to change settings")
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
            raise ValueError("Project name may contain letters, numbers, underscores, and hyphens")
        files = sorted((config_path.resolve().parent / raw_dir).glob("*.json"))
        counts, identities = Counter(), {}
        for path in files:
            payload = read_json(path)
            for p in payload["match"]["info"]["participants"]:
                if p.get("puuid"):
                    counts[p["puuid"]] += 1
                    identities[p["puuid"]] = (
                        f"{p.get('riotIdGameName', '')}#{p.get('riotIdTagline', '')}"
                    )
        if not puuid:
            common = [key for key, count in counts.items() if count == len(files)]
            if not files or len(common) != 1:
                raise ValueError("Cannot infer one common player; initialize with --puuid")
            puuid = common[0]
        if not region:
            prefixes = {path.stem.split("_")[0].lower() for path in files}
            inferred = {REGIONS.get(prefix, prefix) for prefix in prefixes}
            if len(inferred) != 1:
                raise ValueError("Provide --region when the raw files do not identify one region")
            region = inferred.pop()
        region = REGIONS.get(region.lower(), region.lower())
        if region not in {"americas", "asia", "europe", "sea"}:
            raise ValueError("Invalid region")
        config = {
            "version": 1,
            "name": name,
            "player": {"puuid": puuid, "riot_id": identities.get(puuid, ""), "region": region},
            "paths": {
                "raw": raw_dir,
                "catalogs": "data/catalogs",
                "artifacts": f"data/projects/{name}",
                "models": f"models/{name}",
            },
            "collection": {"count": 100, "queue": 420},
            "features": FeatureConfig().to_dict(),
            "feature_overrides_by_patch": {},
            "reward_mode": "sparse",
            "training": {
                "n_steps": 10000,
                "n_steps_per_epoch": 1000,
                "batch_size": 64,
                "device": "cpu",
                "cpu_threads": 2,
                "alpha": 1.0,
                "gamma": 0.99,
                "seed": 42,
                "validation_fraction": 0.2,
            },
        }
        atomic_json(config_path, config)
        return cls(config_path)

    def raw_records(self):
        """Only this player's selected queue; deduplicate by embedded match ID."""
        seen = set()
        for path in sorted(self.path("raw").glob("*.json")):
            payload = read_json(path)
            match = payload["match"]
            info = match["info"]
            if not any(
                p.get("puuid") == self.config["player"]["puuid"] for p in info["participants"]
            ):
                continue
            queue = self.config["collection"]["queue"]
            if queue is not None and info.get("queueId") != queue:
                continue
            match_id = match["metadata"]["matchId"]
            if match_id in seen:
                continue
            if payload["timeline"].get("metadata", {}).get("matchId", match_id) != match_id:
                raise ValueError(f"Timeline/match ID mismatch: {path}")
            seen.add(match_id)
            yield path, payload

    def collect(self, *, count: int | None = None, start: int = 0, api=None) -> dict:
        count = self.config["collection"]["count"] if count is None else count
        if count <= 0 or start < 0:
            raise ValueError("count must be positive and start nonnegative")
        own_api = api is None
        api = api or RiotDataIngestor(
            os.environ.get("RIOT_API_KEY", ""), self.config["player"]["region"]
        )
        batch_id = run_id()
        manifest = {
            "batch_id": batch_id,
            "status": "running",
            "start": start,
            "requested_count": count,
            "discovered": [],
            "downloaded": [],
            "reused": [],
        }
        manifest_path = self.path("artifacts") / "batches" / f"{batch_id}.json"
        atomic_json(manifest_path, manifest)
        try:
            for offset in range(0, count, 100):
                ids = api.fetch_match_ids(
                    self.config["player"]["puuid"],
                    start=start + offset,
                    count=min(100, count - offset),
                    queue=self.config["collection"]["queue"],
                )
                for match_id in ids:
                    if not re.fullmatch(r"[A-Za-z0-9_-]+", match_id):
                        raise ValueError("Invalid match ID")
                    if match_id in manifest["discovered"]:
                        continue
                    manifest["discovered"].append(match_id)
                    path = self.path("raw") / f"{match_id}.json"
                    cached = None
                    if path.exists():
                        try:
                            cached = read_json(path)
                            if (
                                cached["match"]["metadata"]["matchId"] != match_id
                                or not cached["timeline"]["info"]["frames"]
                            ):
                                cached = None
                        except (KeyError, TypeError, ValueError):
                            cached = None
                    if cached is None:
                        payload = {
                            "match": api.fetch_match(match_id),
                            "timeline": api.fetch_timeline(match_id),
                        }
                        atomic_json(path, payload)
                        manifest["downloaded"].append(match_id)
                    else:
                        manifest["reused"].append(match_id)
                    atomic_json(manifest_path, manifest)
                    LOG.info(
                        "Collection: %d downloaded, %d cached",
                        len(manifest["downloaded"]),
                        len(manifest["reused"]),
                    )
                if len(ids) < min(100, count - offset):
                    break
            manifest["status"] = "complete"
            return manifest
        except BaseException as error:
            manifest["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            manifest["error"] = str(error)
            raise
        finally:
            atomic_json(manifest_path, manifest)
            if own_api:
                api.close()

    def build(self, *, refresh_catalogs=False, cache=None) -> dict:
        own_cache = cache is None
        cache = cache or CatalogCache(self.path("catalogs"), refresh=refresh_catalogs)
        try:
            return self._build(cache)
        finally:
            if own_cache:
                cache.close()

    def _build(self, cache) -> dict:
        records = [
            (path, payload["match"]["info"]["gameVersion"]) for path, payload in self.raw_records()
        ]
        if not records:
            raise ValueError("No cached matches for the configured player/queue; run collect first")
        patches = sorted(
            {patch_of(v) for _, v in records}, key=lambda p: tuple(map(int, p.split(".")))
        )
        catalogs = {patch: cache.get(patch) for patch in patches}
        digest = hashlib.sha256()
        digest.update(
            json.dumps(
                {
                    "features": self.config["features"],
                    "overrides": self.config["feature_overrides_by_patch"],
                    "reward": self.config["reward_mode"],
                    "catalogs": {p: v[1:] for p, v in catalogs.items()},
                },
                sort_keys=True,
            ).encode()
        )
        for module in (
            "workflow.py",
            "pipeline/features.py",
            "pipeline/data_pipeline.py",
            "pipeline/action_labeler.py",
        ):
            digest.update((ROOT / module).read_bytes())
        for path, _ in records:
            digest.update(path.read_bytes())
        fingerprint = digest.hexdigest()
        state = self.state()
        old = state.get("dataset", {})
        if old.get("fingerprint") == fingerprint and Path(old.get("path", "")).is_file():
            LOG.info("Dataset unchanged: reusing %s", old["path"])
            return old
        latest_catalog = catalogs[patches[-1]][0]
        latest_options = {
            **self.config["features"],
            **self.config["feature_overrides_by_patch"].get(patches[-1], {}),
        }
        builder = MDPDatasetBuilder(
            FeatureConfig(**latest_options),
            latest_catalog,
            reward_mode=self.config["reward_mode"],
        )
        skipped, accepted, patch_counts = [], [], Counter()
        for index, (path, version) in enumerate(records, 1):
            payload = read_json(path)
            patch = patch_of(version)
            options = {
                **self.config["features"],
                **self.config["feature_overrides_by_patch"].get(patch, {}),
            }
            config = FeatureConfig(**options)
            if (
                config.profile != builder.config.profile
                or FeatureEncoder(config, latest_catalog).names != builder.encoder.names
            ):
                raise ValueError("Patch overrides cannot change the feature schema or profile")
            try:
                snapshots = TimelineProcessor(config).parse(payload["match"], payload["timeline"])
                builder.add_match(payload["match"], snapshots, catalog=catalogs[patch][0])
            except (ValueError, KeyError, TypeError) as error:
                skipped.append({"file": str(path), "reason": str(error)})
                LOG.warning("Skipped %s: %s", path.name, error)
                continue
            accepted.append(payload["match"]["metadata"]["matchId"])
            patch_counts[patch] += 1
            if index % 10 == 0:
                LOG.info("Processed %d/%d matches", index, len(records))
        dataset_id = f"{self.config['name']}-{fingerprint[:16]}"
        directory = self.path("artifacts") / "datasets" / dataset_id
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "transitions.npz"
        # If processing is interrupted, no partial dataset or latest pointer is published.
        temporary = directory / f".{uuid.uuid4().hex}.npz"
        provenance = {
            "fingerprint": fingerprint,
            "patches": dict(patch_counts),
            "catalog_versions": {patch: catalog[1] for patch, catalog in catalogs.items()},
            "patch_feature_overrides": self.config["feature_overrides_by_patch"],
        }
        try:
            builder.save(temporary, provenance=provenance)
            arrays, _ = load_dataset(temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        report = {
            "dataset_id": dataset_id,
            "path": str(path.resolve()),
            "fingerprint": fingerprint,
            "matches": len(accepted),
            "episodes": len(set(arrays["episode_ids"])),
            "transitions": len(arrays["actions"]),
            "low_confidence_label_fraction": float(np.mean(arrays["label_confidence"] < 0.5)),
            "action_counts": {
                ACTION_NAMES[i]: int(n)
                for i, n in enumerate(np.bincount(arrays["actions"], minlength=5))
            },
            "match_ids": accepted,
            "skipped": skipped,
            **provenance,
        }
        atomic_json(directory / "report.json", report)
        state["dataset"] = report
        atomic_json(self.state_path, state)
        return report

    def train(self, dataset: dict, *, force=False) -> dict:
        from offline_rift.training.rl_agent import OfflineMacroTrainer

        options = self.config["training"].copy()
        signature = hashlib.sha256(
            json.dumps(
                {"dataset": dataset["fingerprint"], "training": options}, sort_keys=True
            ).encode()
        )
        signature.update((ROOT / "training/rl_agent.py").read_bytes())
        fingerprint = signature.hexdigest()
        state = self.state()
        old = state.get("model", {})
        if (
            not force
            and old.get("fingerprint") == fingerprint
            and (Path(old.get("path", "")) / "policy.d3").is_file()
            and (Path(old["path"]) / "metadata.json").is_file()
        ):
            LOG.info("Training unchanged: reusing %s", old["path"])
            return old
        import torch

        threads = options.pop("cpu_threads")
        if threads < 1:
            raise ValueError("cpu_threads must be >= 1")
        torch.set_num_threads(threads)
        output = self.path("models") / run_id()
        settings = {k: options.pop(k) for k in ("device", "batch_size", "alpha", "gamma", "seed")}
        output.mkdir(parents=True, exist_ok=False)
        manifest = {
            "path": str(output.resolve()),
            "fingerprint": fingerprint,
            "dataset_id": dataset["dataset_id"],
            "dataset_path": dataset["path"],
            "settings": self.config["training"],
            "status": "running",
        }
        atomic_json(output / "run.json", manifest)
        try:
            trainer = OfflineMacroTrainer(**settings)
            report = trainer.train(dataset["path"], output, **options)
            manifest.update(status="complete", report=report)
        except BaseException as error:
            manifest.update(
                status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                error=str(error),
            )
            raise
        finally:
            atomic_json(output / "run.json", manifest)
        state["model"] = manifest
        atomic_json(self.state_path, state)
        return manifest

    def status(self) -> dict:
        patches, matches = Counter(), 0
        for _, payload in self.raw_records():
            matches += 1
            patches[patch_of(payload["match"]["info"]["gameVersion"])] += 1
        state = self.state()
        return {
            "project": self.config["name"],
            "player": self.config["player"]["riot_id"],
            "region": self.config["player"]["region"],
            "cached_matches": matches,
            "patches": dict(patches),
            "dataset": state.get("dataset"),
            "model": state.get("model"),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("workflow.json"))
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="Adopt existing raw matches and save editable defaults")
    init.add_argument("--name", default="macro")
    init.add_argument("--raw-dir", default="data/raw")
    init.add_argument("--puuid")
    init.add_argument("--region")
    sub.add_parser("status", help="Show cached data and latest completed artifacts")
    collect = sub.add_parser("collect", help="Fetch history, skipping cached match payloads")
    collect.add_argument("--count", type=int)
    collect.add_argument("--start", type=int, default=0)
    build = sub.add_parser("build", help="Build a versioned dataset from all cached matches")
    build.add_argument("--refresh-catalogs", action="store_true")
    train = sub.add_parser("train", help="Build if needed and train if data/settings changed")
    train.add_argument("--retrain", action="store_true")
    run = sub.add_parser("run", help="Build and train; optionally collect the latest games first")
    run.add_argument("--refresh", action="store_true")
    run.add_argument("--refresh-catalogs", action="store_true")
    run.add_argument("--retrain", action="store_true")
    live = sub.add_parser("live", help="Poll using the latest completed live-compatible policy")
    live.add_argument("--team", type=int, choices=(100, 200), default=100)
    live.add_argument("--interval", type=float, default=10)
    live.add_argument("--once", action="store_true")
    live.add_argument("--ca-bundle")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    try:
        if args.command == "init":
            workflow = ResearchWorkflow.initialize(
                args.config,
                name=args.name,
                raw_dir=args.raw_dir,
                puuid=args.puuid,
                region=args.region,
            )
            print(f"Initialized {workflow.config['name']} in {args.config}")
            return
        workflow = ResearchWorkflow(args.config)
        if args.command == "status":
            status = workflow.status()
            # Detailed reports remain in their immutable artifact directories.
            if status["dataset"]:
                status["dataset"] = {
                    k: status["dataset"][k]
                    for k in ("path", "matches", "transitions", "action_counts")
                }
            if status["model"]:
                status["model"] = {k: status["model"][k] for k in ("path", "status", "dataset_id")}
            print(json.dumps(status, indent=2))
        elif args.command == "collect":
            batch = workflow.collect(count=args.count, start=args.start)
            print(
                f"Batch {batch['batch_id']}: {len(batch['downloaded'])} new, "
                f"{len(batch['reused'])} cached"
            )
        elif args.command == "live":
            from offline_rift.inference.live_client import LiveClientInference

            model = workflow.state().get("model")
            if not model:
                raise ValueError("No completed model; run workflow.py run first")
            LiveClientInference(
                model["path"],
                team=args.team,
                interval_seconds=args.interval,
                verify_tls=args.ca_bundle or False,
            ).run(max_polls=1 if args.once else None)
        else:
            if getattr(args, "refresh", False):
                workflow.collect()
            dataset = workflow.build(refresh_catalogs=getattr(args, "refresh_catalogs", False))
            print(
                f"Dataset: {dataset['matches']} matches, {dataset['transitions']} transitions\n"
                f"{dataset['path']}",
                flush=True,
            )
            if args.command != "build":
                model = workflow.train(dataset, force=getattr(args, "retrain", False))
                print(f"Policy: {model['path']}")
    except (ValueError, KeyError, OSError, requests.RequestException, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
