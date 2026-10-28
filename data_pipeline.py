"""Riot Match-V5 ingestion and causal 10-second timeline reconstruction."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import random
import time
from collections import Counter
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import quote

import requests

from features import FeatureConfig, ItemCatalog, ObjectiveTracker, Snapshot

LOG = logging.getLogger(__name__)
REGIONS = {
    **dict.fromkeys(("na1", "br1", "la1", "la2"), "americas"),
    **dict.fromkeys(("euw1", "eun1", "tr1", "ru"), "europe"),
    **dict.fromkeys(("kr", "jp1"), "asia"),
    **dict.fromkeys(("oc1", "ph2", "sg2", "th2", "tw2", "vn2"), "sea"),
}


class RiotAPIError(RuntimeError):
    pass


class RiotDataIngestor:
    def __init__(
        self,
        api_key: str,
        region: str = "europe",
        *,
        session: requests.Session | None = None,
        timeout: float = 15,
        max_retries: int = 5,
        max_retry_wait: float = 120,
        sleep=time.sleep,
    ):
        region = REGIONS.get(region.lower(), region.lower())
        if region not in {"americas", "europe", "asia", "sea"}:
            raise ValueError("Invalid regional or platform routing value")
        if not api_key or timeout <= 0 or max_retries < 0 or max_retry_wait <= 0:
            raise ValueError("API key and valid retry/timeout settings required")
        self.base_url = f"https://{region}.api.riotgames.com"
        self.session = session or requests.Session()
        self.api_key = api_key
        self.timeout, self.max_retries = timeout, max_retries
        self.max_retry_wait, self.sleep = max_retry_wait, sleep

    @staticmethod
    def _retry_after(value: str) -> float:
        try:
            return max(0, float(value))
        except ValueError:
            try:
                return max(
                    0, (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds()
                )
            except (ValueError, TypeError, OverflowError):
                return 1.0

    def _get(self, path: str, params: dict | None = None):
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.get(
                    self.base_url + path,
                    params=params,
                    headers={"X-Riot-Token": self.api_key},
                    timeout=self.timeout,
                )
            except (requests.ConnectionError, requests.Timeout):
                if attempt == self.max_retries:
                    raise RiotAPIError("Riot API connection failed after retries") from None
                self.sleep(min(self.max_retry_wait, 2**attempt + random.random()))
                continue
            if response.status_code == 429 or 500 <= response.status_code < 600:
                if attempt == self.max_retries:
                    raise RiotAPIError(f"Riot HTTP {response.status_code}: retries exhausted")
                delay = self._retry_after(response.headers.get("Retry-After", "1"))
                if delay > self.max_retry_wait:
                    # Do not shorten Retry-After and hit the service prematurely.
                    raise RiotAPIError(f"Retry-After {delay}s exceeds configured retry budget")
                self.sleep(max(delay, min(self.max_retry_wait, 2**attempt)))
                continue
            if not response.ok:
                raise RiotAPIError(
                    f"Riot HTTP {response.status_code} for {path}; check key/routing"
                )
            try:
                return response.json()
            except ValueError:
                raise RiotAPIError("Riot returned invalid JSON") from None
        raise AssertionError("Unreachable")

    def fetch_match_ids(
        self, puuid: str, *, start: int = 0, count: int = 20, queue: int | None = 420
    ) -> list[str]:
        if not puuid or start < 0 or not 1 <= count <= 100:
            raise ValueError("PUUID required; start >= 0 and count in [1,100]")
        params = {"start": start, "count": count}
        if queue is not None:
            params["queue"] = queue
        result = self._get(f"/lol/match/v5/matches/by-puuid/{quote(puuid, safe='')}/ids", params)
        if not isinstance(result, list) or not all(isinstance(x, str) for x in result):
            raise RiotAPIError("Malformed match ID response")
        return result

    def fetch_match(self, match_id: str) -> dict:
        return self._get(f"/lol/match/v5/matches/{quote(match_id, safe='')}")

    def fetch_timeline(self, match_id: str) -> dict:
        return self._get(f"/lol/match/v5/matches/{quote(match_id, safe='')}/timeline")

    def parse_timeline(
        self, match: dict, timeline: dict, config: FeatureConfig | None = None
    ) -> list[Snapshot]:
        return TimelineProcessor(config or FeatureConfig()).parse(match, timeline)

    def close(self):
        self.session.close()


class TimelineProcessor:
    def __init__(self, config: FeatureConfig):
        self.config = config

    def parse(self, match: dict, timeline: dict) -> list[Snapshot]:
        """Forward-fill frames, never interpolate from future frames or final stats.

        Events are timestamped separately, so item/objective changes become visible
        as soon as they occur. Death/ability cooldowns remain unknown.
        """
        info = match["info"]
        participants = info["participants"]
        if info.get("mapId") != 11 or len(participants) != 10:
            raise ValueError("Only 5v5 Summoner's Rift matches are supported")
        if any(sum(p["teamId"] == team for p in participants) != 5 for team in (100, 200)):
            raise ValueError("Invalid team roster")
        if not all("win" in p for p in participants):
            raise ValueError("Completed match outcome required")
        frames = sorted(timeline["info"]["frames"], key=lambda f: f["timestamp"])
        if not frames or frames[0]["timestamp"] != 0:
            raise ValueError("Timeline must start at zero")
        events = sorted(
            (e for f in frames for e in f.get("events", [])), key=lambda e: e["timestamp"]
        )
        end_events = [e["timestamp"] / 1000 for e in events if e["type"] == "GAME_END"]
        duration = min(end_events) if end_events else float(info.get("gameDuration", 0))
        # Old Match-V5 payloads used milliseconds for gameDuration.
        if not end_events and info.get("gameEndTimestamp") and info.get("gameStartTimestamp"):
            duration = (info["gameEndTimestamp"] - info["gameStartTimestamp"]) / 1000
        if duration <= 0 or frames[-1]["timestamp"] / 1000 < duration - 90:
            raise ValueError("Incomplete timeline or invalid match duration")
        tracker = ObjectiveTracker(self.config)
        inventories = {p["participantId"]: Counter() for p in participants}
        inventory_known = dict.fromkeys(inventories, True)
        by_id = {p["participantId"]: p for p in participants}
        times = list(range(0, int(duration), self.config.interval_seconds)) + [duration]
        snapshots, frame_index, event_index = [], 0, 0
        for timestamp in times:
            while (
                frame_index + 1 < len(frames)
                and frames[frame_index + 1]["timestamp"] <= timestamp * 1000
            ):
                frame_index += 1
            while (
                event_index < len(events) and events[event_index]["timestamp"] <= timestamp * 1000
            ):
                event = events[event_index]
                event_index += 1
                event_type = event["type"]
                pid = event.get("participantId")
                inventory = inventories.get(pid)
                if inventory is not None:
                    if event_type == "ITEM_PURCHASED":
                        inventory[event["itemId"]] += 1
                    elif event_type in {"ITEM_SOLD", "ITEM_DESTROYED"}:
                        if inventory[event["itemId"]] <= 0:
                            inventory_known[pid] = False
                        inventory[event["itemId"]] = max(0, inventory[event["itemId"]] - 1)
                    elif event_type == "ITEM_UNDO":
                        before, after = event.get("beforeId", 0), event.get("afterId", 0)
                        if before:
                            if inventory[before] <= 0:
                                inventory_known[pid] = False
                            inventory[before] = max(0, inventory[before] - 1)
                        if after:
                            inventory[after] += 1
                if event_type == "ELITE_MONSTER_KILL":
                    monster = event.get("monsterType", "").lower()
                    kind = {
                        "rift_herald": "herald",
                        "riftherald": "herald",
                        "baron_nashor": "baron",
                    }.get(monster, monster)
                    killer = by_id.get(event.get("killerId"), {})
                    team = event.get("killerTeamId", killer.get("teamId"))
                    event_time = event["timestamp"] / 1000
                    tracker.consume(kind, event_time, team)
                    if event.get("monsterSubType") == "ELDER_DRAGON":
                        tracker.consume("elder", event_time, team)
            frame = frames[frame_index]
            players = []
            for p in participants:
                pid = p["participantId"]
                pf = frame["participantFrames"].get(str(pid), {})
                pos = pf.get("position")
                players.append(
                    {
                        "id": pid,
                        "team": p["teamId"],
                        "role": p.get("teamPosition", ""),
                        "champion": p.get("championName", ""),
                        "gold": pf.get("totalGold"),
                        "level": pf.get("level"),
                        "position": (pos["x"], pos["y"]) if pos else None,
                        "items": list(inventories[pid].elements()),
                        "inventory_known": inventory_known[pid],
                        "cooldown_remaining": None,
                    }
                )
            snapshots.append(
                Snapshot(
                    timestamp,
                    copy.deepcopy(players),
                    tracker.state(timestamp),
                    timestamp - frame["timestamp"] / 1000,
                )
            )
        return snapshots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--puuid", required=True)
    parser.add_argument("--region", default="europe")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--queue", type=int, default=420)
    parser.add_argument(
        "--items", type=Path, required=True, help="Patch-matched Data Dragon item.json"
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/macro.npz"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--reward", choices=("sparse", "gold"), default="sparse")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    from action_labeler import MDPDatasetBuilder

    config = (
        FeatureConfig(**json.loads(args.config.read_text())) if args.config else FeatureConfig()
    )
    catalog = ItemCatalog(json.loads(args.items.read_text()))
    builder = MDPDatasetBuilder(config, catalog, reward_mode=args.reward)
    api = RiotDataIngestor(os.environ.get("RIOT_API_KEY", ""), args.region)
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    try:
        for match_id in dict.fromkeys(
            api.fetch_match_ids(args.puuid, count=args.count, start=args.start, queue=args.queue)
        ):
            match, timeline = api.fetch_match(match_id), api.fetch_timeline(match_id)
            # Match IDs come from Riot; reject unexpected path characters before persistence.
            if not all(c.isalnum() or c in "_-" for c in match_id):
                raise ValueError("Invalid match ID")
            (args.raw_dir / f"{match_id}.json").write_text(
                json.dumps({"match": match, "timeline": timeline})
            )
            try:
                snapshots = api.parse_timeline(match, timeline, config)
                builder.add_match(match, snapshots)
            except ValueError as error:
                LOG.warning("Skipping %s: %s", match_id, error)
        builder.save(args.output)
        LOG.info("Saved %s", args.output)
    finally:
        api.close()


if __name__ == "__main__":
    main()
