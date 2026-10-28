"""Local Live Client polling and inference with the shared offline feature schema."""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlparse

import requests

from features import FeatureConfig, ObjectiveTracker, Snapshot
from rl_agent import MacroPolicy

LOG = logging.getLogger(__name__)
ROLE_MAP = {
    "TOP": "TOP",
    "JUNGLE": "JUNGLE",
    "MIDDLE": "MIDDLE",
    "BOTTOM": "BOTTOM",
    "UTILITY": "UTILITY",
    "SUPPORT": "UTILITY",
}


class LiveClientUnavailable(RuntimeError):
    pass


class LiveStateAdapter:
    def __init__(self, config: FeatureConfig):
        self.config = config

    def snapshot(self, data: dict) -> Snapshot:
        game = data.get("gameData", {})
        if game.get("mapNumber") != 11:
            raise ValueError("Live inference requires Summoner's Rift (map 11)")
        timestamp = float(game["gameTime"])
        if timestamp < 0:
            raise ValueError("Invalid game clock")
        raw_players = data.get("allPlayers", [])
        if len(raw_players) != 10:
            raise ValueError("Expected ten live players")
        players = []
        for i, p in enumerate(raw_players):
            team = {"ORDER": 100, "CHAOS": 200}.get(p.get("team"))
            if team is None:
                raise ValueError("Unknown live team")
            items = [
                int(item["itemID"])
                for item in p.get("items", [])
                for _ in range(int(item.get("count", 1)))
            ]
            players.append(
                {
                    "id": i + 1,
                    "team": team,
                    "champion": p.get("rawChampionName", p.get("championName", "")).removeprefix(
                        "game_character_displayname_"
                    ),
                    "role": ROLE_MAP.get(p.get("position", "").upper(), ""),
                    "level": p.get("level"),
                    "items": items,
                    "gold": None,
                    "inventory_known": "items" in p,
                    # 'position' in allPlayers is a lane role, not an x/y coordinate.
                    "position": None,
                    "cooldown_remaining": None,
                }
            )
        # Rebuild from the returned event history: idempotent and safe across new games.
        tracker = ObjectiveTracker(self.config)
        identities = {}
        for raw, player in zip(raw_players, players):
            for key in ("riotId", "summonerName", "riotIdGameName"):
                if raw.get(key):
                    identities.setdefault(raw[key], set()).add(player["team"])
        event_kinds = {"DragonKill": "dragon", "BaronKill": "baron", "HeraldKill": "herald"}
        events = sorted(
            data.get("events", {}).get("Events", []), key=lambda event: event.get("EventTime", 0)
        )
        for event in events:
            event_time = float(event.get("EventTime", 0))
            if event_time > timestamp:
                continue
            kind = event_kinds.get(event.get("EventName"))
            if kind:
                teams = identities.get(event.get("KillerName"), set())
                team = next(iter(teams)) if len(teams) == 1 else None
                tracker.consume(kind, event_time, team)
                if kind == "dragon" and str(event.get("DragonType", "")).lower() == "elder":
                    tracker.consume("elder", event_time, team)
        return Snapshot(timestamp, players, tracker.state(timestamp))


class LiveClientInference:
    def __init__(
        self,
        model_dir: str | Path | None = None,
        *,
        team: int = 100,
        endpoint: str = "https://127.0.0.1:2999/liveclientdata/allgamedata",
        interval_seconds: float = 10,
        timeout: float = 3,
        verify_tls: bool | str = False,
        session: requests.Session | None = None,
        policy: MacroPolicy | None = None,
    ):
        if team not in (100, 200) or interval_seconds <= 0 or timeout <= 0:
            raise ValueError("Invalid team, polling interval, or timeout")
        parsed = urlparse(endpoint)
        if (
            parsed.scheme not in {"http", "https"}
            or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.username
            or parsed.password
        ):
            raise ValueError("Live endpoint must be local HTTP(S)")
        if policy is None and model_dir is None:
            raise ValueError("Provide a model directory or policy")
        self.policy = policy if policy is not None else MacroPolicy(model_dir)
        if self.policy.config.profile != "live":
            raise ValueError("This live API requires a policy trained with profile='live'")
        self.adapter = LiveStateAdapter(self.policy.config)
        self.team, self.endpoint = team, endpoint
        self.interval_seconds, self.timeout, self.verify_tls = interval_seconds, timeout, verify_tls
        self.session = session or requests.Session()
        self.session.trust_env = False  # Local requests must not be sent through a proxy.

    def fetch_live_data(self) -> dict:
        try:
            response = self.session.get(
                self.endpoint, timeout=self.timeout, verify=self.verify_tls, allow_redirects=False
            )
            response.raise_for_status()
            if 300 <= response.status_code < 400:
                raise LiveClientUnavailable("Live endpoint returned a redirect")
            data = response.json()
            if not isinstance(data, dict):
                raise TypeError("Expected JSON object")
            return data
        except (requests.RequestException, ValueError, TypeError):
            raise LiveClientUnavailable(
                "Live Client unavailable; start an active local match"
            ) from None

    def infer_once(self) -> dict:
        snapshot = self.adapter.snapshot(self.fetch_live_data())
        observation = self.policy.encoder.encode(snapshot, self.team)
        result = self.policy.recommend(observation)
        result.update(
            {
                "game_time": snapshot.time,
                "team": self.team,
                "status": "partial_observation",
                "unavailable_features": [
                    "team_total_gold",
                    "player_coordinates",
                    "ability_cooldowns",
                ],
                "objective_timers": "configured estimates",
            }
        )
        return result

    def run(self, callback: Callable[[dict], None] | None = None, *, max_polls: int | None = None):
        """Poll until Ctrl-C. Errors emit status, never a stale recommendation."""
        callback = callback or (lambda result: print(json.dumps(result), flush=True))
        attempts = 0
        try:
            while max_polls is None or attempts < max_polls:
                started = time.monotonic()
                try:
                    result = self.infer_once()
                except LiveClientUnavailable as error:
                    result = {"status": "offline", "message": str(error)}
                except (KeyError, TypeError, ValueError) as error:
                    result = {"status": "invalid_state", "message": str(error)}
                callback(result)
                attempts += 1
                if max_polls is None or attempts < max_polls:
                    time.sleep(max(0, self.interval_seconds - (time.monotonic() - started)))
        except KeyboardInterrupt:
            LOG.info("Polling stopped")
        finally:
            self.session.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--team", type=int, choices=(100, 200), default=100)
    parser.add_argument("--interval", type=float, default=10)
    parser.add_argument("--endpoint", default="https://127.0.0.1:2999/liveclientdata/allgamedata")
    parser.add_argument("--ca-bundle", help="Optional Riot root certificate for TLS verification")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    client = LiveClientInference(
        args.model,
        team=args.team,
        endpoint=args.endpoint,
        interval_seconds=args.interval,
        verify_tls=args.ca_bundle or False,
    )
    client.run(max_polls=1 if args.once else None)


if __name__ == "__main__":
    main()
