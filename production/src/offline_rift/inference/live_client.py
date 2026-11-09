"""Local live observations, actor selection and history-safe next-action inference."""

import argparse
import json
import math
import time
from collections import deque
from pathlib import Path

from rift_common.actions import ROLES
from rift_common.objectives import ObjectiveRules, ObjectiveTracker
from rift_common.state import Observation, PlayerState

from offline_rift.inference.coach import load_policy
from offline_rift.inference.transport import DEFAULT_ENDPOINT, LiveClientUnavailable, LiveDataSource


class LiveStateAdapter:
    def __init__(self):
        self.history = deque(maxlen=128)
        self.roster = None

    @property
    def previous(self):
        if not self.history:
            return None
        current = self.history[-1]
        return next(
            (value for value in reversed(self.history) if 60 <= current.time - value.time <= 120),
            None,
        )

    def snapshot(self, data):
        game = data.get("gameData", {})
        timestamp = float(game["gameTime"])
        raw = data.get("allPlayers", [])
        if (
            game.get("mapNumber") != 11
            or len(raw) != 10
            or not math.isfinite(timestamp)
            or timestamp < 0
        ):
            raise ValueError("An active 5v5 Summoner’s Rift observation is required")
        if game.get("gameMode") not in (None, "CLASSIC"):
            raise ValueError(
                "Policy advice is unavailable for this game mode; training covers ranked Classic only"
            )
        identities = {}
        prepared = []
        for p in raw:
            team = {"ORDER": 100, "CHAOS": 200}.get(p.get("team"))
            if team is None:
                raise ValueError("Unknown team")
            champion = p.get("championName") or p.get("rawChampionName", "").removeprefix(
                "game_character_displayname_"
            )
            identity = p.get("riotId") or p.get("summonerName") or f"{team}:{champion}"
            prepared.append((p, team, str(identity), champion))
            for name in (p.get("riotId"), p.get("summonerName"), p.get("riotIdGameName")):
                if name:
                    identities.setdefault(name, set()).add((str(identity), team))
        if len({entry[2] for entry in prepared}) != 10 or any(
            sum(entry[1] == t for entry in prepared) != 5 for t in (100, 200)
        ):
            raise ValueError("Ambiguous roster identity")
        active_data = data.get("activePlayer", {})
        names = (
            active_data.get("riotId"),
            active_data.get("summonerName"),
            active_data.get("riotIdGameName"),
        )
        active_matches = set().union(*(identities.get(name, set()) for name in names if name))
        active_id = next(iter(active_matches))[0] if len(active_matches) == 1 else None
        tracker = ObjectiveTracker(ObjectiveRules())
        normalized = []
        raw_events = sorted(
            data.get("events", {}).get("Events", []), key=lambda e: e.get("EventTime", 0)
        )
        complete = timestamp < 60
        for event in raw_events:
            moment = float(event.get("EventTime", 0))
            if not math.isfinite(moment) or moment < 0:
                raise ValueError("Invalid event clock")
            if moment > timestamp:
                continue
            if event.get("EventName") == "GameEnd":
                raise ValueError("Game has ended; waiting for an active game")
            if event.get("EventName") == "GameStart" or (
                event.get("EventID") == 0 and moment <= 60
            ):
                complete = True
            killer = identities.get(event.get("KillerName"), set())
            identity, team = next(iter(killer)) if len(killer) == 1 else (None, None)
            kind = event.get("EventName")
            if kind in {"DragonKill", "BaronKill", "HeraldKill"}:
                objective = {"DragonKill": "dragon", "BaronKill": "baron", "HeraldKill": "herald"}[
                    kind
                ]
                subtype = event.get("DragonType")
                tracker.consume(
                    objective,
                    moment,
                    team,
                    is_elder=None
                    if objective == "dragon" and subtype is None
                    else str(subtype).lower() == "elder",
                )
            elif kind == "ChampionKill":
                victim = identities.get(event.get("VictimName"), set())
                victim_id, victim_team = next(iter(victim)) if len(victim) == 1 else (None, None)
                normalized.append(
                    {
                        "kind": "kill",
                        "time": moment,
                        "actor": identity,
                        "victim": victim_id,
                        "victim_team": victim_team,
                        "assists": [],
                    }
                )
            elif kind == "TurretKilled":
                normalized.append(
                    {"kind": "tower", "time": moment, "team": 300 - team if team else None}
                )
        players = []
        for p, team, identity, champion in prepared:
            scores = p.get("scores", {})
            stats = active_data.get("championStats", {}) if identity == active_id else {}
            health, maximum = stats.get("currentHealth"), stats.get("maxHealth")
            inventory = p.get("items")
            items = tuple(
                int(item["itemID"])
                for item in (inventory or [])
                for _ in range(int(item.get("count", 1)))
            )
            role = p.get("position", "").upper()
            role = {"SUPPORT": "UTILITY", "MID": "MIDDLE", "BOT": "BOTTOM"}.get(role, role)
            players.append(
                PlayerState(
                    id=identity,
                    team=team,
                    champion=champion,
                    role=role if role in ROLES else "",
                    level=p.get("level"),
                    items=items,
                    inventory_known=inventory is not None,
                    cs=scores.get("creepScore"),
                    kills=scores.get("kills"),
                    deaths=scores.get("deaths"),
                    assists=scores.get("assists"),
                    is_dead=p.get("isDead"),
                    health=health / maximum if health is not None and maximum else None,
                )
            )
        objective_state = tracker.state(timestamp)
        if not complete:
            for objective in objective_state.values():
                objective.update(enabled=False, available=False, eta=0)
        result = Observation(
            timestamp, tuple(players), objective_state, tuple(normalized), active_id, complete
        )
        roster = tuple(sorted((p.id, p.team) for p in players))
        if self.history and (timestamp < self.history[-1].time or roster != self.roster):
            self.history.clear()
        if self.history and timestamp == self.history[-1].time:
            self.history.pop()
        self.history.append(result)
        self.roster = roster
        while self.history and timestamp - self.history[0].time > 180:
            self.history.popleft()
        return result


class LiveClientInference:
    def __init__(
        self,
        model_dir=None,
        *,
        team=100,
        actor="active",
        role=None,
        interval_seconds=10,
        transport="auto",
        policy=None,
        session=None,
        endpoint=DEFAULT_ENDPOINT,
        timeout=3,
        verify_tls=False,
        capture_availability=None,
        capture_session=None,
    ):
        if team not in (100, 200) or not math.isfinite(interval_seconds) or interval_seconds <= 0:
            raise ValueError("Invalid team or polling interval")
        self.source = LiveDataSource(
            endpoint, timeout=timeout, verify_tls=verify_tls, transport=transport, session=session
        )
        try:
            self.policy = policy or load_policy(model_dir)
            self.capture = None
            self.session_capture = None
            if capture_availability is not None:
                from offline_rift.inference.availability import AvailabilityCapture

                self.capture = AvailabilityCapture(capture_availability)
            if capture_session is not None:
                from offline_rift.inference.session import SessionCapture

                self.session_capture = SessionCapture(
                    capture_session, model=model_dir, team=team, actor=actor, role=role
                )
        except Exception:
            self.source.close()
            raise
        self.adapter = LiveStateAdapter()
        from offline_rift.inference.advice import AdviceController

        self.advice = AdviceController()
        self.team, self.actor, self.role, self.interval_seconds = (
            team,
            actor,
            role,
            interval_seconds,
        )

    def fetch_live_data(self):
        return self.source.fetch()

    def recommend(self, snapshot, team, actor="active", role=None):
        if actor.startswith("champion:"):
            champion = actor.split(":", 1)[1]
            matches = [
                p
                for p in snapshot.players
                if p.team == team and p.champion.casefold() == champion.casefold()
            ]
            if len(matches) != 1:
                raise ValueError("Selected champion is not uniquely present on the selected team")
            actor = matches[0].id
        if actor == "active":
            if snapshot.active_player is None:
                if self.capture is not None:
                    self.capture.unresolved()
                return {
                    "status": "needs_player",
                    "message": "Select the player you want advice for, or choose the team view.",
                    "team": team,
                    "game_time": snapshot.time,
                }
            actor = snapshot.active_player
            team = snapshot.player(actor).team
        elif actor == "team":
            actor = None
        if self.capture is not None:
            self.capture.record(
                snapshot, self.adapter.previous, self.policy.encoder, team, actor, role
            )
        result = self.policy.recommend(
            snapshot, team, actor, role=role, previous=self.adapter.previous
        )
        if isinstance(result, dict):
            if "macro_recommendation" in result:
                from rift_common.coaching import alignment

                result["macro_recommendation"] = self.advice.select(result["macro_recommendation"])
                result["coordination"]["status"] = alignment(result["macro_recommendation"], result)
            result = self.advice.select(result)
        if self.session_capture is not None:
            observed = self.policy.encoder.encode(
                snapshot, team, actor, role=role, previous=self.adapter.previous
            )
            self.session_capture.decision(
                result, observation=observed, feature_names=self.policy.encoder.names
            )
        return result

    def observe_once(self):
        data = self.fetch_live_data()
        if self.session_capture is not None:
            if self.actor.startswith("champion:"):
                champion = self.actor.split(":", 1)[1].casefold()
                side = "ORDER" if self.team == 100 else "CHAOS"
                if not any(
                    p.get("team") == side and str(p.get("championName", "")).casefold() == champion
                    for p in data.get("allPlayers", [])
                ):
                    raise ValueError("Waiting for selected champion on selected side")
            self.session_capture.record(data)
        try:
            return self.adapter.snapshot(data)
        except (ValueError, KeyError, TypeError):
            if self.session_capture is not None:
                self.session_capture.decision({"status": "invalid_state"})
            raise

    def infer_once(self):
        return self.recommend(self.observe_once(), self.team, self.actor, self.role)

    def run(self, *, max_polls=None):
        polls = 0
        capture_status = "complete"
        try:
            while max_polls is None or polls < max_polls:
                start = time.monotonic()
                try:
                    result = self.infer_once()
                except LiveClientUnavailable as error:
                    result = {"status": "offline", "message": str(error)}
                except (ValueError, KeyError, TypeError) as error:
                    result = {"status": "invalid_state", "message": str(error)}
                print(json.dumps(result, allow_nan=False), flush=True)
                if self.session_capture is not None:
                    self.session_capture.heartbeat(result["status"])
                polls += 1
                if self.session_capture is not None and self.session_capture.ended:
                    break
                if max_polls is None or polls < max_polls:
                    time.sleep(max(0, self.interval_seconds - (time.monotonic() - start)))
        except KeyboardInterrupt:
            capture_status = "interrupted"
        except BaseException:
            capture_status = "failed"
            raise
        finally:
            self.close(capture_status=capture_status)

    def close(self, *, capture_status="complete"):
        try:
            if self.capture is not None:
                self.capture.close(capture_status)
        finally:
            try:
                if self.session_capture is not None:
                    self.session_capture.close(capture_status)
            finally:
                self.source.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--team", type=int, choices=(100, 200), default=100)
    parser.add_argument(
        "--actor", default="active", help="active, team, or a live player identifier"
    )
    parser.add_argument("--role", choices=ROLES)
    parser.add_argument("--interval", type=float, default=10)
    parser.add_argument("--transport", choices=("auto", "local", "windows"), default="auto")
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--capture-session",
        type=Path,
        help="Opt in to measured live samples and advice in a new directory",
    )
    parser.add_argument(
        "--capture-availability",
        type=Path,
        help="Opt in to aggregate availability counts in a new local directory",
    )
    args = parser.parse_args()
    LiveClientInference(
        args.model,
        team=args.team,
        actor=args.actor,
        role=args.role,
        interval_seconds=args.interval,
        transport=args.transport,
        capture_availability=args.capture_availability,
        capture_session=args.capture_session,
    ).run(max_polls=1 if args.once else None)
