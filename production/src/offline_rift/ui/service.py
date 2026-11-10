"""Live HUD producer over the maintained observation, six-policy and persistence path."""

import hashlib
import json
import logging
import threading
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from rift_common.actions import ACTION_NAMES, GROUPS, ROLES, Action, action_catalog
from rift_common.artifacts import file_digest

from offline_rift.inference.transport import LiveClientUnavailable

LOG = logging.getLogger(__name__)


class LiveHudService:
    def __init__(self, client, *, model_dir=None):
        if client.interval_seconds < 2:
            raise ValueError("UI polling interval must be at least 2 seconds")
        self.client = client
        self.lock = threading.RLock()
        self.session_id = uuid4().hex
        self.context_version = 0
        self.last_clock = None
        self.last_roster = None
        self.next_poll = 0.0
        self.snapshot = None
        self.checked_at = 0
        self.failure = None
        self.cache = {}
        metadata = client.policy.metadata
        report = metadata["training_report"]
        metrics = report["validation"]
        action_counts = report["action_counts"]
        baseline = report["majority_baseline"]
        scope = metadata.get("advisory_groups", list(GROUPS))
        if len(scope) == 1 and scope[0] in report.get("per_group", {}):
            metrics = report["per_group"][scope[0]]
            action_counts = report["counts_by_group"][GROUPS.index(scope[0])]
            majority = max(range(len(action_counts)), key=action_counts.__getitem__)
            baseline = metrics["action_support"][majority] / max(1, sum(metrics["action_support"]))
        self.model = {
            "name": client.policy.name,
            "prediction_method": metadata["algorithm"],
            "action_counts": action_counts,
            "prediction_counts": metrics["prediction_counts"],
            "action_coverage": sum(n > 0 for n in action_counts),
            "behavior_agreement": metrics["behavior_agreement"],
            "majority_baseline": baseline,
            "advisory_groups": scope,
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "collapsed": metrics["collapsed"],
            "stage": "experimental_next_action_policy",
            "per_group": report.get("per_group", {}),
            "policy_count": 6 if metadata.get("bundle_schema") == "six-policy-coach-v1" else 1,
        }
        directory = Path(model_dir) if model_dir is not None else None
        self.model.update(
            display_name=(
                f"{directory.parent.parent.name} · {directory.parent.name}"
                if directory and directory.name == "model"
                else directory.name
                if directory
                else client.policy.name
            ),
            fingerprint=file_digest(directory / "coach.json") if directory else "test-service",
            policies={
                scope: {
                    "algorithm": child.algorithm,
                    "candidate": metadata.get("selection", {})
                    .get(scope, {})
                    .get("name", child.algorithm),
                    "development_gate_passed": metadata.get("selection", {})
                    .get(scope, {})
                    .get("development_gate_passed"),
                    "diagnostics": {
                        key: report.get("per_group", {}).get(scope, {}).get(key)
                        for key in (
                            "prediction_counts",
                            "abstentions",
                            "collapsed",
                            "behavior_agreement",
                        )
                    },
                    "supported_actions": int(
                        sum(child.counts[GROUPS.index(scope)] >= child.config["minimum_support"])
                    ),
                }
                for scope, child in getattr(client.policy, "policies", {}).items()
            }
            if isinstance(getattr(client.policy, "policies", None), dict)
            else {},
        )
        self.model["per_group"] = deepcopy(self.model["per_group"])
        for group in self.model["per_group"].values():
            voluntary = group["prediction_counts"].copy()
            voluntary[Action.WAIT_RESPAWN] = 0
            group["voluntary_prediction_counts"] = voluntary

    def state(self, *, team=None, actor=None, role=None):
        role = self.client.role if role is None else role
        team = self.client.team if team is None else team
        actor = self.client.actor if actor is None else actor
        if team not in (100, 200) or role is not None and role not in ROLES:
            raise ValueError("Invalid team or role")
        with self.lock:
            if time.monotonic() >= self.next_poll:
                self.cache.clear()
                self.snapshot = None
                self.failure = None
                try:
                    self.snapshot = self.client.observe_once()
                    roster = tuple(sorted((p.id, p.team) for p in self.snapshot.players))
                    if self.last_clock is not None and (
                        self.snapshot.time < self.last_clock or roster != self.last_roster
                    ):
                        self.session_id = uuid4().hex
                        self.client.advice.contexts.clear()
                    self.last_clock, self.last_roster = self.snapshot.time, roster
                except LiveClientUnavailable as error:
                    self.failure = ("offline", str(error))
                except (ValueError, TypeError, KeyError) as error:
                    self.failure = ("invalid_state", str(error))
                except Exception:
                    LOG.exception("Live observation failed")
                    self.failure = ("error", "Observation unavailable. Check the server terminal.")
                self.checked_at = time.time()
                self.next_poll = time.monotonic() + self.client.interval_seconds
            base = {
                "checked_at": self.checked_at,
                "valid_for_ms": max(0, int((self.next_poll - time.monotonic()) * 1000)),
                "interval_seconds": self.client.interval_seconds,
                "transport": self.client.source.transport,
                "model": self.model,
                "actions": ACTION_NAMES,
                "catalog": action_catalog(),
                "default_team": self.client.team,
                "players": [],
                "active_player": None,
            }
            if self.failure:
                return deepcopy({**base, "status": self.failure[0], "message": self.failure[1]})
            snapshot = self.snapshot
            base["players"] = [
                {"id": p.id, "team": p.team, "champion": p.champion, "role": p.role}
                for p in snapshot.players
            ]
            base["active_player"] = snapshot.active_player
            key = (team, actor, role)
            if key not in self.cache:
                try:
                    self.cache[key] = self.client.recommend(snapshot, team, actor, role)
                except (ValueError, KeyError, TypeError) as error:
                    self.cache[key] = {"status": "invalid_context", "message": str(error)}
                except Exception:
                    LOG.exception("Policy evaluation failed")
                    self.cache[key] = {
                        "status": "error",
                        "message": "Policy evaluation failed. Check the server terminal.",
                    }
            return deepcopy({**base, "status": "live", "recommendation": self.cache[key]})

    def configure(self, values):
        """Explicit player/role selection. Never change collection identities or model weights."""
        if not isinstance(values, dict) or set(values) - {"team", "actor", "role"}:
            raise ValueError("Only team, actor and role may be selected")
        with self.lock:
            team = values.get("team", self.client.team)
            actor = values.get("actor", self.client.actor)
            role = values.get("role", self.client.role)
            if type(team) is not int or team not in (100, 200):
                raise ValueError("Select blue or red team")
            if not isinstance(actor, str) or not actor or len(actor) > 180:
                raise ValueError("Select an active player, observed player or team")
            if role not in (*ROLES, None):
                raise ValueError("Select one of the five roles or the reported role")
            if (
                self.snapshot is not None
                and actor not in {"active", "team"}
                and actor not in {p.id for p in self.snapshot.players}
            ):
                raise ValueError("Selected player is not in the current game")
            self.client.team, self.client.actor, self.client.role = team, actor, role
            if self.snapshot is not None and actor not in {"active", "team"}:
                self.client.team = self.snapshot.player(actor).team
            self.context_version += 1
            self.cache.clear()
            self.client.advice.contexts.clear()
            return self.frame()

    def frame(self):
        """Project real served proposals into the HUD wire contract; no calibrated metrics invented."""
        with self.lock:
            raw = self.state()
            context = {
                "team": self.client.team,
                "actor": self.client.actor,
                "role": self.client.role,
            }
            result = raw.get("recommendation", {})
            frame = {
                "schema": "macro-hud-frame-v1",
                "session_id": self.session_id,
                "context_version": self.context_version,
                "status": "waiting_for_game",
                "message": "Waiting for League of Legends",
                "detail": "Start a Summoner's Rift game. Your selected coach is loaded and waiting.",
                "valid_for_ms": min(60000, raw["valid_for_ms"]),
                "hud": None,
                "context": context,
                "players": raw["players"],
                "active_player": raw["active_player"],
                "model": {
                    key: self.model[key]
                    for key in ("display_name", "fingerprint", "policy_count", "policies")
                },
            }
            if raw["status"] != "live":
                if raw["status"] != "offline":
                    frame.update(
                        status="unavailable",
                        message="Waiting for a supported game state",
                        detail="Advice is withheld for an ended game, unsupported mode or incomplete observation.",
                    )
                return frame
            status = result.get("status")
            if status in {"needs_player", "invalid_context", "error"}:
                frame.update(
                    status=status,
                    message="Choose your player"
                    if status == "needs_player"
                    else "Check the selected player",
                    detail="Open HUD settings and select your current player and role.",
                )
                return frame
            if status == "opening":
                frame.update(
                    status="opening",
                    message="Follow your opening plan",
                    detail="The learned decision window starts after 90 seconds.",
                )
                return frame
            team = result.get("team", self.client.team)
            macro = result.get("macro_recommendation")
            if macro is None:
                macro = (
                    result
                    if self.client.actor == "team"
                    else self.state(team=team, actor="team")["recommendation"]
                )
            macro_supported = macro.get("status") == "recommendation"
            selected_role = result.get("role") or self.client.role
            if (
                selected_role not in ROLES
                and self.snapshot.active_player
                and self.client.actor == "active"
            ):
                selected_role = self.snapshot.player(self.snapshot.active_player).role
            role_supported = (
                status in {"recommendation", "respawning"}
                and self.client.actor != "team"
                and selected_role in ROLES
            )
            conflict = (
                status == "coordination_conflict"
                or result.get("coordination", {}).get("status") == "withheld_conflict"
            )
            role_payload = None
            if selected_role in ROLES and self.client.actor != "team":
                role_payload = {
                    "player_role": selected_role,
                    "action": result.get("action", ACTION_NAMES.get(result.get("action_id"), ""))
                    if role_supported
                    else "",
                    "details": " ".join(
                        filter(None, (result.get("instruction"), result.get("check")))
                    )
                    if role_supported
                    else "",
                    "status": "withheld_conflict"
                    if conflict
                    else "respawning"
                    if status == "respawning"
                    else "recommendation"
                    if role_supported
                    else "insufficient_support",
                }
            macro_payload = None
            urgency = "normal"
            if macro_supported:
                macro_payload = {
                    "title": macro.get(
                        "action", ACTION_NAMES.get(macro.get("action_id"), "Team direction")
                    ),
                    "subtext": " ".join(
                        filter(None, (macro.get("instruction"), macro.get("check")))
                    ),
                }
                objective = {
                    Action.PREPARE_DRAGON: "dragon",
                    Action.PREPARE_BARON: "baron",
                    Action.PREPARE_HERALD: "herald",
                }.get(macro.get("action_id"))
                observed = self.snapshot.objectives.get(objective, {}) if objective else {}
                if observed.get("enabled") and isinstance(observed.get("eta"), (int, float)):
                    seconds = max(0, min(59999, int(observed["eta"])))
                    macro_payload["timer"] = f"{seconds // 60:02d}:{seconds % 60:02d}"
                    macro_payload["subtext"] += " Objective timing is a patch-scoped estimate."
                    urgency = "high" if seconds <= 45 else "normal"
            if not macro_supported and not role_supported and not conflict:
                frame.update(
                    status="insufficient_support",
                    message="Waiting for a supported action",
                    detail="No supported next action is available for this measured state.",
                )
                return frame
            seconds = int(self.snapshot.time)
            digest = hashlib.sha256(
                json.dumps(
                    {
                        "observation": asdict(self.snapshot),
                        "context": context,
                        "model": self.model["fingerprint"],
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest()[:12]
            frame.update(
                status="live",
                message="",
                detail="",
                hud={
                    "timestamp": f"{seconds // 60:02d}:{seconds % 60:02d}",
                    "urgency": urgency,
                    "macro": macro_payload,
                    "role": role_payload,
                    "notice": "Choose your role in HUD settings"
                    if selected_role not in ROLES and self.client.actor != "team"
                    else None,
                    "telemetry": {"ai_confidence": None, "state_hash": digest},
                    "coordination": "withheld_conflict"
                    if conflict
                    else result.get("coordination", {}).get("status", "independent")
                    if macro_supported and role_supported
                    else "missing_support",
                    "valid_for_ms": min(60000, raw["valid_for_ms"]),
                },
            )
            if frame["hud"]["coordination"] not in {
                "aligned",
                "independent",
                "withheld_conflict",
                "missing_support",
            }:
                frame["hud"]["coordination"] = (
                    "independent" if macro_supported and role_supported else "missing_support"
                )
            return frame

    def close(self):
        self.client.close()
