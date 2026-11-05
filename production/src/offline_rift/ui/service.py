"""Shared observation cache; actor/role decisions are evaluated against the same snapshot."""

import logging
import threading
import time
from copy import deepcopy

from rift_common.actions import ACTION_NAMES, ROLES, action_catalog

from offline_rift.inference.transport import LiveClientUnavailable

LOG = logging.getLogger(__name__)


class LiveDashboard:
    def __init__(self, client):
        if client.interval_seconds < 2:
            raise ValueError("UI polling interval must be at least 2 seconds")
        self.client = client
        self.lock = threading.Lock()
        self.next_poll = 0.0
        self.snapshot = None
        self.checked_at = 0
        self.failure = None
        self.cache = {}
        metadata = client.policy.metadata
        report = metadata["training_report"]
        metrics = report["validation"]
        self.model = {
            "name": client.policy.name,
            "prediction_method": metadata["algorithm"],
            "action_counts": report["action_counts"],
            "prediction_counts": metrics["prediction_counts"],
            "action_coverage": sum(n > 0 for n in report["action_counts"]),
            "behavior_agreement": metrics["behavior_agreement"],
            "majority_baseline": report["majority_baseline"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "collapsed": metrics["collapsed"],
            "stage": "experimental_next_action_policy",
            "per_group": report.get("per_group", {}),
        }

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
                    self.snapshot = self.client.adapter.snapshot(self.client.fetch_live_data())
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

    def close(self):
        self.client.close()
