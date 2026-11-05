"""One model, one bounded snapshot cache, and no background polling."""

from __future__ import annotations

import logging
import threading
import time
from copy import deepcopy

from rift_common.pipeline.actions import action_names

from offline_rift.inference.live_client import LiveClientInference, LiveClientUnavailable

LOG = logging.getLogger(__name__)


def model_summary(metadata: dict) -> dict:
    report = metadata.get("training_report", {})
    predictions = report.get("validation_prediction_counts", [])
    counts = report.get("action_counts", [])
    return {
        "collapsed": bool(predictions) and sum(n > 0 for n in predictions) == 1,
        "action_coverage": sum(n > 0 for n in counts),
        "action_counts": counts,
        "heuristic_agreement": report.get("validation_heuristic_agreement"),
        "majority_baseline": report.get("validation_majority_baseline"),
        "prediction_counts": predictions,
        "balanced_accuracy": report.get("validation_balanced_accuracy"),
        "macro_f1": report.get("validation_macro_f1"),
        "prediction_method": report.get("prediction_mode"),
    }


class LiveDashboard:
    def __init__(self, client: LiveClientInference):
        if client.interval_seconds < 2:
            raise ValueError("UI polling interval must be at least 2 seconds")
        self.client = client
        self.model = model_summary(client.policy.metadata)
        self._lock = threading.Lock()
        self._next_poll = 0.0
        self._state: dict = {}

    def state(self) -> dict:
        # Multiple tabs share one game fetch. The lock also serializes model inference.
        with self._lock:
            if time.monotonic() >= self._next_poll:
                self._state = self._poll()
                self._next_poll = time.monotonic() + self.client.interval_seconds
            state = deepcopy(self._state)
            state["valid_for_ms"] = max(0, int((self._next_poll - time.monotonic()) * 1000))
            return state

    def _poll(self) -> dict:
        state = {
            "status": "offline",
            "checked_at": time.time(),
            "recommendations": {},
            "model": self.model,
            "actions": action_names(self.client.policy.metadata),
            "transport": self.client.source.transport,
            "default_team": self.client.team,
            "interval_seconds": self.client.interval_seconds,
        }
        try:
            snapshot = self.client.adapter.snapshot(self.client.fetch_live_data())
            recommendations = {
                str(team): self.client.recommend(snapshot, team) for team in (100, 200)
            }
            state.update(status="live", recommendations=recommendations)
        except LiveClientUnavailable as error:
            state["message"] = str(error)
        except (KeyError, TypeError, ValueError) as error:
            state.update(status="invalid_state", message=str(error))
        except Exception:
            # Discard every old recommendation, including after unexpected model errors.
            LOG.exception("Live dashboard inference failed")
            state.update(status="error", message="Inference failed. Check the server terminal.")
        state["checked_at"] = time.time()
        return state

    def close(self):
        self.client.close()
