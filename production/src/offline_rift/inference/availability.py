"""Explicit opt-in aggregate availability capture; no identities or state values."""

from collections import Counter
from pathlib import Path

from rift_common.actions import ROLES
from rift_common.artifacts import atomic_json
from rift_common.observation_masks import EVIDENCE_FIELDS, evidence_patterns


class AvailabilityCapture:
    def __init__(self, output):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.cells, self.known = Counter(), Counter()
        self.last = None
        self.report = {
            "schema": "availability-capture-v1",
            "status": "running",
            "polls": 0,
            "duplicate_polls": 0,
            "observation_sessions": 0,
            "unresolved_subject_polls": 0,
            "evidence_fields": list(EVIDENCE_FIELDS),
            "cells": {},
            "known_feature_counts": {},
            "limitations": [
                "Polls are correlated; sessions are not verified unique matches",
                "Availability only; no decisions, identities, state values or outcomes stored",
                "Objective enabled flags combine evidence availability and patch rules",
            ],
        }
        self._save()

    def _save(self):
        self.report.update(cells=dict(self.cells), known_feature_counts=dict(self.known))
        atomic_json(self.output / "report.json", self.report)

    def unresolved(self):
        self.report["unresolved_subject_polls"] += 1
        self._save()

    def record(self, current, previous, encoder, team, actor, role):
        # Identity is used only in memory to detect roster/context changes.
        roster = tuple(sorted((p.id, p.team) for p in current.players))
        context = (team, actor, role)
        if self.last and self.last == (roster, current.time, context):
            self.report["duplicate_polls"] += 1
            self._save()
            return
        if self.last is None or roster != self.last[0] or current.time < self.last[1]:
            self.report["observation_sessions"] += 1
        self.last = (roster, current.time, context)
        state = encoder.encode(current, team, actor, role=role, previous=previous)
        reported_role = role or (current.player(actor).role if actor is not None else "TEAM")
        group = "TEAM" if actor is None else reported_role if reported_role in ROLES else "UNKNOWN"
        role_source = (
            "team"
            if actor is None
            else "explicit"
            if role
            else "reported"
            if reported_role in ROLES
            else "missing"
        )
        pattern = evidence_patterns(state, encoder.names)[0]
        bucket = (
            "opening"
            if current.time < 90
            else "early"
            if current.time < 840
            else "middle"
            if current.time < 1500
            else "late"
        )
        cell = f"{group}:{pattern}:{bucket}:{role_source}:previous={int(previous is not None)}:events={int(current.history_complete)}"
        self.cells[cell] += 1
        self.known.update(
            {
                name: int(state[i] > 0)
                for i, name in enumerate(encoder.names)
                if name.endswith(("_known", "_enabled"))
            }
        )
        self.report["polls"] += 1
        self._save()

    def close(self, status="complete"):
        self.report["status"] = status
        self._save()
