"""Versioned foundation data-contract acceptance tests."""

import pytest
from pydantic import ValidationError

from offlinerift.domain.contracts import DatasetRef, ParticipantKey, Review


def test_participant_slot_key_round_trip() -> None:
    key = ParticipantKey(match_id="EUW1_123", participant_slot=10)
    assert ParticipantKey.model_validate_json(key.model_dump_json()) == key
    with pytest.raises(ValidationError):
        ParticipantKey(match_id="EUW1_123", participant_slot=11)
    with pytest.raises(ValidationError):
        ParticipantKey.model_validate({"match_id": "EUW1_123", "participant_slot": 1, "puuid": "x"})


def test_contracts_reject_unknown_versions_and_non_abstaining_foundation() -> None:
    with pytest.raises(ValidationError):
        ParticipantKey.model_validate(
            {"schema_version": "2", "match_id": "EUW1_123", "participant_slot": 1}
        )
    with pytest.raises(ValidationError):
        DatasetRef(version="v1", sha256="invalid")
    review = Review(key=ParticipantKey(match_id="EUW1_123", participant_slot=1))
    assert review.abstained and review.reason
