"""V1 interchange contracts; pure validation without I/O or model assumptions."""

from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field


class Contract(BaseModel):
    """Reject undeclared fields and mutation at every contract boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    schema_version: Literal["1"] = "1"


class ParticipantKey(Contract):
    """Dataset identity is a match and its one-based participant slot."""

    match_id: str = Field(pattern=r"^(EUW1|NA1|KR)_[0-9]+$")
    participant_slot: int = Field(ge=1, le=10)


class DatasetRef(Contract):
    """Immutable input version and content digest, without player identifiers."""

    version: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class Review(Contract):
    """Foundation response always abstains until later model phases exist."""

    key: ParticipantKey
    abstained: Literal[True] = True
    reason: str = "No recommendation model is implemented in Phase 1."


class ReviewProvider(Protocol):
    """Inbound use-case boundary for a future post-game adapter."""

    def review(self, key: ParticipantKey) -> Review:
        """Return a review using only validated decision-time information."""
        ...


class ArtifactStore(Protocol):
    """Storage port: callers inject an adapter rather than a global store."""

    def write_json(self, name: str, payload: str) -> None:
        """Persist an artifact atomically."""
        ...

    def read_json(self, name: str) -> str:
        """Read a previously persisted artifact."""
        ...
