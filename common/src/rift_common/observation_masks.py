"""Named evidence omissions shared by training augmentation and research stress tests."""

import numpy as np

HEALTH_FIELDS = ("self_health", "self_health_known")
COLD_START_FIELDS = tuple(
    f"{field}{suffix}"
    for field in (
        "self_cs_rate",
        "self_level_rate",
        "ally_deaths_60",
        "enemy_deaths_60",
        "own_towers_lost",
        "enemy_towers_lost",
    )
    for suffix in ("", "_known")
) + tuple(
    f"{kind}_{field}"
    for kind in ("dragon", "baron", "herald")
    for field in ("eta", "available", "enabled")
)

EVIDENCE_FIELDS = ("self_health_known",) + tuple(
    name for name in COLD_START_FIELDS if name.endswith(("_known", "_enabled"))
)


def evidence_patterns(observations, names):
    """Exact availability signatures; objective enabled flags also encode game rules."""
    bits = np.atleast_2d(observations)[:, [names.index(name) for name in EVIDENCE_FIELDS]] > 0
    return np.asarray(["".join("1" if bit else "0" for bit in row) for row in bits])
