"""Versioned action contracts. Labels describe geometry, never unobserved intent."""

ACTION_SCHEMA = "macro-v2"
ACTION_NAMES = dict(
    enumerate(
        (
            "Dragon area setup",
            "Baron area setup",
            "Herald area setup",
            "Cross-map pressure",
            "Side-lane split formation",
            "Group mid",
            "Group top side",
            "Group bottom side",
            "Own-half defensive formation",
            "Base regroup",
            "Dispersed lane coverage",
            "Unclassified / insufficient evidence",
        )
    )
)


def action_names(metadata: dict) -> dict[int, str]:
    if metadata.get("action_schema") != ACTION_SCHEMA:
        raise ValueError("Incompatible action schema; rebuild the dataset and retrain")
    actual = {int(k): v for k, v in metadata["action_names"].items()}
    if actual != ACTION_NAMES:
        raise ValueError("Action names do not match the versioned contract")
    return actual
