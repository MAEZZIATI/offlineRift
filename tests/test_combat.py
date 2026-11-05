import pytest
from rift_common.artifacts import atomic_json, file_digest, read_json
from rift_common.dataset import DatasetBuilder
from rift_research.combat import audit_combat


def test_combat_audit_preserves_death_only_overlap_and_checks_raw_provenance(
    tmp_path, match_payloads, catalog
):
    match, timeline = match_payloads("KR_audit")
    raw = tmp_path / "raw"
    raw.mkdir()
    cached = raw / "KR_audit.json"
    atomic_json(cached, {"match": match, "timeline": timeline})
    builder = DatasetBuilder()
    builder.add_match(match, timeline, catalog)
    dataset = tmp_path / "data.npz"
    builder.save(
        dataset, catalog=catalog, provenance={"input_sha256": {"KR_audit": file_digest(cached)}}
    )
    report = audit_combat(dataset, raw, tmp_path / "audit", samples_per_stratum=1)
    assert report["status"] == "complete"
    assert report["counts"]["combat_transitions"] == 3
    assert report["counts"]["death_only"] == 1
    assert report["counts"]["kill_or_assist_without_death"] == 2
    assert report["counts"]["lane_cs_also_increased"] == 2
    assert report["counts"]["jungle_cs_also_increased"] == 1
    assert sum(report["without_combat_priority"].values()) == 3
    samples = read_json(tmp_path / "audit/review-sample.json")
    assert samples["TOP:death_only"][0]["events"][0]["victim"] == "1"
    assert samples["TOP:death_only"][0]["alternative_without_combat_priority"] == "FARM_TOP"
    with pytest.raises(FileExistsError):
        audit_combat(dataset, raw, tmp_path / "audit")
    cached.write_text("{}")
    with pytest.raises(ValueError, match="provenance"):
        audit_combat(dataset, raw, tmp_path / "tampered")
    assert read_json(tmp_path / "tampered/report.json")["status"] == "failed"
