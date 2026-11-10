from unittest.mock import Mock

import pytest
from offline_rift.catalogs import CatalogCache
from offline_rift.collection import Collector, RiotAPIError, RiotClient
from rift_common.artifacts import atomic_json, read_json


def test_regional_route_header_rate_limit_and_no_early_retry():
    session = Mock()
    limited = Mock(status_code=429, headers={"Retry-After": "7"})
    success = Mock(status_code=200, ok=True)
    success.json.return_value = ["KR_1"]
    session.get.side_effect = [limited, success]
    sleep = Mock()
    client = RiotClient("private-test-key", "kr", session=session, sleep=sleep)
    assert client.fetch_match_ids("verified-puuid") == ["KR_1"]
    assert session.get.call_args.args[0].startswith("https://asia.api.riotgames.com/")
    assert session.get.call_args.kwargs["headers"] == {"X-Riot-Token": "private-test-key"}
    assert "private-test-key" not in session.get.call_args.args[0]
    assert sleep.call_args.args[0] >= 7
    session.get.side_effect = None
    session.get.return_value = Mock(status_code=429, headers={"Retry-After": "999"})
    with pytest.raises(RiotAPIError, match="exceeds"):
        client.fetch_match_ids("verified-puuid")


def test_collection_pagination_dedup_resume_and_interruption(tmp_path, match_payloads):
    player = {"puuid": "player-1", "riot_id": "Example#TEST", "region": "asia"}
    collector = Collector(tmp_path / "raw", tmp_path / "artifacts")
    api = Mock()
    api.fetch_match_ids.side_effect = [["KR_1", "KR_1"] + ["KR_1"] * 98, ["KR_2"]]
    api.fetch_match.side_effect = lambda mid: match_payloads(mid)[0]
    api.fetch_timeline.side_effect = lambda mid: match_payloads(mid)[1]
    report = collector.collect(player, count=101, api=api)
    assert report["downloaded"] == ["KR_1", "KR_2"]
    assert api.fetch_match_ids.call_args_list[1].kwargs["start"] == 100
    api.fetch_match_ids.side_effect = None
    api.fetch_match_ids.return_value = ["KR_1", "KR_2"]
    api.fetch_match.reset_mock()
    assert collector.collect(player, count=2, api=api)["reused"] == ["KR_1", "KR_2"]
    api.fetch_match.assert_not_called()
    api.fetch_match_ids.return_value = ["KR_3"]
    api.fetch_match.side_effect = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        collector.collect(player, count=1, api=api)
    statuses = [read_json(p)["status"] for p in (tmp_path / "artifacts/batches").glob("*.json")]
    assert "interrupted" in statuses
    assert not (tmp_path / "raw/KR_3.json").exists()


def test_exact_catalog_patch_and_missing_patch_refusal(tmp_path, catalog):
    atomic_json(tmp_path / "items-16.22.json", {"version": "16.22.1", "data": catalog.data})
    cache = CatalogCache(tmp_path, session=Mock())
    assert cache.get("16.22")[1] == "16.22.1"
    cache.session.get.assert_not_called()
    response = Mock()
    response.json.return_value = ["16.22.1"]
    cache.session.get.return_value = response
    with pytest.raises(ValueError, match="No catalog"):
        cache.get("16.21")
    atomic_json(tmp_path / "items-16.20.json", {"version": "16.22.1", "data": catalog.data})
    with pytest.raises(ValueError, match="mismatch"):
        cache.get("16.20")


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("fault", ["timeline_id", "empty_frames", "identity"])
def test_bad_payloads_are_not_published_or_silently_reused(tmp_path, match_payloads, cached, fault):
    match, timeline = match_payloads("KR_1")
    if fault == "timeline_id":
        timeline["metadata"] = {"matchId": "KR_wrong"}
    elif fault == "empty_frames":
        timeline["info"]["frames"] = []
    else:
        for participant in match["info"]["participants"]:
            participant["puuid"] = "different"
    path = tmp_path / "raw/KR_1.json"
    if cached:
        atomic_json(path, {"match": match, "timeline": timeline})
        original = path.read_bytes()
    api = Mock()
    api.fetch_match_ids.return_value = ["KR_1"]
    api.fetch_match.return_value = match
    api.fetch_timeline.return_value = timeline
    with pytest.raises(ValueError, match="configured identity"):
        Collector(tmp_path / "raw", tmp_path / "artifacts").collect(
            {"puuid": "player-1", "region": "asia"}, count=1, api=api
        )
    if cached:
        assert path.read_bytes() == original
        api.fetch_match.assert_not_called()
    else:
        assert not path.exists()
    reports = list((tmp_path / "artifacts/batches").glob("*.json"))
    assert len(reports) == 1 and read_json(reports[0])["status"] == "failed"
