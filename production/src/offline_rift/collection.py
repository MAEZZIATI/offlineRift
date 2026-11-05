"""Bounded authenticated Riot collection, regional routing and resumable raw caching."""

import os
import random
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import quote

import requests
from rift_common.artifacts import atomic_json, read_json, run_id

REGIONS = {
    **dict.fromkeys(("na1", "br1", "la1", "la2"), "americas"),
    **dict.fromkeys(("euw1", "eun1", "tr1", "ru"), "europe"),
    **dict.fromkeys(("kr", "jp1"), "asia"),
    **dict.fromkeys(("oc1", "ph2", "sg2", "th2", "tw2", "vn2"), "sea"),
}


class RiotAPIError(RuntimeError):
    pass


class RiotClient:
    def __init__(
        self,
        api_key: str,
        region: str = "europe",
        *,
        session: requests.Session | None = None,
        timeout: float = 15,
        max_retries: int = 5,
        max_retry_wait: float = 120,
        sleep=time.sleep,
    ):
        region = REGIONS.get(region.lower(), region.lower())
        if region not in {"americas", "europe", "asia", "sea"}:
            raise ValueError("Invalid regional or platform routing value")
        if not api_key or timeout <= 0 or max_retries < 0 or max_retry_wait <= 0:
            raise ValueError("API key and valid retry/timeout settings required")
        self.base_url = f"https://{region}.api.riotgames.com"
        self.session = session or requests.Session()
        self.api_key = api_key
        self.timeout, self.max_retries = timeout, max_retries
        self.max_retry_wait, self.sleep = max_retry_wait, sleep

    @staticmethod
    def _retry_after(value: str) -> float:
        try:
            return max(0, float(value))
        except ValueError:
            try:
                return max(
                    0, (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds()
                )
            except (ValueError, TypeError, OverflowError):
                return 1.0

    def _get(self, path: str, params: dict | None = None, *, platform: str | None = None):
        if platform is not None and platform not in REGIONS:
            raise ValueError("Unknown Riot platform")
        base_url = self.base_url if platform is None else f"https://{platform}.api.riotgames.com"
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.get(
                    base_url + path,
                    params=params,
                    headers={"X-Riot-Token": self.api_key},
                    timeout=self.timeout,
                )
            except (requests.ConnectionError, requests.Timeout):
                if attempt == self.max_retries:
                    raise RiotAPIError("Riot API connection failed after retries") from None
                self.sleep(min(self.max_retry_wait, 2**attempt + random.random()))
                continue
            if response.status_code == 429 or 500 <= response.status_code < 600:
                if attempt == self.max_retries:
                    raise RiotAPIError(f"Riot HTTP {response.status_code}: retries exhausted")
                delay = self._retry_after(response.headers.get("Retry-After", "1"))
                if delay > self.max_retry_wait:
                    # Do not shorten Retry-After and hit the service prematurely.
                    raise RiotAPIError(f"Retry-After {delay}s exceeds configured retry budget")
                self.sleep(max(delay, min(self.max_retry_wait, 2**attempt)))
                continue
            if not response.ok:
                raise RiotAPIError(
                    f"Riot HTTP {response.status_code} for {path}; check key/routing"
                )
            try:
                return response.json()
            except ValueError:
                raise RiotAPIError("Riot returned invalid JSON") from None
        raise AssertionError("Unreachable")

    def fetch_match_ids(
        self, puuid: str, *, start: int = 0, count: int = 20, queue: int | None = 420
    ) -> list[str]:
        if not puuid or start < 0 or not 1 <= count <= 100:
            raise ValueError("PUUID required; start >= 0 and count in [1,100]")
        params = {"start": start, "count": count}
        if queue is not None:
            params["queue"] = queue
        result = self._get(f"/lol/match/v5/matches/by-puuid/{quote(puuid, safe='')}/ids", params)
        if not isinstance(result, list) or not all(isinstance(x, str) for x in result):
            raise RiotAPIError("Malformed match ID response")
        return result

    def fetch_match(self, match_id: str) -> dict:
        return self._get(f"/lol/match/v5/matches/{quote(match_id, safe='')}")

    def fetch_ranked_accounts(self, platform: str, tier: str) -> list[dict]:
        """Returned current top-tier roster; current tier is not historical match rank."""
        if tier not in {"MASTER", "GRANDMASTER", "CHALLENGER"}:
            raise ValueError("Use MASTER, GRANDMASTER or CHALLENGER")
        result = self._get(
            f"/lol/league/v4/{tier.lower()}leagues/by-queue/RANKED_SOLO_5x5", platform=platform
        )
        if not isinstance(result, dict) or not isinstance(result.get("entries"), list):
            raise RiotAPIError("Malformed League-V4 roster")
        if any(
            not isinstance(entry, dict) or not entry.get("puuid") for entry in result["entries"]
        ):
            raise RiotAPIError("League-V4 roster lacks verified PUUIDs; no identity fallback")
        return result["entries"]

    def resolve_account(self, riot_id: str) -> dict:
        game_name, separator, tag_line = riot_id.rpartition("#")
        if not separator or not game_name or not tag_line:
            raise ValueError("Riot ID must be gameName#tagLine")
        result = self._get(
            f"/riot/account/v1/accounts/by-riot-id/{quote(game_name, safe='')}/"
            f"{quote(tag_line, safe='')}"
        )
        if not isinstance(result, dict) or not result.get("puuid"):
            raise RiotAPIError("Malformed account response")
        return result

    def fetch_timeline(self, match_id: str) -> dict:
        return self._get(f"/lol/match/v5/matches/{quote(match_id, safe='')}/timeline")

    def close(self):
        self.session.close()


class Collector:
    def __init__(self, raw, artifacts):
        self.raw, self.artifacts = Path(raw), Path(artifacts)

    def collect(self, player, *, count=100, start=0, queue=420, api=None):
        if count < 1 or start < 0:
            raise ValueError("Positive count and nonnegative start required")
        own_api = api is None
        api = api or RiotClient(os.environ.get("RIOT_API_KEY", ""), player["region"])
        report = {
            "batch_id": run_id(),
            "status": "running",
            "player": player,
            "start": start,
            "requested_count": count,
            "discovered": [],
            "downloaded": [],
            "reused": [],
        }
        report_path = self.artifacts / "batches" / (report["batch_id"] + ".json")
        atomic_json(report_path, report)
        try:
            for offset in range(0, count, 100):
                limit = min(100, count - offset)
                ids = api.fetch_match_ids(
                    player["puuid"], start=start + offset, count=limit, queue=queue
                )
                for match_id in ids:
                    if not re.fullmatch(r"[A-Za-z0-9_-]+", match_id):
                        raise ValueError("Invalid match ID")
                    if match_id in report["discovered"]:
                        continue
                    report["discovered"].append(match_id)
                    path = self.raw / (match_id + ".json")
                    if path.exists():
                        payload = read_json(path)
                        if (
                            payload["match"]["metadata"]["matchId"] != match_id
                            or not payload["timeline"]["info"]["frames"]
                        ):
                            raise ValueError("Invalid cached match preserved for inspection")
                        report["reused"].append(match_id)
                    else:
                        match = api.fetch_match(match_id)
                        timeline = api.fetch_timeline(match_id)
                        if (
                            match["metadata"]["matchId"] != match_id
                            or timeline.get("metadata", {}).get("matchId", match_id) != match_id
                            or not any(
                                p.get("puuid") == player["puuid"]
                                for p in match["info"]["participants"]
                            )
                        ):
                            raise ValueError("Downloaded match does not bind configured identity")
                        atomic_json(path, {"match": match, "timeline": timeline})
                        report["downloaded"].append(match_id)
                    atomic_json(report_path, report)
                if len(ids) < limit:
                    break
            report["status"] = "complete"
            return report
        except BaseException as error:
            report.update(
                status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                error=str(error),
            )
            raise
        finally:
            atomic_json(report_path, report)
            if own_api:
                api.close()
