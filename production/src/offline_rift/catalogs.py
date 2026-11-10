"""Exact-patch Data Dragon catalog resolution; no nearest-patch fallback."""

from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from rift_common.artifacts import atomic_json, file_digest, read_json
from rift_common.items import ItemCatalog
from urllib3.util.retry import Retry


def patch_of(version):
    parts = version.split(".")
    if len(parts) < 3 or not all(part.isdigit() for part in parts[:3]):
        raise ValueError("Malformed game version")
    return ".".join(parts[:2])


class CatalogCache:
    def __init__(self, directory, *, session=None):
        self.directory = Path(directory)
        self.session = session or requests.Session()
        if session is None:
            self.session.mount(
                "https://",
                HTTPAdapter(
                    max_retries=Retry(
                        total=4,
                        backoff_factor=1,
                        status_forcelist=[429, 500, 502, 503, 504],
                        respect_retry_after_header=True,
                    )
                ),
            )
        self.versions = None
        self.catalogs = {}

    def get(self, patch):
        if patch in self.catalogs:
            return self.catalogs[patch]
        path = self.directory / f"items-{patch}.json"
        if path.exists():
            data = read_json(path)
        else:
            if self.versions is None:
                response = self.session.get(
                    "https://ddragon.leagueoflegends.com/api/versions.json", timeout=30
                )
                response.raise_for_status()
                self.versions = response.json()
            matches = [version for version in self.versions if version.startswith(patch + ".")]
            if not matches:
                raise ValueError(f"No catalog for patch {patch}")
            version = max(matches, key=lambda v: tuple(map(int, v.split("."))))
            response = self.session.get(
                f"https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/item.json",
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            if not data.get("data") or data.get("version") != version:
                raise ValueError("Invalid downloaded catalog")
            atomic_json(path, data)
        if not data.get("data") or patch_of(data.get("version", "")) != patch:
            raise ValueError("Catalog patch mismatch")
        self.catalogs[patch] = (ItemCatalog(data), data["version"], file_digest(path))
        return self.catalogs[patch]

    def close(self):
        self.session.close()
