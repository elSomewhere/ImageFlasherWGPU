"""Random Wikidata items that expose official-site (P856) URLs."""
from __future__ import annotations

import json
from urllib.parse import urlencode

from ...core.types import SeedNode


class WikidataOfficialSeedSource:
    name = "wikidata_official_sites"

    def __init__(self, world) -> None:
        self.world = world

    async def poll(self, limit: int = 1) -> list[SeedNode]:
        random_query = urlencode(
            {
                "action": "query",
                "generator": "random",
                "grnnamespace": "0",
                "grnlimit": str(max(1, min(limit * 3, 10))),
                "prop": "info",
                "maxlag": "1",
                "format": "json",
            }
        )
        resource = await self.world.fetch(f"https://www.wikidata.org/w/api.php?{random_query}")
        random_payload = json.loads(resource.body.decode("utf-8"))
        ids = [page.get("title") for page in random_payload.get("query", {}).get("pages", {}).values()]
        ids = [item for item in ids if item]
        if not ids:
            return []
        entity_query = urlencode(
            {
                "action": "wbgetentities",
                "ids": "|".join(ids),
                "props": "claims|labels",
                "languages": "en",
                "maxlag": "1",
                "format": "json",
            }
        )
        resource = await self.world.fetch(f"https://www.wikidata.org/w/api.php?{entity_query}")
        payload = json.loads(resource.body.decode("utf-8"))
        seeds: list[SeedNode] = []
        for entity in payload.get("entities", {}).values():
            label = (entity.get("labels", {}).get("en", {}) or {}).get("value", "")
            for claim in entity.get("claims", {}).get("P856", []):
                value = (((claim.get("mainsnak") or {}).get("datavalue") or {}).get("value"))
                if isinstance(value, str):
                    seeds.append(SeedNode(url=value, context=label, source=self.name))
                    if len(seeds) >= limit:
                        return seeds
        return seeds
