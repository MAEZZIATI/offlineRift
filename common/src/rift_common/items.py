"""Patch-matched inventory valuation, with explicit unknown catalog entries."""


class ItemCatalog:
    """Data Dragon item JSON. Values are inventory investment proxies, not DPS."""

    def __init__(self, data: dict | None = None):
        self.data = (data or {}).get("data", data or {})

    def value(self, item_id: int) -> float:
        return float(self.data.get(str(item_id), {}).get("gold", {}).get("total", 0))

    def power(self, items: list[int], threshold: int) -> tuple[float, int, bool]:
        known = all(str(i) in self.data for i in items)
        return (
            sum(self.value(i) for i in items),
            sum(self.value(i) >= threshold for i in items),
            known,
        )
