from __future__ import annotations

from typing import Dict, List, Optional


LOCATION_CATALOG: List[Dict] = [
    {"location_id": "haikou_cn", "name": "Haikou", "country": "China", "region": "Hainan", "lat": 20.0442, "lon": 110.1999},
    {"location_id": "colombo_lk", "name": "Colombo", "country": "Sri Lanka", "region": "Western", "lat": 6.9271, "lon": 79.8612},
    {"location_id": "berlin_de", "name": "Berlin", "country": "Germany", "region": "Berlin", "lat": 52.5200, "lon": 13.4050},
    {"location_id": "paris_fr", "name": "Paris", "country": "France", "region": "Ile-de-France", "lat": 48.8566, "lon": 2.3522},
    {"location_id": "amsterdam_nl", "name": "Amsterdam", "country": "Netherlands", "region": "North Holland", "lat": 52.3676, "lon": 4.9041},
]


class LocationCatalog:
    def __init__(self, catalog: List[Dict] | None = None):
        self._catalog = list(catalog or LOCATION_CATALOG)
        self._by_id = {str(r["location_id"]): dict(r) for r in self._catalog}

    def list_locations(self) -> List[Dict]:
        return [dict(r) for r in self._catalog]

    def get_by_id(self, location_id: str) -> Optional[Dict]:
        if not location_id:
            return None
        row = self._by_id.get(str(location_id))
        return dict(row) if row else None
