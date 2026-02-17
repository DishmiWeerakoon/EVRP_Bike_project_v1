from __future__ import annotations
import math
from typing import Any
from data.esogu_parser import Instance


# ----------------------------
# Coordinate extraction
# ----------------------------

def _node_latlon(node: Any) -> tuple[float, float]:
    """
    Extract latitude/longitude from Node.
    ESOGU txt stores coordinates like 39750124 (micro-degrees).
    """
    lat = getattr(node, "latitude", None)
    lon = getattr(node, "longitude", None)

    if lat is None:
        lat = getattr(node, "lat", None)
    if lon is None:
        lon = getattr(node, "lon", None)

    if lat is None:
        lat = getattr(node, "x", None)
    if lon is None:
        lon = getattr(node, "y", None)

    if lat is None or lon is None:
        raise AttributeError("Node has no coordinate attributes.")

    lat = float(lat)
    lon = float(lon)

    # Convert micro-degrees to degrees if needed
    if abs(lat) > 1000 or abs(lon) > 1000:
        lat /= 1_000_000.0
        lon /= 1_000_000.0

    return lat, lon


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# ----------------------------
# Distance
# ----------------------------

def dist_km(inst: Instance, a: str, b: str) -> float:
    if a == b:
        return 0.0

    # 1) direct matrix lookup
    row = inst.dist.get(a)
    if row and b in row:
        return float(row[b])

    # 2) reverse lookup
    row2 = inst.dist.get(b)
    if row2 and a in row2:
        return float(row2[a])

    # 3) fallback to haversine from coordinates
    if a in inst.nodes and b in inst.nodes:
        na = inst.nodes[a]
        nb = inst.nodes[b]
        lat1, lon1 = _node_latlon(na)
        lat2, lon2 = _node_latlon(nb)
        return _haversine_km(lat1, lon1, lat2, lon2)

    raise KeyError(f"Distance not found for pair ({a} -> {b}).")


# ----------------------------
# Travel Time
# ----------------------------

def travel_time_min(inst: Instance, distance_km: float) -> float:
    """
    Uses v_mps from Params (convert to km/h)
    """
    v_mps = inst.params.v_mps
    if v_mps <= 0:
        return 1e9

    v_kmph = v_mps * 3.6  # convert m/s → km/h
    return (distance_km / v_kmph) * 60.0


# ----------------------------
# Energy
# ----------------------------

def energy_need_kwh(inst: Instance, a, b=None) -> float:
    """
    Supports:
      energy_need_kwh(inst, distance_km)
      energy_need_kwh(inst, from_id, to_id)
    """
    if b is None:
        # called with distance
        distance_km = float(a)
    else:
        # called with node IDs
        distance_km = dist_km(inst, str(a), str(b))

    rate = inst.params.consumption_rate  # kWh per km
    return distance_km * rate
