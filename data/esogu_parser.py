from dataclasses import dataclass
from typing import Dict, List, Optional
import re

@dataclass(frozen=True)
class Node:
    node_id: str
    node_type: str   # d, c, cs
    lat: float
    lon: float
    D: float
    P: float
    service_time: float
    tw_earliest: float
    tw_latest: float

@dataclass(frozen=True)
class Params:
    v_mps: float = 12.5
    load_cap_kg: float = 350.0
    battery_kwh: float = 1.0
    consumption_rate: float = 1.0
    recharge_rate: float = 0.18

@dataclass
class Instance:
    name: str
    nodes: Dict[str, Node]
    depot_id: str
    request_ids: List[str]
    charging_ids: List[str]
    params: Params
    dist: dict  # distance matrix dict-of-dicts

def parse_esogu_txt(path: str) -> Dict[str, Node]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    nodes: Dict[str, Node] = {}

    for ln in lines:
        if ln.lower().startswith("id") or ln.startswith("V ") or ln.startswith("C ") or ln.startswith("Q ") or ln.startswith("h(") or ln.startswith("g "):
            continue

        parts = re.split(r"\s+", ln)
        if len(parts) < 9:
            continue

        nid, ntype = parts[0], parts[1]
        lat, lon = float(parts[2]), float(parts[3])
        D, P = float(parts[4]), float(parts[5])
        st = float(parts[6])
        tw_e = float(parts[7])
        tw_l = float(parts[8])

        nodes[nid] = Node(nid, ntype, lat, lon, D, P, st, tw_e, tw_l)

    return nodes
