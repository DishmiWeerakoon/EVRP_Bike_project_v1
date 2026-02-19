from __future__ import annotations
from typing import List, Tuple, Optional
from data.esogu_parser import Instance
from core.energy_time import dist_km, travel_time_min, energy_need_kwh


def charge_time_min(inst: Instance, energy_added_kwh: float) -> float:
    g = inst.params.recharge_rate
    if g <= 0:
        return 1e9
    return (energy_added_kwh / g) * 60.0


def nearest_station(inst: Instance, from_id: str) -> str:
    best = None
    best_d = 1e18
    for cs in inst.charging_ids:
        d = dist_km(inst, from_id, cs)
        if d < best_d:
            best_d = d
            best = cs
    if best is None:
        raise ValueError("No charging stations found.")
    return best


def dynamic_partial_charge_if_needed(
    inst: Instance,
    current_id: str,
    soc_kwh: float,
    remaining_route: List[str],
    reserve_kwh: float = 0.05
) -> Tuple[str, float, float, float, float, int]:
    """
    If battery is insufficient to complete remaining_route (approx),
    detour to nearest charging station, charge minimally, and continue from that station.

    Returns:
      new_current_id (same as current_id if no charge, else chosen station id),
      new_soc_kwh,
      extra_travel_time_min (detour),
      extra_charge_time_min,
      extra_detour_dist_km,
      battery_violation (0/1)
    """

    # Estimate energy required to finish remaining route (from current_id through remaining_route)
    need = 0.0
    prev = current_id
    for nxt in remaining_route:
        dk = dist_km(inst, prev, nxt)
        need += energy_need_kwh(inst, dk)
        prev = nxt
    need += reserve_kwh

    if soc_kwh >= need:
        return current_id, soc_kwh, 0.0, 0.0, 0.0, 0

    # Detour to nearest station
    cs = nearest_station(inst, current_id)
    d_to_cs = dist_km(inst, current_id, cs)
    e_to_cs = energy_need_kwh(inst, d_to_cs)

    if soc_kwh < e_to_cs:
        # cannot reach the charging station
        return current_id, soc_kwh, 0.0, 0.0, 0.0, 1

    # travel to station
    soc_kwh -= e_to_cs
    t_detour = travel_time_min(inst, d_to_cs)

    # minimal charge to meet 'need' (cap by battery)
    target = min(inst.params.battery_kwh, need)
    add = max(0.0, target - soc_kwh)
    t_charge = charge_time_min(inst, add)

    soc_kwh += add

    return cs, soc_kwh, t_detour, t_charge, d_to_cs, 0
