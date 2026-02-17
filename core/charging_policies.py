from __future__ import annotations
from typing import List, Tuple
from data.esogu_parser import Instance
from core.energy_time import dist_km, travel_time_min, energy_need_kwh


def charge_time_min(inst: Instance, energy_added_kwh: float) -> float:
    kw = inst.params.recharge_rate
    if kw <= 0:
        return 1e9
    hours = energy_added_kwh / kw
    return hours * 60.0


def nearest_station(inst: Instance, from_id: str) -> str:
    best = None
    best_d = 1e18
    for cs in inst.charging_ids:
        d = dist_km(inst, from_id, cs)
        if d < best_d:
            best_d = d
            best = cs
    if best is None:
        raise ValueError("No charging stations in instance.")
    return best


def _remaining_energy_need(inst: Instance, start_id: str, remaining_route: List[str]) -> float:
    need = 0.0
    prev = start_id
    for nxt in remaining_route:
        # ✅ energy_need_kwh expects (a, b), NOT distance
        need += energy_need_kwh(inst, prev, nxt)
        prev = nxt
    return need


def maybe_charge(
    inst: Instance,
    current_id: str,
    soc_kwh: float,
    remaining_route: List[str],
    policy: str = "dynamic",           # "dynamic" | "full" | "fixed"
    fixed_target_soc: float = 0.80,    # only for policy="fixed"
    reserve_kwh: float = 0.05
) -> Tuple[str, float, float, float, float, int, int]:
    """
    Charging policies (applied only if current battery can't finish remaining route approx):
      - dynamic: charge only required (need + reserve), capped by battery
      - full:    charge to 100% battery
      - fixed:   charge to fixed_target_soc * battery

    Returns:
      new_current_id,
      new_soc_kwh,
      detour_time_min,
      charge_time_min,
      detour_dist_km,
      did_charge (0/1),
      battery_violation (0/1)
    """
    H = 8  # rolling horizon: only plan energy for next 4 legs
    need = _remaining_energy_need(inst, current_id, remaining_route[:H]) + reserve_kwh

    # ---------- HYSTERESIS ----------
    HYST_KWH = 0.08  # prevents tiny recharge triggers

    if soc_kwh >= need - HYST_KWH:
        return current_id, soc_kwh, 0.0, 0.0, 0.0, 0, 0
    # --------------------------------
    # choose station
    cs = nearest_station(inst, current_id)
    d_to_cs = dist_km(inst, current_id, cs)

    # ✅ energy_need_kwh expects (a, b), NOT distance
    e_to_cs = energy_need_kwh(inst, current_id, cs)

    if soc_kwh < e_to_cs:
        # cannot reach station
        return current_id, soc_kwh, 0.0, 0.0, 0.0, 0, 1

    # travel to station
    soc_kwh -= e_to_cs
    t_detour = travel_time_min(inst, d_to_cs)

    # choose target SOC by policy
    if policy == "dynamic":
        BUFFER_KWH = 0.25  # tune 0.15–0.30
        target = min(inst.params.battery_kwh, need + BUFFER_KWH)
    elif policy == "full":
        target = inst.params.battery_kwh
    elif policy == "fixed":
        target = max(0.0, min(inst.params.battery_kwh, fixed_target_soc * inst.params.battery_kwh))
    else:
        raise ValueError(f"Unknown policy: {policy}")

    add = max(0.0, target - soc_kwh)
    t_charge = charge_time_min(inst, add)
    soc_kwh += add

    return cs, soc_kwh, t_detour, t_charge, d_to_cs, 1, 0
