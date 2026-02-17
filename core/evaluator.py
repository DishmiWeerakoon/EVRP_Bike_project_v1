from __future__ import annotations
from dataclasses import dataclass
from typing import List
from data.esogu_parser import Instance
from core.energy_time import dist_km, travel_time_min, energy_need_kwh
from core.charging_policies import maybe_charge


@dataclass
class SimResult:
    feasible: bool
    total_dist_km: float
    total_time_min: float
    travel_time_min: float
    charge_time_min: float
    charge_stops: int
    late_count: int
    load_violations: int
    battery_violations: int


def simulate_plan(
    inst: Instance,
    bike_routes: List[List[str]],
    charging_policy: str = "dynamic",     # "dynamic" | "full" | "fixed"
    fixed_target_soc: float = 0.80
) -> SimResult:
    depot = inst.depot_id

    total_dist = 0.0
    total_time = 0.0
    travel_time_total = 0.0
    charge_time_total = 0.0
    charge_stops = 0

    late_count = 0
    load_viol = 0
    batt_viol = 0

    for route in bike_routes:
        seq = route[:]
        if not seq or seq[0] != depot:
            seq = [depot] + seq
        if seq[-1] != depot:
            seq = seq + [depot]

        # simple load check
        load = 0.0
        for nid in seq:
            n = inst.nodes[nid]
            if n.node_type == "c":
                load += (n.D + n.P)
        if load > inst.params.load_cap_kg:
            load_viol += 1

        t = 0.0
        soc = inst.params.battery_kwh
        cur = seq[0]

        for i in range(1, len(seq)):
            nxt = seq[i]
            remaining = seq[i:]  # nxt..end

            # charging step (policy-based)
            new_cur, soc, t_detour, t_charge, d_detour, did_charge, bv = maybe_charge(
                inst,
                current_id=cur,
                soc_kwh=soc,
                remaining_route=remaining,
                policy=charging_policy,
                fixed_target_soc=fixed_target_soc,
                reserve_kwh=0.05,
            )

            if bv:
                batt_viol += 1
                break

            if did_charge:
                charge_stops += 1
                total_dist += d_detour
                total_time += (t_detour + t_charge)
                travel_time_total += t_detour
                charge_time_total += t_charge
                t += (t_detour + t_charge)
                cur = new_cur  # continue from station

            # travel cur -> nxt
            d = dist_km(inst, cur, nxt)
            e = energy_need_kwh(inst, d)
            if soc < e:
                batt_viol += 1
                break

            soc -= e
            t_leg = travel_time_min(inst, d)

            total_dist += d
            total_time += t_leg
            travel_time_total += t_leg
            t += t_leg

            node = inst.nodes[nxt]
            if node.node_type == "c":
                # waiting
                if t < node.tw_earliest:
                    wait = node.tw_earliest - t
                    t += wait
                    total_time += wait

                # late check
                if t > node.tw_latest:
                    late_count += 1

                # service
                t += node.service_time
                total_time += node.service_time

            cur = nxt

    feasible = (late_count == 0 and load_viol == 0 and batt_viol == 0)

    return SimResult(
        feasible=feasible,
        total_dist_km=total_dist,
        total_time_min=total_time,
        travel_time_min=travel_time_total,
        charge_time_min=charge_time_total,
        charge_stops=charge_stops,
        late_count=late_count,
        load_violations=load_viol,
        battery_violations=batt_viol,
    )
