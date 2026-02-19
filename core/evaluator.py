from __future__ import annotations
from dataclasses import dataclass
from typing import List, Set, Dict, Any, Tuple
from data.esogu_parser import Instance
from core.energy_time import dist_km, travel_time_min, energy_need_kwh
from core.charging_policies import maybe_charge
from dataclasses import dataclass, field
from typing import List


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
    unserved: int
    bike_dist_km: List[float] = field(default_factory=list)
    bike_total_time_min: List[float] = field(default_factory=list)
    bike_travel_time_min: List[float] = field(default_factory=list)
    bike_charge_time_min: List[float] = field(default_factory=list)
    bike_customers: List[int] = field(default_factory=list)
    makespan_min: float = 0.0


def simulate_plan(
    inst: Instance,
    bike_routes: List[List[str]],
    charging_policy: str = "dynamic",
    fixed_target_soc: float = 0.80,
    return_trace: bool = False, initial_soc: float = 1.0) -> Tuple[SimResult, List[Dict[str, Any]]]:
    """
    If return_trace=True, returns (SimResult, trace_rows).
    trace_rows is a list of dict rows you can write directly to CSV.
    """
    depot = inst.depot_id

    total_dist = 0.0
    total_time = 0.0
    travel_time_total = 0.0
    charge_time_total = 0.0
    charge_stops = 0

    late_count = 0
    load_viol = 0
    batt_viol = 0

    served: Set[str] = set()

    trace_rows: List[Dict[str, Any]] = []

    B = len(bike_routes)

    bike_dist = [0.0] * B
    bike_total_time = [0.0] * B
    bike_travel_time = [0.0] * B
    bike_charge_time = [0.0] * B
    bike_customers = [0] * B

    def log_event(**kw):
        if return_trace:
            trace_rows.append(kw)

    for bike_idx, route in enumerate(bike_routes):
        seq = route[:]
        if not seq or seq[0] != depot:
            seq = [depot] + seq
        if seq[-1] != depot:
            seq = seq + [depot]


        
        t = 0.0
        initial_soc = max(0.0, min(1.0, initial_soc))
        soc_kwh = inst.params.battery_kwh * initial_soc
        cur = seq[0]

        # ✅ Start-of-route preload: vehicle leaves depot carrying all deliveries assigned to this route
        # (Only count customer nodes)
        route_customers = [nid for nid in seq if nid in inst.nodes and inst.nodes[nid].node_type == "c"]
        current_load = sum(inst.nodes[nid].D for nid in route_customers)

        # Capacity check at departure (if too big, route is infeasible)
        if current_load > inst.params.load_cap_kg:
            load_viol += 1

        # log start
        n0 = inst.nodes[cur]
        log_event(
            instance=inst.name,
            bike_id=bike_idx,
            event="start",
            from_id=cur, to_id=cur,
            node_type=n0.node_type,
            lat=n0.lat, lon=n0.lon,
            depart_time_min=t,
            arrive_time_min=t,
            soc_before_kwh=soc_kwh,
            soc_after_kwh=soc_kwh,
            dist_km=0.0,
            travel_time_min=0.0,
            wait_time_min=0.0,
            service_time_min=0.0,
            charge_time_min=0.0,
            charge_added_kwh=0.0,
            late=0,
            feasible_so_far=1,
            current_load=current_load,
            capacity=inst.params.load_cap_kg,
        )

        for i in range(1, len(seq)):
            nxt = seq[i]
            remaining = seq[i:]  # nxt..end

            # --------------- CHARGE DECISION ---------------
            soc_before_charge = soc_kwh
            cur_before_charge = cur
            t_before_charge = t

            new_cur, soc_kwh, t_detour, t_charge, d_detour, did_charge, bv = maybe_charge(
                inst,
                current_id=cur,
                soc_kwh=soc_kwh,
                remaining_route=remaining,
                policy=charging_policy,
                fixed_target_soc=fixed_target_soc,
                reserve_kwh=0.10,
            )

            if bv:
                batt_viol += 1
                log_event(
                    instance=inst.name,
                    bike_id=bike_idx,
                    event="battery_violation",
                    from_id=cur, to_id=cur,
                    node_type=inst.nodes[cur].node_type,
                    lat=inst.nodes[cur].lat, lon=inst.nodes[cur].lon,
                    depart_time_min=t,
                    arrive_time_min=t,
                    soc_before_kwh=soc_kwh,
                    soc_after_kwh=soc_kwh,
                    dist_km=0.0,
                    travel_time_min=0.0,
                    wait_time_min=0.0,
                    service_time_min=0.0,
                    charge_time_min=0.0,
                    charge_added_kwh=0.0,
                    late=0,
                    feasible_so_far=0,
                    current_load=current_load,
                    capacity=inst.params.load_cap_kg,
                )
                break

            if did_charge:
                # detour travel + charge event log
                charge_stops += 1
                total_dist += d_detour
                total_time += (t_detour + t_charge)
                travel_time_total += t_detour
                charge_time_total += t_charge
                bike_dist[bike_idx] += d_detour
                bike_total_time[bike_idx] += (t_detour + t_charge)
                bike_travel_time[bike_idx] += t_detour
                bike_charge_time[bike_idx] += t_charge

                # compute charge added
                charge_added = max(0.0, soc_kwh - soc_before_charge)  # includes travel-to-cs deduction already applied in maybe_charge

                # update time and location
                t += (t_detour + t_charge)
                cur = new_cur

                ncs = inst.nodes[cur]
                log_event(
                    instance=inst.name,
                    bike_id=bike_idx,
                    event="charge",
                    from_id=cur_before_charge, to_id=cur,
                    node_type=ncs.node_type,
                    lat=ncs.lat, lon=ncs.lon,
                    depart_time_min=t_before_charge,
                    arrive_time_min=t_before_charge + t_detour,
                    soc_before_kwh=soc_before_charge,
                    soc_after_kwh=soc_kwh,
                    dist_km=d_detour,
                    travel_time_min=t_detour,
                    wait_time_min=0.0,
                    service_time_min=0.0,
                    charge_time_min=t_charge,
                    charge_added_kwh=charge_added,
                    late=0,
                    feasible_so_far=1,
                    current_load=current_load,
                    capacity=inst.params.load_cap_kg,
                )

                # --------------- PRE-LEG FEASIBILITY GUARDRAIL ---------------
                RESERVE_KWH = 0.10  # recommended 0.10; try 0.15 for hard R instances

                e_leg = energy_need_kwh(inst, cur, nxt)  # energy needed for the immediate next leg

                if soc_kwh < e_leg + RESERVE_KWH:
                    # force a charge BEFORE attempting the leg
                    soc_before_charge2 = soc_kwh
                    cur_before_charge2 = cur
                    t_before_charge2 = t

                    new_cur2, soc_kwh, t_detour2, t_charge2, d_detour2, did_charge2, bv2 = maybe_charge(
                        inst,
                        current_id=cur,
                        soc_kwh=soc_kwh,
                        remaining_route=remaining,        # includes nxt..end already
                        policy=charging_policy,
                        fixed_target_soc=fixed_target_soc,
                        reserve_kwh=RESERVE_KWH,
                    )

                    if bv2:
                        batt_viol += 1
                        log_event(
                            instance=inst.name,
                            bike_id=bike_idx,
                            event="battery_violation",
                            from_id=cur, to_id=cur,
                            node_type=inst.nodes[cur].node_type,
                            lat=inst.nodes[cur].lat, lon=inst.nodes[cur].lon,
                            depart_time_min=t,
                            arrive_time_min=t,
                            soc_before_kwh=soc_kwh,
                            soc_after_kwh=soc_kwh,
                            dist_km=0.0,
                            travel_time_min=0.0,
                            wait_time_min=0.0,
                            service_time_min=0.0,
                            charge_time_min=0.0,
                            charge_added_kwh=0.0,
                            late=0,
                            feasible_so_far=0,
                            current_load=current_load,
                            capacity=inst.params.load_cap_kg,
                        )
                        break

                    if did_charge2:
                        charge_stops += 1
                        total_dist += d_detour2
                        total_time += (t_detour2 + t_charge2)
                        travel_time_total += t_detour2
                        charge_time_total += t_charge2

                        bike_dist[bike_idx] += d_detour2
                        bike_total_time[bike_idx] += (t_detour2 + t_charge2)
                        bike_travel_time[bike_idx] += t_detour2
                        bike_charge_time[bike_idx] += t_charge2

                        charge_added2 = max(0.0, soc_kwh - soc_before_charge2)

                        t += (t_detour2 + t_charge2)
                        cur = new_cur2

                        ncs2 = inst.nodes[cur]
                        log_event(
                            instance=inst.name,
                            bike_id=bike_idx,
                            event="charge",
                            from_id=cur_before_charge2, to_id=cur,
                            node_type=ncs2.node_type,
                            lat=ncs2.lat, lon=ncs2.lon,
                            depart_time_min=t_before_charge2,
                            arrive_time_min=t_before_charge2 + t_detour2,
                            soc_before_kwh=soc_before_charge2,
                            soc_after_kwh=soc_kwh,
                            dist_km=d_detour2,
                            travel_time_min=t_detour2,
                            wait_time_min=0.0,
                            service_time_min=0.0,
                            charge_time_min=t_charge2,
                            charge_added_kwh=charge_added2,
                            late=0,
                            feasible_so_far=1,
                            current_load=current_load,
                            capacity=inst.params.load_cap_kg,
                        )
                # ------------------------------------------------------------


            # --------------- TRAVEL cur -> nxt ---------------
            d = dist_km(inst, cur, nxt)
            e = energy_need_kwh(inst, cur, nxt)

            if soc_kwh < e:
                batt_viol += 1
                log_event(
                    instance=inst.name,
                    bike_id=bike_idx,
                    event="battery_violation",
                    from_id=cur, to_id=nxt,
                    node_type=inst.nodes[nxt].node_type,
                    lat=inst.nodes[cur].lat, lon=inst.nodes[cur].lon,
                    depart_time_min=t,
                    arrive_time_min=t,
                    soc_before_kwh=soc_kwh,
                    soc_after_kwh=soc_kwh,
                    dist_km=d,
                    travel_time_min=0.0,
                    wait_time_min=0.0,
                    service_time_min=0.0,
                    charge_time_min=0.0,
                    charge_added_kwh=0.0,
                    late=0,
                    feasible_so_far=0,
                    current_load=current_load,
                    capacity=inst.params.load_cap_kg,
                )
                break

            soc_before_travel = soc_kwh
            soc_kwh -= e
            t_leg = travel_time_min(inst, d)

            total_dist += d
            total_time += t_leg
            travel_time_total += t_leg
            t_depart = t
            t += t_leg

            bike_dist[bike_idx] += d
            bike_total_time[bike_idx] += t_leg
            bike_travel_time[bike_idx] += t_leg

            n_to = inst.nodes[nxt]
            log_event(
                instance=inst.name,
                bike_id=bike_idx,
                event="travel",
                from_id=cur, to_id=nxt,
                node_type=n_to.node_type,
                lat=n_to.lat, lon=n_to.lon,
                depart_time_min=t_depart,
                arrive_time_min=t,
                soc_before_kwh=soc_before_travel,
                soc_after_kwh=soc_kwh,
                dist_km=d,
                travel_time_min=t_leg,
                wait_time_min=0.0,
                service_time_min=0.0,
                charge_time_min=0.0,
                charge_added_kwh=0.0,
                late=0,
                feasible_so_far=1,
                current_load=current_load,
                capacity=inst.params.load_cap_kg,
            )

            # --------------- ARRIVAL PROCESSING ---------------
            if n_to.node_type == "c":
                served.add(nxt)

                # Apply delivery first (must have items to deliver)
                if current_load < n_to.D:
                    load_viol += 1
                    # still apply to keep simulation consistent
                current_load -= n_to.D

                # Apply pickup
                current_load += n_to.P

                # Enforce bounds
                if current_load > inst.params.load_cap_kg:
                    load_viol += 1

                if current_load < 0:
                    load_viol += 1  # should never happen now
                    current_load = 0.0

                wait = 0.0
                if t < n_to.tw_earliest:
                    wait = n_to.tw_earliest - t
                    t += wait
                    total_time += wait
                    bike_total_time[bike_idx] += wait

                late = 1 if t > n_to.tw_latest else 0
                if late:
                    late_count += 1

                # service
                t_service_start = t
                t += n_to.service_time
                total_time += n_to.service_time
                bike_total_time[bike_idx] += n_to.service_time
                bike_customers[bike_idx] += 1

                log_event(
                    instance=inst.name,
                    bike_id=bike_idx,
                    event="service",
                    from_id=nxt, to_id=nxt,
                    node_type=n_to.node_type,
                    lat=n_to.lat, lon=n_to.lon,
                    depart_time_min=t_service_start,
                    arrive_time_min=t_service_start,  # same point
                    soc_before_kwh=soc_kwh,
                    soc_after_kwh=soc_kwh,
                    dist_km=0.0,
                    travel_time_min=0.0,
                    wait_time_min=wait,
                    service_time_min=n_to.service_time,
                    charge_time_min=0.0,
                    charge_added_kwh=0.0,
                    late=late,
                    feasible_so_far=1,
                    current_load=current_load,
                    capacity=inst.params.load_cap_kg,
                )

            cur = nxt

    unserved = len(inst.request_ids) - len(served)
    feasible = (late_count == 0 and load_viol == 0 and batt_viol == 0 and unserved == 0)
    #load_violation_flag = 1 if current_load > inst.params.load_cap_kg else 0

    makespan_min = max(bike_total_time) if bike_total_time else 0.0


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
    unserved=unserved,
    bike_dist_km=bike_dist,
    bike_total_time_min=bike_total_time,
    bike_travel_time_min=bike_travel_time,
    bike_charge_time_min=bike_charge_time,
    bike_customers=bike_customers,
    makespan_min=makespan_min,   # ✅ ADD THIS
), trace_rows
