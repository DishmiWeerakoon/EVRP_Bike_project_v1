from __future__ import annotations
from typing import List
import random

from data.esogu_parser import Instance
from core.evaluator import simulate_plan
from core.energy_time import dist_km, travel_time_min


def baseline_random_assignment(inst: Instance, bikes: int = 3, seed: int = 1):
    """
    Randomly assigns customers to bikes, keeps order random.
    Route format: list of routes, each route is list of node IDs (customers only).
    Depot/return handling is done inside simulate_plan (based on your evaluator logic).
    """
    rng = random.Random(seed)
    customers = list(inst.request_ids)
    rng.shuffle(customers)

    routes = [[] for _ in range(bikes)]
    for i, cid in enumerate(customers):
        routes[i % bikes].append(cid)

    sim, _ = simulate_plan(
        inst,
        routes,
        charging_policy="dynamic",
        fixed_target_soc=0.80,
        return_trace=False
    )
    return sim, routes


def baseline_distance_greedy(inst: Instance, bikes: int = 3):
    """
    Time-window-aware greedy baseline (fix for huge total_time_min).

    Old version selected nearest neighbor only -> arrived too early -> waited a lot.
    New version selects the next customer by a score that penalizes:
      - lateness heavily
      - waiting lightly
      - overall finish time
    """
    unassigned = set(inst.request_ids)

    # per-bike state
    cur = [inst.depot_id for _ in range(bikes)]
    t_now = [0.0 for _ in range(bikes)]  # minutes since start
    routes: List[List[str]] = [[] for _ in range(bikes)]

    # tune if needed (same spirit as your GA decode)
    LATE_W = 200.0   # 100..500
    WAIT_W = 10.0    # 1..30

    while unassigned:
        progressed = False

        for b in range(bikes):
            if not unassigned:
                break

            best_cid = None
            best_score = float("inf")

            for cid in unassigned:
                node = inst.nodes[cid]

                d = dist_km(inst, cur[b], cid)
                t_travel = travel_time_min(inst, d)
                arrival_raw = t_now[b] + t_travel

                # wait if early
                wait = 0.0
                arrival = arrival_raw
                if arrival < node.tw_earliest:
                    wait = node.tw_earliest - arrival
                    arrival = node.tw_earliest

                # late if beyond latest
                late = 0.0
                if arrival > node.tw_latest:
                    late = arrival - node.tw_latest

                finish = arrival + node.service_time

                # score prioritizes TW-feasibility
                score = finish + WAIT_W * wait + LATE_W * late

                if score < best_score:
                    best_score = score
                    best_cid = cid

            if best_cid is None:
                continue

            # assign
            routes[b].append(best_cid)

            # update bike state using the chosen customer
            node = inst.nodes[best_cid]
            d = dist_km(inst, cur[b], best_cid)
            t_travel = travel_time_min(inst, d)
            arrival = t_now[b] + t_travel
            if arrival < node.tw_earliest:
                arrival = node.tw_earliest

            t_now[b] = arrival + node.service_time
            cur[b] = best_cid
            unassigned.remove(best_cid)

            progressed = True

        if not progressed:
            # safety fallback: assign remaining arbitrarily (prevents infinite loop)
            rest = list(unassigned)
            for i, cid in enumerate(rest):
                routes[i % bikes].append(cid)
            break

    sim, _ = simulate_plan(
        inst,
        routes,
        charging_policy="dynamic",
        fixed_target_soc=0.80,
        return_trace=False
    )
    return sim, routes
