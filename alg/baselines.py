from __future__ import annotations
from typing import List, Tuple
import random

from data.esogu_parser import Instance
from core.evaluator import simulate_plan
from core.energy_time import dist_km


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

    sim = simulate_plan(inst, routes, charging_policy="dynamic", fixed_target_soc=0.80)
    return sim, routes


def baseline_distance_greedy(inst: Instance, bikes: int = 3):
    """
    Greedy nearest-neighbor assignment:
    - each bike starts at depot
    - repeatedly pick nearest unassigned customer to each bike's current position
    Uses dist_km() instead of inst.dist[][] to avoid KeyError when matrix is incomplete.
    """
    unassigned = set(inst.request_ids)

    # current location of each bike (start at depot)
    cur = [inst.depot_id for _ in range(bikes)]
    routes: List[List[str]] = [[] for _ in range(bikes)]

    # keep assigning until all customers assigned
    while unassigned:
        progressed = False

        for b in range(bikes):
            if not unassigned:
                break

            best_r = None
            best_d = 1e18

            # find nearest remaining customer from current position
            for r in unassigned:
                d = dist_km(inst, cur[b], r)  # ✅ safe distance
                if d < best_d:
                    best_d = d
                    best_r = r

            if best_r is None:
                continue

            routes[b].append(best_r)
            cur[b] = best_r
            unassigned.remove(best_r)
            progressed = True

        if not progressed:
            # should never happen, but avoid infinite loop
            # assign remaining arbitrarily
            rest = list(unassigned)
            for i, r in enumerate(rest):
                routes[i % bikes].append(r)
            break

    sim = simulate_plan(inst, routes, charging_policy="dynamic", fixed_target_soc=0.80)
    return sim, routes
