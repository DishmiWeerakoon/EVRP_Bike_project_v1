from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple

from data.esogu_parser import Instance
from core.evaluator import simulate_plan


# ----------------------------
# Config
# ----------------------------
@dataclass
class GAConfig:
    bikes: int = 3
    pop_size: int = 80
    generations: int = 150
    cx_rate: float = 0.8
    mut_rate: float = 0.25
    tournament_k: int = 3
    seed: int | None = None

    # objective weights (same as yours)
    w_dist: float = 1.0
    w_time: float = 0.8
    w_stops: float = 10.0
    w_late: float = 2000.0
    w_loadviol: float = 400.0
    w_battviol: float = 2000.0
    w_unserved: float = 10000.0
    w_balance: float = 2.0   # tune 10..200 (bigger = more balanced)



# ----------------------------
# Representation
#   Individual = permutation of all customer IDs: List[str]
#   Decode(permutation) -> routes: List[List[str]] with length = bikes
# ----------------------------
Individual = List[str]
Routes = List[List[str]]


def _make_rng(cfg: GAConfig) -> random.Random:
    return random.Random(cfg.seed)


def _random_individual(inst: Instance, rng: random.Random) -> Individual:
    perm = inst.request_ids[:]
    rng.shuffle(perm)
    return perm


from core.energy_time import dist_km, travel_time_min

def _decode_to_routes(inst: Instance, perm: Individual, bikes: int) -> Routes:
    if bikes <= 0:
        raise ValueError("bikes must be >= 1")

    routes: Routes = [[] for _ in range(bikes)]
    cur = [inst.depot_id for _ in range(bikes)]
    t_now = [0.0 for _ in range(bikes)]

    # keep your load feasibility penalty (soft) but DO NOT hard-balance
    cur_load = [0.0 for _ in range(bikes)]
    cap = inst.params.load_cap_kg

    LATE_W = 200.0
    WAIT_W = 10.0
    LOAD_W = 50.0
    NEAR_CAP_W = 5.0

    # ✅ only encourage non-empty bikes early (very light)
    EMPTY_W = 200.0  # tune 50..500 (only affects first few assignments)

    for idx, cust in enumerate(perm):
        node = inst.nodes[cust]

        best_b = 0
        best_score = float("inf")

        for b in range(bikes):
            d = dist_km(inst, cur[b], cust)
            t_travel = travel_time_min(inst, d)

            arrival = t_now[b] + t_travel

            wait = 0.0
            if arrival < node.tw_earliest:
                wait = node.tw_earliest - arrival
                arrival = node.tw_earliest

            late = 0.0
            if arrival > node.tw_latest:
                late = arrival - node.tw_latest

            finish = arrival + node.service_time

            proj_load = cur_load[b] + node.P - node.D
            if proj_load > cap:
                continue

            util = proj_load / cap if cap > 0 else 0.0
            load_pen = LOAD_W * util + NEAR_CAP_W * (util**4)

            # ✅ small bias to use empty bikes early
            empty_pen = 0.0
            if idx < bikes * 5 and len(routes[b]) == 0:  # only early stage
                empty_pen = -EMPTY_W  # reward empty bikes

            score = finish + WAIT_W * wait + LATE_W * late + load_pen + empty_pen

            if score < best_score:
                best_score = score
                best_b = b

        routes[best_b].append(cust)

        # update
        d = dist_km(inst, cur[best_b], cust)
        t_travel = travel_time_min(inst, d)

        arrival = t_now[best_b] + t_travel
        if arrival < node.tw_earliest:
            arrival = node.tw_earliest

        t_now[best_b] = arrival + node.service_time
        cur[best_b] = cust
        cur_load[best_b] += node.P - node.D

    return routes

def _assert_valid_perm(inst: Instance, perm: Individual) -> None:
    reqs = set(inst.request_ids)
    if len(perm) != len(inst.request_ids):
        raise RuntimeError(f"Bad GA individual length: {len(perm)} != {len(inst.request_ids)}")
    if set(perm) != reqs:
        missing = list(reqs - set(perm))[:10]
        extra = list(set(perm) - reqs)[:10]
        raise RuntimeError(f"Bad GA individual content. Missing={missing}, Extra={extra}")


def _fitness(inst: Instance, perm: Individual, cfg: GAConfig) -> float:
    _assert_valid_perm(inst, perm)
    routes = _decode_to_routes(inst, perm, cfg.bikes)

    # balance by customers count (stable + simple)
    lens = [len(r) for r in routes]
    bal_count = max(lens) - min(lens)

    # balance by delivery preload estimate (optional)
    del_loads = [sum(inst.nodes[c].D for c in r) for r in routes]
    bal_del = (max(del_loads) - min(del_loads)) / max(1.0, inst.params.load_cap_kg)  # normalized

    balance_pen = bal_count + 0.5 * bal_del

    sim, _ = simulate_plan(
        inst, routes,
        charging_policy="dynamic",
        fixed_target_soc=0.80,
        return_trace=False
    )

    # ✅ Balance metric based on DELIVERY preload per bike (matches simulator logic)
    del_loads = [sum(inst.nodes[c].D for c in r) for r in routes]
    balance_pen = (max(del_loads) - min(del_loads))  # kg spread

    return (
        cfg.w_dist * sim.total_dist_km +
        cfg.w_time * sim.total_time_min +
        cfg.w_stops * sim.charge_stops +
        cfg.w_late * sim.late_count +
        cfg.w_loadviol * sim.load_violations +
        cfg.w_battviol * sim.battery_violations +
        cfg.w_unserved * sim.unserved +
        cfg.w_balance * balance_pen
    )


    return (
        cfg.w_dist * sim.total_dist_km +
        cfg.w_time * sim.total_time_min +
        cfg.w_stops * sim.charge_stops +
        cfg.w_late * sim.late_count +
        cfg.w_loadviol * sim.load_violations +
        cfg.w_battviol * sim.battery_violations
        + cfg.w_unserved * sim.unserved
    )


# ----------------------------
# Selection
# ----------------------------
def _tournament(pop: List[Individual], fit: List[float], k: int, rng: random.Random) -> Individual:
    best_i = None
    best_f = float("inf")
    for _ in range(k):
        i = rng.randrange(len(pop))
        if fit[i] < best_f:
            best_f = fit[i]
            best_i = i
    return pop[best_i][:]  # copy


# ----------------------------
# Crossover (Order Crossover - OX)
# Keeps permutation validity always.
# ----------------------------
def _ox_crossover(p1: Individual, p2: Individual, rng: random.Random) -> Tuple[Individual, Individual]:
    n = len(p1)
    if n < 2:
        return p1[:], p2[:]

    a, b = sorted(rng.sample(range(n), 2))

    def ox(parent_a: Individual, parent_b: Individual) -> Individual:
        child = [None] * n  # type: ignore
        # copy slice from parent_a
        child[a:b+1] = parent_a[a:b+1]
        used = set(child[a:b+1])

        # fill remaining positions with order from parent_b
        pos = (b + 1) % n
        for gene in parent_b:
            if gene in used:
                continue
            child[pos] = gene
            pos = (pos + 1) % n
        return child  # type: ignore

    c1 = ox(p1, p2)
    c2 = ox(p2, p1)
    return c1, c2


# ----------------------------
# Mutation (safe permutation mutations)
# ----------------------------
def _mutate(perm: Individual, p: float, rng: random.Random) -> None:
    if rng.random() > p or len(perm) < 2:
        return

    r = rng.random()

    if r < 0.34:
        # swap
        i, j = rng.sample(range(len(perm)), 2)
        perm[i], perm[j] = perm[j], perm[i]

    elif r < 0.67:
        # inversion
        i, j = sorted(rng.sample(range(len(perm)), 2))
        perm[i:j+1] = reversed(perm[i:j+1])

    else:
        # insertion (remove one, insert elsewhere)
        i, j = rng.sample(range(len(perm)), 2)
        gene = perm.pop(i)
        perm.insert(j, gene)


# ----------------------------
# Main GA
# Returns ROUTES (decoded) + best fitness
# ----------------------------
def run_ga(inst: Instance, cfg: GAConfig) -> Tuple[Routes, float]:
    rng = _make_rng(cfg)

    pop: List[Individual] = [_random_individual(inst, rng) for _ in range(cfg.pop_size)]
    fit: List[float] = [_fitness(inst, ind, cfg) for ind in pop]

    best_i = min(range(len(pop)), key=lambda i: fit[i])
    best_perm = pop[best_i][:]
    best_f = fit[best_i]

    for _gen in range(cfg.generations):
        new_pop: List[Individual] = []

        # elitism (keep best)
        new_pop.append(best_perm[:])

        while len(new_pop) < cfg.pop_size:
            p1 = _tournament(pop, fit, cfg.tournament_k, rng)
            p2 = _tournament(pop, fit, cfg.tournament_k, rng)

            if rng.random() < cfg.cx_rate:
                c1, c2 = _ox_crossover(p1, p2, rng)
            else:
                c1, c2 = p1[:], p2[:]

            _mutate(c1, cfg.mut_rate, rng)
            _mutate(c2, cfg.mut_rate, rng)

            new_pop.append(c1)
            if len(new_pop) < cfg.pop_size:
                new_pop.append(c2)

        pop = new_pop
        fit = [_fitness(inst, ind, cfg) for ind in pop]

        bi = min(range(len(pop)), key=lambda i: fit[i])
        if fit[bi] < best_f:
            best_f = fit[bi]
            best_perm = pop[bi][:]

    # final decode
    routes = _decode_to_routes(inst, best_perm, cfg.bikes)

    # final safety check: must serve all customers exactly once
    served = [x for r in routes for x in r]
    if len(served) != len(inst.request_ids) or set(served) != set(inst.request_ids):
        raise RuntimeError("Decoded routes do not serve all customers exactly once (GA bug).")

    return routes, best_f
