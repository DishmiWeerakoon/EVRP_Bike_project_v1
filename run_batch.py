import os
from data.distance_loader import load_esogu_distance_matrix
from data.esogu_parser import parse_esogu_txt, Instance, Params
from alg.baselines import baseline_random_assignment, baseline_distance_greedy
from alg.ga import run_ga, GAConfig
from core.evaluator import simulate_plan


def sheet_from_filename(fname: str) -> str:
    # ESOGU_C20_TW3.txt -> c20, ESOGU_RC40_TW1.txt -> rc40
    import re
    m = re.search(r"ESOGU_(RC|C|R)(\d+)_TW", fname.upper())
    if not m:
        raise ValueError(f"Filename not recognized: {fname}")
    typ, n = m.group(1).lower(), m.group(2)
    return f"{typ}{n}"


def build_instance(txt_path: str, excel_path: str) -> Instance:
    name = os.path.basename(txt_path)
    sheet = sheet_from_filename(name)

    nodes = parse_esogu_txt(txt_path)
    depot_id = next(nid for nid, n in nodes.items() if n.node_type == "d")
    request_ids = [nid for nid, n in nodes.items() if n.node_type == "c"]
    charging_ids = [nid for nid, n in nodes.items() if n.node_type == "cs"]

    dist = load_esogu_distance_matrix(excel_path, sheet_name=sheet, to_km=True)

    # sanity check: ensure all required nodes exist in matrix
    must_exist = [depot_id] + request_ids + charging_ids
    missing = [nid for nid in must_exist if nid not in dist]
    if missing:
        raise KeyError(f"Missing IDs in matrix sheet '{sheet}': {missing[:10]} (and more)" if len(missing) > 10 else
                       f"Missing IDs in matrix sheet '{sheet}': {missing}")

    return Instance(
        name=name,
        nodes=nodes,
        depot_id=depot_id,
        request_ids=request_ids,
        charging_ids=charging_ids,
        params=Params(),   # adjust if you want bike-realistic params later
        dist=dist
    )


def fmt(sim):
    return (
        f"{'Y' if sim.feasible else 'N':>2} | "
        f"{sim.total_dist_km:8.2f} | "
        f"{sim.total_time_min:10.1f} | "
        f"{sim.charge_time_min:10.1f} | "
        f"{sim.charge_stops:5d} | "
        f"{sim.late_count:4d} | "
        f"{sim.load_violations:4d} | "
        f"{sim.battery_violations:4d}"
    )


def main():
    BASE = os.path.dirname(os.path.abspath(__file__))
    txt_folder = os.path.join(BASE, "dataset", "ESOGU-EVRP-PDP-TW")
    excel_path = os.path.join(BASE, "dataset", "distance_matrix.xlsx")

    # choose which sizes & TWs you want
    sizes = [5, 10, 20, 40, 60, 80, 100]
    tws = [1, 2, 3]
    types = ["C", "R", "RC"]  # clustered, random, mixed

    bikes = 3
    ga_cfg = GAConfig(bikes=bikes, pop_size=20, generations=50)

    print("Instance | Baseline(Random)                          | Baseline(Greedy)                           | Proposed(GA+DynamicCharge)")
    print("         | Fe |   Dist |     Time |  ChgTime | Stops | Late | LdV | BtV | Fe |   Dist |     Time |  ChgTime | Stops | Late | LdV | BtV | Fe |   Dist |     Time |  ChgTime | Stops | Late | LdV | BtV")
    print("-" * 170)

    for n in sizes:
        for tw in tws:
            for typ in types:
                fname = f"ESOGU_{typ}{n}_TW{tw}.txt"
                path = os.path.join(txt_folder, fname)
                if not os.path.exists(path):
                    continue

                inst = build_instance(path, excel_path)

                sim_rnd, _ = baseline_random_assignment(inst, bikes=bikes, seed=1)
                sim_grd, _ = baseline_distance_greedy(inst, bikes=bikes)

                best_routes, _ = run_ga(inst, ga_cfg)
                sim_ga = simulate_plan(inst, best_routes)

                label = f"{typ}{n}_TW{tw}"
                print(f"{label:10s} | {fmt(sim_rnd)} | {fmt(sim_grd)} | {fmt(sim_ga)}")


if __name__ == "__main__":
    main()
