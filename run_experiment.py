import os, re
from data.esogu_parser import parse_esogu_txt, Instance, Params
from data.distance_loader import load_esogu_distance_matrix

# your algorithm imports
from alg.ga import run_ga, GAConfig
from alg.baselines import baseline_random_assignment, baseline_distance_greedy
from core.evaluator import simulate_plan

def sheet_from_filename(fname: str) -> str:
    # ESOGU_C20_TW3.txt -> c20, ESOGU_RC40_TW1.txt -> rc40
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

    # sanity: ensure ids exist in matrix
    for nid in [depot_id] + request_ids[:3] + charging_ids[:3]:
        if nid not in dist:
            raise KeyError(f"Node {nid} not found in distance matrix sheet '{sheet}'")

    return Instance(
        name=name,
        nodes=nodes,
        depot_id=depot_id,
        request_ids=request_ids,
        charging_ids=charging_ids,
        params=Params(),   # you can override here if needed
        dist=dist
    )

def print_result(tag, sim):
    print(f"\n--- {tag} ---")
    print(f"Feasible: {sim.feasible}")
    print(f"Total Distance (km): {sim.total_dist_km:.3f}")
    print(f"Total Time (min): {sim.total_time_min:.2f}")
    print(f"Charging Stops: {sim.charge_stops}")
    print(f"Late Count: {sim.late_count}")
    print(f"Load Violations: {sim.load_violations}")
    print(f"Battery Violations: {sim.battery_violations}")

def main():
    BASE = os.path.dirname(os.path.abspath(__file__))

    # dataset paths (match your layout)
    txt_folder = os.path.join(BASE, "dataset", "ESOGU-EVRP-PDP-TW")
    excel_path = os.path.join(BASE, "dataset", "distance_matrix.xlsx")

    # choose any instance here
    txt_path = os.path.join(txt_folder, "ESOGU_C5_TW1.txt")
    inst = build_instance(txt_path, excel_path)

    bikes = 3

    sim_rnd, _ = baseline_random_assignment(inst, bikes=bikes, seed=1)
    print_result("Baseline: Random Assignment", sim_rnd)

    sim_greedy, _ = baseline_distance_greedy(inst, bikes=bikes)
    print_result("Baseline: Distance Greedy", sim_greedy)

    cfg = GAConfig(bikes=bikes, pop_size=80, generations=150)
    best_routes, best_fit = run_ga(inst, cfg)
    sim_best = simulate_plan(inst, best_routes)
    print_result("Proposed: GA + Dynamic Partial Charging", sim_best)

    print("\nBest routes:")
    for i, r in enumerate(best_routes):
        print(f"Bike {i+1}: {r}")

if __name__ == "__main__":
    main()
