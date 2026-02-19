import os
import csv
import re
from typing import Dict, Tuple, List, Any, Optional

from alg.baselines import baseline_random_assignment, baseline_distance_greedy
from alg.ga import run_ga, GAConfig
from core.evaluator import simulate_plan
from data.esogu_parser import parse_esogu_txt, Instance, Params
from data.distance_loader import load_esogu_distance_matrix


# ----------------------------
# Per-bike terminal summary (only selected methods)
# ----------------------------
PRINT_METHODS = {
    "baseline_random_dynamic",
    "GA_routes_dynamic_charge",
    "GA_routes_full_charge",
    "GA_routes_fixed80_charge",
}

def print_bike_summary(instance_name: str, method: str, sim) -> None:
    if method not in PRINT_METHODS:
        return

    # If per-bike arrays are missing, don't crash
    if not getattr(sim, "bike_dist_km", None):
        print(f"\n=== {instance_name} | {method} ===")
        print("Per-bike arrays not found in SimResult (did you update simulate_plan return?)")
        return

    print(f"\n=== {instance_name} | {method} ===")
    for b in range(len(sim.bike_dist_km)):
        print(
            f"Bike {b}: "
            f"customers={sim.bike_customers[b]:3d} | "
            f"dist_km={sim.bike_dist_km[b]:7.2f} | "
            f"travel_min={sim.bike_travel_time_min[b]:7.2f} | "
            f"charge_min={sim.bike_charge_time_min[b]:7.2f} | "
            f"total_min={sim.bike_total_time_min[b]:7.2f}"
        )

# ----------------------------
# ID normalization
# ----------------------------

def variant_sibling_target(missing_id: str, dist_keys: set) -> Optional[str]:
    """
    If missing_id looks like '61A_2', try to map it to an existing sibling like '61A_1'.
    """
    m = re.match(r"^(.+)_\d+$", missing_id)
    if not m:
        return None
    base = m.group(1)

    siblings = [k for k in dist_keys if k == base or k.startswith(base + "_")]
    if not siblings:
        return None

    if base in siblings:
        return base

    def suffix_num(k: str) -> int:
        mm = re.match(r"^.+_(\d+)$", k)
        return int(mm.group(1)) if mm else 10**9

    siblings.sort(key=suffix_num)
    return siblings[0]


def normalize_id_str(x: object) -> str:
    s = str(x).strip()

    # Excel numeric artifacts like "33.0"
    if s.endswith(".0"):
        s = s[:-2]

    s = s.replace("\\", "/")
    s = s.replace("/", "_").replace("-", "_").replace(" ", "_")
    s = re.sub(r"_+", "_", s)
    s = s.upper().strip("_")

    # '60B1' -> '60B_1'
    s = re.sub(r"^(\d+)([A-Z])(\d+)$", r"\1\2_\3", s)

    return s


def base_id_candidates(nid: str) -> List[str]:
    s = normalize_id_str(nid)
    cands = [s]

    cands.append(re.sub(r"_\d+$", "", s))
    tmp = re.sub(r"_\d+$", "", s)
    cands.append(re.sub(r"[A-Z]$", "", tmp))
    cands.append(s.replace("_", ""))

    out = []
    for x in cands:
        x = x.strip("_")
        if x and x not in out:
            out.append(x)
    return out


# ----------------------------
# Distance matrix helpers
# ----------------------------
def normalize_dist_keys(dist: Dict[Any, Dict[Any, float]]) -> Dict[str, Dict[str, float]]:
    new: Dict[str, Dict[str, float]] = {}
    for a, row in dist.items():
        aa = normalize_id_str(a)
        if aa not in new:
            new[aa] = {}
        for b, val in row.items():
            bb = normalize_id_str(b)
            new[aa][bb] = val
    return new


def add_dist_alias(dist: Dict[str, Dict[str, float]], alias: str, target: str) -> None:
    """
    Makes alias behave exactly like target in dist[a][b].
    """
    if alias in dist:
        return
    dist[alias] = dist[target]
    for _, row in dist.items():
        if target in row:
            row[alias] = row[target]


# ----------------------------
# Robust coordinate extraction
# ----------------------------
def get_xy(node: Any) -> Optional[Tuple[float, float]]:
    """
    Try common coordinate representations.
    Returns (x, y) if found else None.
    """
    for ax, ay in [("x", "y"), ("X", "Y"), ("px", "py")]:
        if hasattr(node, ax) and hasattr(node, ay):
            try:
                return float(getattr(node, ax)), float(getattr(node, ay))
            except Exception:
                pass

    # your case: lat/lon
    for ax, ay in [("lon", "lat"), ("lng", "lat"), ("lat", "lon")]:
        if hasattr(node, ax) and hasattr(node, ay):
            try:
                return float(getattr(node, ax)), float(getattr(node, ay))
            except Exception:
                pass

    for attr in ["pos", "coord", "coords", "location", "xy"]:
        if hasattr(node, attr):
            v = getattr(node, attr)
            if isinstance(v, (tuple, list)) and len(v) >= 2:
                try:
                    return float(v[0]), float(v[1])
                except Exception:
                    pass

    return None


def coord_key(node: Any, ndigits: int = 6) -> Optional[Tuple[float, float]]:
    xy = get_xy(node)
    if xy is None:
        return None
    x, y = xy
    return (round(x, ndigits), round(y, ndigits))


def nearest_coord_target(
    missing_node: Any,
    coord_to_id_raw: Dict[str, Tuple[float, float]],
    tol: float = 1e-4,
) -> Optional[str]:
    xy = get_xy(missing_node)
    if xy is None:
        return None
    mx, my = xy

    best_id = None
    best_d2 = None
    tol2 = tol * tol

    for nid, (x, y) in coord_to_id_raw.items():
        dx = mx - x
        dy = my - y
        d2 = dx * dx + dy * dy
        if best_d2 is None or d2 < best_d2:
            best_d2 = d2
            best_id = nid

    if best_d2 is not None and best_d2 <= tol2:
        return best_id
    return None


# ----------------------------
# Instance builder
# ----------------------------
def build_instance(txt_path: str, excel_path: str) -> Instance:
    name = os.path.basename(txt_path)
    sheet = "distance v3.2"

    nodes_raw = parse_esogu_txt(txt_path)

    nodes: Dict[str, Any] = {}
    for nid, node in nodes_raw.items():
        nodes[normalize_id_str(nid)] = node

    depot_id = next(nid for nid, n in nodes.items() if n.node_type == "d")
    request_ids = [nid for nid, n in nodes.items() if n.node_type not in ("d", "cs")]
    charging_ids = [nid for nid, n in nodes.items() if n.node_type == "cs"]

    print("node types:", {t: sum(1 for x in nodes.values() if x.node_type == t)
                         for t in set(n.node_type for n in nodes.values())})

    if not request_ids:
        raise RuntimeError(f"No requests detected in {name}. Check parser node_type values.")

    dist = load_esogu_distance_matrix(excel_path, sheet_name=sheet, to_km=True)
    dist = normalize_dist_keys(dist)

    must_exist = [depot_id] + request_ids + charging_ids
    created: List[Tuple[str, str, str]] = []

    # Pass 1: string aliasing
    missing = [nid for nid in must_exist if nid not in dist]
    if missing:
        dist_keys = set(dist.keys())
        for nid in missing:
            target = None
            for cand in base_id_candidates(nid):
                if cand in dist_keys:
                    target = cand
                    break
            if target is not None:
                add_dist_alias(dist, nid, target)
                created.append((nid, target, "string"))

    # Pass 2: exact coordinate aliasing
    missing2 = [nid for nid in must_exist if nid not in dist]
    if missing2:
        coord_to_id_exact: Dict[Tuple[float, float], str] = {}
        coord_to_id_raw: Dict[str, Tuple[float, float]] = {}

        for nid, node in nodes.items():
            if nid in dist:
                xy = get_xy(node)
                if xy is not None:
                    coord_to_id_raw[nid] = xy
                ck = coord_key(node)
                if ck is not None:
                    coord_to_id_exact[ck] = nid

        for nid in list(missing2):
            node = nodes.get(nid)
            if node is None:
                continue

            ck = coord_key(node)
            if ck is not None and ck in coord_to_id_exact:
                target = coord_to_id_exact[ck]
                add_dist_alias(dist, nid, target)
                created.append((nid, target, "coord_exact"))

    # Pass 3: sibling variant aliasing
    missing3 = [nid for nid in must_exist if nid not in dist]
    if missing3:
        dist_keys = set(dist.keys())
        for nid in list(missing3):
            tgt = variant_sibling_target(nid, dist_keys)
            if tgt is not None and tgt in dist:
                add_dist_alias(dist, nid, tgt)
                created.append((nid, tgt, "variant_sibling"))

    # Pass 4: nearest-neighbor coordinate aliasing (tolerance)
    missing4 = [nid for nid in must_exist if nid not in dist]
    if missing4:
        coord_to_id_raw: Dict[str, Tuple[float, float]] = {}
        for nid, node in nodes.items():
            if nid in dist:
                xy = get_xy(node)
                if xy is not None:
                    coord_to_id_raw[nid] = xy

        for nid in list(missing4):  # ✅ FIX: keep this loop inside the if-block
            node = nodes.get(nid)
            if node is None:
                continue
            target = nearest_coord_target(node, coord_to_id_raw, tol=1e-3)
            if target is not None:
                add_dist_alias(dist, nid, target)
                created.append((nid, target, "coord_nearest"))

    # Final validation
    missing_final = [nid for nid in must_exist if nid not in dist]
    if missing_final:
        sample_keys = list(dist.keys())[:50]
        sample_node = nodes.get(missing_final[0])
        attrs = sorted([a for a in dir(sample_node) if not a.startswith("_")]) if sample_node else []
        raise KeyError(
            f"Missing IDs in matrix sheet '{sheet}': {missing_final[:10]}\n"
            f"Example dist keys (first 50): {sample_keys}\n"
            f"Sample missing node attrs (first 30): {attrs[:30]}\n"
            f"NOTE: We tried string + exact coord + nearest coord aliasing."
        )
    '''
    if created:
        print("Distance aliasing applied (up to 20):", created[:20])
    '''
    return Instance(
        name=name,
        nodes=nodes,
        depot_id=depot_id,
        request_ids=request_ids,
        charging_ids=charging_ids,
        params=Params(),
        dist=dist,
    )


# ----------------------------
# CSV row writer
# ----------------------------
def to_row(instance_label: str, method: str, sim):
    return {
        "instance": instance_label,
        "method": method,
        "feasible": int(sim.feasible),
        "total_dist_km": round(sim.total_dist_km, 4),
        "total_time_min": round(sim.total_time_min, 4),
        "travel_time_min": round(sim.travel_time_min, 4),
        "charge_time_min": round(sim.charge_time_min, 4),
        "charge_stops": sim.charge_stops,
        "late_count": sim.late_count,
        "load_violations": sim.load_violations,
        "battery_violations": sim.battery_violations,
        "unserved": sim.unserved,
    }


# ----------------------------
# Main
# ----------------------------
def main():
    BASE = os.path.dirname(os.path.abspath(__file__))
    txt_folder = os.path.join(BASE, "dataset", "ESOGU-EVRP-PDP-TW")
    excel_path = os.path.join(BASE, "dataset", "distance_matrix.xlsx")

    sizes = [100]
    tws = [2]
    types = ["C", "R", "RC"]

    bikes = 4
    ga_cfg = GAConfig(bikes=bikes, pop_size=60, generations=120)

    rows = []

    for n in sizes:
        for tw in tws:
            for typ in types:
                fname = f"ESOGU_{typ}{n}_TW{tw}.txt"
                txt_path = os.path.join(txt_folder, fname)
                if not os.path.exists(txt_path):
                    continue

                label = f"{typ}{n}_TW{tw}"
                inst = build_instance(txt_path, excel_path)

                demands = [(inst.nodes[r].D + inst.nodes[r].P) for r in inst.request_ids]
                #print("Demand min/max:", min(demands), max(demands))
                #print("Total demand:", sum(demands))
                #print("Vehicle capacity:", inst.params.load_cap_kg)

                # Baselines
                sim_rnd, _ = baseline_random_assignment(inst, bikes=bikes, seed=1)
                rows.append(to_row(label, "baseline_random_dynamic", sim_rnd))

                print_bike_summary(label, "baseline_random_dynamic", sim_rnd)

                sim_grd, _ = baseline_distance_greedy(inst, bikes=bikes)
                rows.append(to_row(label, "baseline_greedy_dynamic", sim_grd))

                # GA
                # ---------------- GA ----------------
                best_routes, _ = run_ga(inst, ga_cfg)

                trace_dir = os.path.join(BASE, "results", "traces")
                os.makedirs(trace_dir, exist_ok=True)

                # -------- Dynamic --------
                sim_dyn, trace_dyn = simulate_plan(
                    inst, best_routes,
                    charging_policy="dynamic",
                    fixed_target_soc=0.80,
                    return_trace=True,
                    initial_soc=0.60
                )
                rows.append(to_row(label, "GA_routes_dynamic_charge", sim_dyn))
                print_bike_summary(label, "GA_routes_dynamic_charge", sim_dyn)

                trace_csv_dyn = os.path.join(trace_dir, f"{label}_GA_routes_dynamic_charge_trace.csv")
                if trace_dyn:
                    fieldnames = list(trace_dyn[0].keys())
                    with open(trace_csv_dyn, "w", newline="", encoding="utf-8") as f:
                        w = csv.DictWriter(f, fieldnames=fieldnames)
                        w.writeheader()
                        w.writerows(trace_dyn)
                    print("Saved trace to:", trace_csv_dyn)


                # -------- Full --------
                sim_full, trace_full = simulate_plan(
                    inst, best_routes,
                    charging_policy="full",
                    fixed_target_soc=0.80,
                    return_trace=True,
                    initial_soc=0.60
                )
                rows.append(to_row(label, "GA_routes_full_charge", sim_full))
                print_bike_summary(label, "GA_routes_full_charge", sim_full)

                trace_csv_full = os.path.join(trace_dir, f"{label}_GA_routes_full_charge_trace.csv")
                if trace_full:
                    fieldnames = list(trace_full[0].keys())
                    with open(trace_csv_full, "w", newline="", encoding="utf-8") as f:
                        w = csv.DictWriter(f, fieldnames=fieldnames)
                        w.writeheader()
                        w.writerows(trace_full)
                    print("Saved trace to:", trace_csv_full)


                # -------- Fixed 80 --------
                sim_fixed, trace_fixed = simulate_plan(
                    inst, best_routes,
                    charging_policy="fixed",
                    fixed_target_soc=0.80,
                    return_trace=True,
                    initial_soc=0.60
                )
                rows.append(to_row(label, "GA_routes_fixed80_charge", sim_fixed))
                print_bike_summary(label, "GA_routes_fixed80_charge", sim_fixed)

                trace_csv_fixed = os.path.join(trace_dir, f"{label}_GA_routes_fixed80_charge_trace.csv")
                if trace_fixed:
                    fieldnames = list(trace_fixed[0].keys())
                    with open(trace_csv_fixed, "w", newline="", encoding="utf-8") as f:
                        w = csv.DictWriter(f, fieldnames=fieldnames)
                        w.writeheader()
                        w.writerows(trace_fixed)
                    print("Saved trace to:", trace_csv_fixed)

                print(f"Done: {label}")

    out_dir = os.path.join(BASE, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "esogu_results.csv")

    fieldnames = [
        "instance", "method", "feasible",
        "total_dist_km", "total_time_min", "travel_time_min", "charge_time_min",
        "charge_stops", "late_count", "load_violations", "battery_violations", "unserved"
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"\nSaved results to: {out_csv}")


if __name__ == "__main__":
    main()
