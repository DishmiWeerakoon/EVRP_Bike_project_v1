import sys
import pandas as pd
import matplotlib.pyplot as plt

# Event colors (optional)
EVENT_COLOR = {
    "travel": "tab:blue",
    "service": "tab:green",
    "charge": "tab:red",
    "wait": "tab:orange",
}

def plot_static(trace_csv: str, title: str = ""):
    df = pd.read_csv(trace_csv)

    required = ["bike_id", "event", "from_id", "to_id", "lat", "lon"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}\nFound: {list(df.columns)}")

    df["event"] = df["event"].astype(str).str.lower()
    df["from_id"] = df["from_id"].astype(str)
    df["to_id"] = df["to_id"].astype(str)

    # Build node -> (lon, lat) map using to_id rows (works with your trace schema)
    node_xy = {}
    for _, r in df.iterrows():
        nid = r["to_id"]
        if pd.notna(r["lon"]) and pd.notna(r["lat"]):
            node_xy[nid] = (float(r["lon"]), float(r["lat"]))

    fig, ax = plt.subplots(figsize=(11, 8))

    # Plot all nodes faintly
    xs = [xy[0] for xy in node_xy.values()]
    ys = [xy[1] for xy in node_xy.values()]
    ax.scatter(xs, ys, s=10, alpha=0.25, label="Nodes")

    # Mark depot / stations if present in node_type column
    if "node_type" in df.columns:
        types = df[["to_id", "node_type", "lat", "lon"]].dropna()
        types["node_type"] = types["node_type"].astype(str).str.lower()

        dep = types[types["node_type"].eq("d")]
        cs = types[types["node_type"].eq("cs")]

        if not dep.empty:
            ax.scatter(dep["lon"], dep["lat"], s=120, marker="s", label="Depot")
        if not cs.empty:
            ax.scatter(cs["lon"], cs["lat"], s=60, marker="^", label="Charging stations")

    # Draw route segments by bike (color per bike)
    bikes = sorted(df["bike_id"].unique().tolist())
    cmap = plt.get_cmap("tab10")

    for i, b in enumerate(bikes):
        sub = df[df["bike_id"] == b]

        # Draw each leg line (from_id -> to_id) if both endpoints known
        for _, r in sub.iterrows():
            a = r["from_id"]
            c = r["to_id"]
            if a in node_xy and c in node_xy and a != c:
                (x1, y1) = node_xy[a]
                (x2, y2) = node_xy[c]
                ax.plot([x1, x2], [y1, y2], linewidth=1.6, alpha=0.85, color=cmap(i % 10))

        # Mark each bike's final node
        last = sub.iloc[-1]
        end_id = last["to_id"]
        if end_id in node_xy:
            ax.scatter([node_xy[end_id][0]], [node_xy[end_id][1]], s=70, color=cmap(i % 10))

    ax.set_title(title or f"Static Routes from Trace: {trace_csv}")
    ax.set_xlabel("Longitude (x)")
    ax.set_ylabel("Latitude (y)")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()


def main():
    """
    Usage:
      python experiments/plot_static_from_trace.py results/traces/C100_TW2_GA_routes_dynamic_charge_trace.csv
    """
    if len(sys.argv) < 2:
        print(main.__doc__)
        sys.exit(1)

    trace_csv = sys.argv[1]
    title = sys.argv[2] if len(sys.argv) >= 3 else ""
    plot_static(trace_csv, title=title)

if __name__ == "__main__":
    main()
