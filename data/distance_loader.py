import pandas as pd
import re


def normalize_id(x) -> str:
    """
    Normalize node IDs so TXT IDs match Excel headers.

    Examples:
      37a   -> 37A
      CS1   -> cs1
      58/2  -> 58_2        (you can change '_' to '-' if your Excel uses '-')
      61A/2 -> 61A_2
      61a/2 -> 61A_2
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""

    s = str(x).strip()

    # remove trailing .0 from numbers read as floats (e.g., 48.0 -> "48")
    s = re.sub(r"\.0$", "", s)

    # unify separators: / or - -> _
    s = s.replace("/", "_").replace("-", "_")

    # collapse multiple underscores
    s = re.sub(r"_+", "_", s)

    # charging station ids: keep lowercase 'cs'
    # (some Excels store CS1, Cs1 etc.)
    if s.lower().startswith("cs"):
        return "cs" + s[2:].strip()

    # everything else: uppercase (keeps 37A, 61A_2)
    return s.upper()


def load_esogu_distance_matrix(excel_path: str, sheet_name: str = "distance v3.2", to_km: bool = True):
    """
    Loads the ESOGU master distance matrix.
    Returns dictionary: dist[from_id][to_id] = distance

    IMPORTANT: IDs are normalized with normalize_id() to match TXT IDs.
    """
    raw = pd.read_excel(excel_path, sheet_name=sheet_name, header=None, engine="openpyxl")

    # Row 0 = header row (destination IDs)
    header_ids_raw = raw.iloc[0, 2:].tolist()
    header_ids = [normalize_id(x) for x in header_ids_raw]

    # Column 0 = source IDs
    row_ids_raw = raw.iloc[2:, 0].tolist()
    row_ids = [normalize_id(x) for x in row_ids_raw]

    dist = {}

    for i, from_id in enumerate(row_ids):
        if not from_id:
            continue

        dist[from_id] = {}

        for j, to_id in enumerate(header_ids):
            if not to_id:
                continue

            value = raw.iloc[i + 2, j + 2]
            if pd.isna(value):
                continue

            value = float(value)
            if to_km:
                value /= 1000.0  # meters → km

            dist[from_id][to_id] = value

    return dist
