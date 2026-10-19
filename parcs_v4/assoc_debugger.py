# parcs_v3/assoc_debugger.py

import numpy as np

def _centroid(obj):
    try:
        c = obj.get("centroid", None)
        if c is None:
            return None
        return float(c[0]), float(c[1])
    except Exception:
        return None

def _is_slice_like(x):
    return isinstance(x, slice)

def _is_non_int(x):
    return not isinstance(x, int)

def _debug_frame(idx, assoc_dict):
    out = []
    if not isinstance(assoc_dict, dict):
        out.append(f"Frame {idx}: NOT A DICT")
        return out

    # Check keys
    keys = list(assoc_dict.keys())
    for k in keys:
        if isinstance(k, slice):
            out.append(f"Frame {idx}: slice-like key {k}")
        if isinstance(k, str):
            # fine
            pass
        else:
            if not isinstance(k, str):
                out.append(f"Frame {idx}: non-string key {k}")

    # Check matches
    for m in assoc_dict.get("matches", []):
        if not isinstance(m, dict):
            out.append(f"Frame {idx}: malformed match entry {m}")
            continue

        gt = m.get("gt", {})
        det = m.get("det", {})

        if _centroid(gt) is None:
            out.append(f"Frame {idx}: GT missing centroid {gt}")
        if _centroid(det) is None:
            out.append(f"Frame {idx}: DET missing centroid {det}")

    return out


def detect_assoc_anomalies(assoc_seq):
    anomalies = []
    for idx, frame in enumerate(assoc_seq):
        anomalies.extend(_debug_frame(idx, frame))
    return anomalies


def print_assoc_anomalies(anoms):
    if not anoms:
        print("No anomalies detected.")
        return
    print("=== Association Anomalies ===")
    for a in anoms:
        print(a)


def debug_assoc_dict_sequence(assoc_seq):
    print("=== Detailed Association Debug ===")
    for idx, frame in enumerate(assoc_seq):
        print(f"\n--- Frame {idx} ---")
        if not isinstance(frame, dict):
            print("NOT A DICT")
            continue

        print("Keys:", list(frame.keys()))
        for k, v in frame.items():
            print(f"  {k}: type={type(v)} len={len(v) if hasattr(v,'__len__') else 'n/a'}")





