# parcs_v3/assoc_debugger_v3.py

def detect_assoc_anomalies_v3(seq):
    anomalies = []
    for f_idx, frame in enumerate(seq):
        if not isinstance(frame, list):
            anomalies.append((f_idx, "not_list"))
            continue

        for i, x in enumerate(frame):
            if isinstance(x, slice):
                anomalies.append((f_idx, f"slice_at_{i}"))
            elif not (isinstance(x, int) or x is None):
                anomalies.append((f_idx, f"non_int_at_{i}:{type(x)}"))
            elif isinstance(x, int) and x < 0:
                anomalies.append((f_idx, f"negative_at_{i}:{x}"))
    return anomalies


def print_assoc_anomalies_v3(anoms):
    if not anoms:
        print("No anomalies detected.")
        return
    print("=== Association Anomalies (v3) ===")
    for f_idx, msg in anoms:
        print(f"Frame {f_idx}: {msg}")


def debug_assoc_dict_sequence_v3(seq):
    print("=== Detailed Association Debug (v3) ===")
    for f_idx, frame in enumerate(seq):
        print(f"\n--- Frame {f_idx} ---")
        print(f"type={type(frame).__name__} len={len(frame)}")
        for i, x in enumerate(frame):
            print(f"  [{i}] -> {x} ({type(x).__name__})")
