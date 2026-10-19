# parcs_v3/assoc_stabilizer_v3.py

def stabilize_assoc_dict_sequence_v3(gt_traj, assoc_seq):
    n_frames = len(gt_traj)
    n_gt = len(gt_traj[0]) if n_frames > 0 else 0

    cleaned = []

    for f_idx in range(n_frames):
        raw = assoc_seq[f_idx]
        frame = []

        for i in range(n_gt):
            if i < len(raw):
                x = raw[i]
                if isinstance(x, int) and x >= 0:
                    frame.append(x)
                else:
                    frame.append(None)
            else:
                frame.append(None)

        cleaned.append(frame)

    return cleaned


