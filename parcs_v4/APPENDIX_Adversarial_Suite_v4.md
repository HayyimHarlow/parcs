# Appendix C — PARCS v4 Adversarial Perturbation Suite

**The PARCS v4 adversarial suite** applies all perturbations in a fixed, reproducible order.
This ensures that semantic, geometric, and temporal robustness are evaluated under the full adversarial envelope.

| Perturbation              | Description                                     | Purpose                        | Notes |
| ------------------------- | ----------------------------------------------- | ------------------------------ | ----------------------------------- |
| **Centroid Jumps**        | Random spatial displacement of object centroids | Tests geometric stability      | Inherited from v3                   |
| **Radius Corruption**     | Perturbs object radii / extents                 | Tests shape‑based invariants   | Inherited from v3                   |
| **Semantic Flips**        | Randomly changes semantic type                  | Tests semantic resilience      | Inherited from v3                   |
| **Decoy Swarms**          | Injects coordinated decoy clusters              | Tests multi‑target association | v4 requires full swarm mode         |
| **Temporal Jitter**       | Perturbs timestamps and frame continuity        | Tests temporal consistency     | v4 uses expanded jitter envelope    |
| **Sensor Dropout**        | Removes detections for random frames            | Tests identity continuity      | Critical for v4 identity stability  |
| **False‑Positive Bursts** | Injects spurious detections                     | Tests robustness to clutter    | Required for DARPA‑grade evaluation |

