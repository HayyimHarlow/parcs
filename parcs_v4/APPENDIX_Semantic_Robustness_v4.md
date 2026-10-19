# Appendix A — PARCS v4 Semantic‑Robustness Evaluation
##A.1 Overview
PARCS v4 introduces a fully corrected semantic‑robustness evaluation pipeline.
The purpose of this appendix is to document the methodology, metrics, and results used to quantify the ability of the PARCS v4 operator to maintain correct semantic classification under adversarial perturbation.

Semantic robustness is evaluated using the TypeConfusionEvaluator, which compares:

- ground‑truth semantic type
- inferred semantic type

for every object in every frame of the adversarially perturbed scene.

The semantic taxonomy is fixed to the PARCS v4 class set:

| Class | Meaning                       |
| ----- | ----------------------------- |
| 0     | missile                       |
| 1     | drone                         |
| 2     | decoy                         |
| 3     | none (dropout / no detection) |

This taxonomy is used consistently across the operator, perturbation suite, and evaluation pipeline.

##A.2 Methodology
The semantic‑robustness evaluation follows these steps:

1. Generate a synthetic forward scene using the v4 forward model.

2. Apply the full adversarial suite (centroid jumps, radius corruption, semantic flips, decoy swarms, temporal jitter, sensor dropout, false‑positive bursts).

3. Run the PARCS v4 operator using TieredMatchPolicy as the sole association mechanism.

4. Compute the confusion matrix using corrected indexing:

det_traj[𝑓𝑟𝑎𝑚𝑒_𝑖𝑑𝑥][𝑑𝑒𝑡_𝑖𝑑𝑥]

5. Report semantic robustness using the DARPA‑formatted confusion matrix.

This process ensures that semantic robustness is measured after all adversarial perturbations and without any GT‑position matching.

##A.3 Confusion Matrix Definition
The confusion matrix 

𝐶

 is defined as:

   𝐶_{𝑖𝑗} = number of frames where GT class 𝑖 was inferred as class 𝑗

- Rows correspond to ground‑truth classes.
- Columns correspond to inferred classes.

This matrix provides a direct measure of semantic stability under adversarial conditions.

##A.4 Interpretation
High diagonal values indicate strong semantic robustness.
Off‑diagonal values quantify:

- misclassification under perturbation,
- semantic drift,
- confusion between missile/drone/decoy classes,
- dropout behavior.

This appendix provides the formal basis for reporting semantic‑class performance in DARPA milestone documents.

End of Appendix A