# PARCS: Phased‑Array Radiance‑Channel Sensing

PARCS: Phased‑Array Radiance‑Channel Sensing
A Stand‑Alone Multi‑Version Association Operator for Contested‑Airspace Autonomy

Overview
PARCS is a stand‑alone, multi‑version object‑association framework designed for synthetic sensing, multi‑target tracking, and adversarial‑resilient inference in contested environments.

The PARCS operator family (v1, v2, v3) provides deterministic, temporal‑semantic, and adversarial‑resilient association pipelines suitable for:

- multi‑object trajectory reconstruction
- decoy discrimination
- perturbation stability analysis
- anomaly detection
- real‑time streaming
- scaling demonstrations (25 → 2000 frames)

PARCS is fully self‑contained and does not require the SU(8) radiance‑channel framework or GRMHDsu8 geometry to operate.

##PARCS Versions
**PARCS v1 — Deterministic Baseline Operator**
A minimal, deterministic association pipeline.

geometric nearest‑neighbor matching

no temporal consistency

no semantic discrimination

no perturbation modeling

suitable for baseline tests and simple synthetic scenes

This version establishes the foundation for the operator family.

**PARCS v2** — Temporal‑Semantic Operator
Adds temporal and semantic reasoning.

- geometric + temporal scoring
- semantic‑type discrimination (missile / drone / decoy)
- multi‑object support
- improved stability under mild noise
- suitable for moderate‑complexity synthetic or physical scenes

This version is appropriate for more realistic multi‑target scenarios.

**PARCS v3** — Adversarial‑Resilient Operator
A full adversarial‑resilient association engine.

- tiered geometric → temporal → semantic scoring
- perturbation‑aware identity stability analysis
- anomaly detection (spoofing, jitter, dropout, decoy swarms)
- adversarial stress‑test mode (enable_adversarial_mode = True)
- real‑time streaming mode
- scaling demonstration (25 → 2000 frames)
- semantic‑type stability breakdown
- identity‑stability variance diagnostics

This version is designed for contested‑airspace autonomy, denial‑and‑deception environments, and robustness studies.

Documentation
Documentation is available in the docs/ directory:

Operator Theory — deterministic, temporal‑semantic, adversarial‑resilient

Diagnostics — identity stability, anomaly detection, confusion matrices

Adversarial Mode — spoofing, decoy swarms, jitter, dropout

Streaming Mode — real‑time association

Scaling Analysis — runtime, entropy, stability variance

License
PARCS Phased‑Array Radiance‑Channel Sensing 
Copyright (c) 2026
HadronQ - Hadron Quantum Technologies Corp.
J Harlow, Principal, VP‑Engineering

Licensed under the Creative Commons Attribution–NonCommercial–NoDerivatives 4.0 International License (CC BY‑NC‑ND 4.0).

You are free to:

- Share — copy and redistribute the material in any medium or format

Under the following terms:

- Attribution — Provide appropriate credit and indicate changes.

- NonCommercial — No commercial use.

- NoDerivatives — No distribution of modified versions.

Full legal text:
https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode

Citation:  If you use PARCS in academic or technical work, please cite:

Code
Harlow, J. (2026).
PARCS: Phased‑Array Radiance‑Channel Sensing (v1–v3).
Hadron Quantum Technologies Corp.
