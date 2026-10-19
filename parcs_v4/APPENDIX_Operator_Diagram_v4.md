                 ┌──────────────────────────┐
                 │  ForwardSceneGenerator   │
                 └──────────────┬───────────┘
                                │
                                ▼
                 ┌──────────────────────────┐
                 │ FullAdversarialGenerator │
                 │  (frozen perturbation)   │
                 └──────────────┬───────────┘
                                │
                                ▼
                 ┌──────────────────────────┐
                 │      PARCSv4Operator     │
                 │  TieredMatchPolicy only  │
                 └──────────────┬───────────┘
                                │
                                ▼
                 ┌──────────────────────────┐
                 │       ResultsClass       │
                 └──────────────┬───────────┘
                                │
                                ▼
        ┌──────────────────────────────────────────────────────┐
        │                    Metrics_v4                        │
        │  - identity stability                                │
        │  - TP / FP / FN                                      │
        │  - localization error                                │
        │  - semantic confusion matrix                         │
        └──────────────────────────────────────────────────────┘
