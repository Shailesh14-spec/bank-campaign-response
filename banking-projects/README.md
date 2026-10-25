# Banking Python Portfolio Projects

Two completed, independently runnable public-data research projects prepared for standalone repositories.

| Project | Main contribution | Measured test result |
| --- | --- | --- |
| [Credit risk modelling and validation](credit-risk-lab) | Disjoint profile groups, probability calibration, score scaling and loss sensitivity | ROC AUC 0.780; Brier 0.1345 |
| [Card fraud monitoring and analyst triage](fraud-monitoring-lab) | Chronological validation, separate calibration and strict hourly analyst budgets | Average precision 0.782; 78/114 frauds flagged at 120 reviews |

Both include executable code, data provenance, model cards, model comparisons, measured findings, figures, tests and CI workflow definitions. All 13 tests and both unlabelled batch-scoring smoke tests passed locally. Raw data, row-level predictions and fitted binary models are not committed.

Prepared standalone repository names: `bank-credit-risk-python` and `bank-fraud-monitoring-python`.

This branch preserves the completed source while those repositories are created. Workflows under each project's own `.github/` directory will run after that directory is published as a repository root; nested workflow definitions do not execute from this preparation branch.
