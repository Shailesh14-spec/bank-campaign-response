# Credit risk model card

**Purpose:** demonstrate probability estimation, calibration, validation and sensitivity reporting on public historical data. No automated customer eligibility or credit decision is implemented.

**Inputs:** CSV columns `LIMIT_BAL`, `PAY_0`, `PAY_2`…`PAY_6`, `BILL_AMT1`…`BILL_AMT6`, and `PAY_AMT1`…`PAY_AMT6`. All must be numeric and nonmissing; credit limit must be positive. Extra columns are not used. Target, ID, sex, age, education and marriage are excluded from modelling. No data from after the label horizon is intentionally engineered.

**Outputs:** calibrated next-month-default research probability and a custom score. Local model bundle also preserves a validation-selected review threshold. These outputs have no validated lending-policy interpretation.

**Selection:** logistic regression versus histogram gradient boosting, calibrated on a separate profile-group fold, selected on validation Brier score. Actual selected model and performance are in `reports/metrics.json`; no test-set selection is performed.

**Validation:** historical group holdout, not out-of-time or UK validation. Calibration, risk bands, subgroup diagnostics and illustrative bill-exposure scenarios are included. The demographic audit is limited to fields available in the source and does not establish a fairness guarantee. No real bank system, API, access-control boundary, regulatory approval or live monitoring is deployed.

**Before any future use:** obtain an appropriate contemporary data population and label definition, establish governance and policy objectives, evaluate genuine out-of-time and subgroup behaviour, and validate exposure/recovery assumptions. The portfolio results alone do not justify real lending decisions.
