# Credit experiment protocol

## Population and split

One historical credit-card cohort from Taiwan; outcome is next-month default. There are no independent observation vintages to support chronological validation. Original ID and demographic columns do not enter fitting.

A fingerprint of the 19 raw financial-history predictors forms a profile group. `StratifiedGroupKFold` assigns groups to five seeded folds. Folds 3–4 fit candidate models (about 40%), fold 2 fits probability calibrators (20%), fold 1 selects the model and review threshold (20%), and fold 0 is the final test set (20%). Identical profiles cannot span those partitions. Rows with conflicting outcomes within a profile are retained together. This is a leakage-reduction policy, not proof that every customer dependency is known.

## Features and selection

Features include credit limit, six payment-status values, six bill balances, six prior payments, utilisation ratios, count/maximum of positive delays, bill change and payment-to-bill ratio. Sentinel payment-status values remain as supplied; feature interpretation must follow the source dictionary. Credit limit must be positive; missing financial inputs raise an error. Engineered ratios are intentionally simple and are not accounting definitions of regulatory utilisation or debt-service burden.

Standardisation is fitted only inside the logistic training pipeline. Gradient boosting receives the financial features directly. Models do not use class weighting because the main target is probability quality. For each candidate, a near-unregularised logistic calibrator fits raw log-odds to labels on the separate calibration fold. Select the calibrated candidate with the lowest validation Brier score. Neither estimator nor calibrator is refitted after selection. A validation 80th-percentile probability sets a research review threshold; ties or distribution changes can alter test review share.

## Evaluation and diagnostics

Final test metrics: ROC AUC, average precision, Brier score, constant-prior Brier baseline, calibration by quantile, risk deciles, and recall at the frozen threshold. The ROC interval is a row-bootstrap interval; profile dependencies can make it optimistic. Reported PSI uses smoothed, training-quantile score bins; this is a distribution diagnostic without a universal significance threshold. Random-cohort PSI is not evidence of temporal stability.

Permutation importance uses only validation data and the underlying fitted candidate's ranking. It is not causal attribution. Logistic coefficients are standardised challenger coefficients, not customer adverse-action reasons. Sex code and age bands are used only for an aggregate held-out audit with groups of at least 200. Observed differences do not establish fairness or discrimination; other groups and intersectional analysis are absent.

## Score and scenarios

Custom score: `600 + 20 / ln(2) × ln(((1 − PD) / PD) / 50)`. A PD of 1/51 maps to 600; doubling good/bad odds adds 20. Higher scores indicate lower modelled risk. This mapping is a presentation convention, not an approved lender scorecard.

Scenario odds multipliers are 1, 1.25 and 1.5; assumed LGDs are 35%, 45% and 60%. Exposure proxy is max(latest bill, 0). One-month proxy loss is `sum(stressed PD × exposure proxy × assumed LGD)` in source NT-dollar units. It excludes undrawn credit conversion, recoveries, discounted cash flows, multi-period transition dynamics and macroeconomic calibration. It cannot be relabelled lifetime or IFRS 9 ECL.
