# TRL-5 validation plan

TRL-5 target means integrated prototype validation in a relevant controlled
environment. It is **not** a current achievement claim. The plan below sequences
the activities required to reach and document TRL-5 evidence for the AI-assisted
weld-defect detection product.

## Activities

1. Map employer procedures and applicable standards / editions.
2. Obtain written data authorization, modality inventory, retention and
   provenance.
3. Create reference labels with qualified reviewers and documented adjudication.
4. Integrate ingestion, QC, inference, dashboard and audit trail.
5. Verify software, security, robustness and independent weld-group holdout.
6. Pilot with representative NDT export formats / workflow in a controlled
   non-operational environment.
7. Preserve protocol, dataset / model cards, test report, deviations, reviewer
   sign-off and pilot report.

## Gates

Modality / device / material / thickness coverage; no weld leakage between train
and test; class-wise metrics with confidence intervals; manual review of critical
false negatives; end-to-end human sign-off and audit; security and restore
tests; employer / NDT authority approval.

## Evidence status

Source requirements extracted; synthetic smoke data only; public dataset
candidates documented in `data/raw/sources.yaml`; no Mazandaran customer data
supplied; expert validation and relevant-environment pilot pending.
