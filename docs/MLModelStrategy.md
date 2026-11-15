# ML model strategy

This document defines the modeling strategy for AI-assisted weld-defect
detection. It separates the reproducible baseline used for smoke testing from
candidate production models that require authorized data and expert labels
before any deployment claim. No model here is a validated industrial detector.

## Baseline

HOG image descriptors combined with a class-balanced RandomForest classifier
provide a reproducible end-to-end smoke test on synthetic data. It does not
localize defects and is used only to validate the data pipeline, training loop
and evaluation harness.

## Candidate production models

- **Object detection:** YOLO family for expert-reviewed bounding boxes when
  sufficient labeled data is available.
- **Segmentation:** U-Net / Mask R-CNN when pixel masks and modality support
  justify the annotation cost.
- **Tabular metadata:** calibrated gradient boosting as a supplementary signal
  only; it must never substitute for NDT interpretation by a qualified reviewer.

## Evaluation

Group-aware split by weld / inspection record; class-wise precision, recall, F1,
PR curves, confusion matrix, calibration, latency and critical false-negative
review. Report confidence intervals and domain coverage. Numeric acceptance
gates are approved with the employer / NDT authority and recorded in the SRS.
