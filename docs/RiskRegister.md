# Risk register

This register tracks the principal risks for the AI-assisted weld-defect detection
product under development for Sate Mazandaran Gas. It is maintained alongside the
SRS and BRS and reviewed at each milestone gate. Each risk carries an owner, a
severity and a mitigation that is verifiable through process assets under
`docs/CMMI_ML5/`.

| Risk | Severity | Owner | Mitigation |
|---|---|---|---|
| No authorized customer data available | High | Product owner | Data agreement and intake gate before any model training on operational data |
| Public dataset license conflict | High | Legal / ML lead | Research-only restrictions; seek commercial permission; record license per source in `data/raw/sources.yaml` |
| Label disagreement between reviewers | Medium | NDT Level III | Double review and Level III adjudication; track inter-rater agreement |
| Rare critical defects underrepresented | High | ML lead | Class-wise metrics, stratified sampling plan and mandatory manual review |
| Train / test leakage across welds | High | ML lead | Split by weld and inspection record; group-aware cross-validation |
| Device / procedure domain shift | Medium | ML lead | Stratified external validation and drift monitoring in MLOps pipeline |
| AI over-reliance by operators | High | Product owner | Advisory-only UI, human sign-off and explicit limitations in dashboard |
| Model drift in production | Medium | MLOps | MLflow tracking, drift detection DAG and retraining trigger policy |
| Security exposure of API / model artifacts | Medium | DevOps | Auth, rate limiting, container scanning in CI, signed images |
| Loss of traceability / audit evidence | Medium | QA | CMMI ML5 process assets, immutable audit trail, model and dataset cards |
