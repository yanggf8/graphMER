# Model Card (Template)

Model: GraphMER-SE (v0.1)

Intended Use
- Ontology-aligned code understanding for software engineering tasks.

Training Data
- OSS code (permissive), API docs; provenance recorded.

Model Architecture
- Encoder-only Transformer (~80M), HGAT relation-aware attention, MLM+MNM objectives.
- **Validated: Attention bias provides 50.11% MNM improvement (2025-10-20)**

Metrics (Targets)
- See eval_spec.yaml thresholds (MRR, Hits@k, accuracy, F1, latency).

Ethical/Legal
- License allowlist enforcement; PII/secret scanning; compliance notes.

Limitations
- Python/Java only (v0), model relies on ontology and KG coverage.
