# Problem Spec

Title: GraphMER for Software Engineering (Personal Project)

Goals
- Build an ~80M parameter encoder-only neurosymbolic model for code understanding aligned to a formal ontology.
- Support ontology-respecting tasks: link prediction (calls/inherits/imports), disambiguation, code search reranking, call-graph completion, dependency inference.

Users / Stakeholders
- You (personal project). Future consumers: developers, tooling, research.

Scope (v0)
- Languages: Python, Java.
- Data: permissively licensed OSS repos, API docs.
- Compute: CPU-first; TPU optional; GPU not required.

Constraints
- Encoder-only; latency suitable for offline/batch and eventual IDE usage.
- Licensing allowlist only.

Acceptance Criteria (quantitative)
- Link prediction: MRR ≥ 0.52; Hits@10 ≥ 0.78; constraint violations ≤ 1%.
- Disambiguation: top-1 ≥ 92%.
- Code search reranking: MRR@10 ≥ 0.44 and ≥ +10% over CodeBERT baseline on same split.
- Call-graph completion: F1 ≥ 0.63 (+8–12% over static-analysis baseline).
- Dependency inference: F1 ≥ 0.70 (+10% over heuristics).

Non-goals (v0)
- Generative code synthesis.
- Proprietary data ingestion.
