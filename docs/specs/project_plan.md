# GraphMER for Software Engineering — Project Plan

This plan proposes a spec-driven approach to adapt GraphMER from the biomedical domain to software engineering, leveraging a neurosymbolic encoder (~80M params) with Leafy Chain Graph Encoding, HGAT, and dual MLM+MNM objectives.

## Principles
- Spec-first: explicit, versioned specs for problem, ontology, data, and evaluation.
- Reproducible: pinned environments, data/model versioning, deterministic pipelines.
- Ontology-aligned: constraints enforced in data building, encoding, training, and evaluation.
- Incremental: start with 1–2 languages and a narrow task set; expand with confidence.

## Scope (v0 → v1)
- Languages: Python and Java (v0), expansion to JS/TS or Go (v1).
- Target tasks:
  - Intrinsic: link prediction (calls, inherits_from, imports), type/namespace disambiguation.
  - Extrinsic: code search reranking (MRR@10), call-graph completion on held-out repos, dependency inference from partial manifests.
- Constraints: encoder-only inference, latency budget suitable for IDE or server use.

## Milestones & Deliverables

### M0: Repo Bootstrap and Specs (Weeks 1–3)
Deliverables:
- docs/specs/
  - problem_spec.md (scope, stakeholders, constraints, acceptance criteria)
  - ontology_spec.yaml (entities, relations, constraints, versions)
  - data_spec.yaml (triple schema, provenance, licensing)
  - eval_spec.yaml (datasets, metrics, test cases, thresholds)
  - model_card.md (template)
- Engineering:
  - repo scaffold (src/, configs/, scripts/, tests/)
  - CI with lint/tests + schema gates for ontology/data
  - Dockerfile + pyproject.toml; MLflow/W&B stub; optional DVC

### M1: Corpus & Seed KG (Weeks 3–7)
Deliverables:
- Parsing adapters (tree-sitter, LibCST/astroid for Python; JavaParser for Java)
- KG builder producing curated ~20–50k triples with provenance
- Ontology validator (type/domain/range checks, acyclicity where required)
- Data quality report: coverage, error buckets, language mix
- Validation checkpoint: ≥ 98% type-correct edges; ≤ 0.5% forbidden cycles

### M2: Tokenizer & Encoding (Weeks 6–8)
Deliverables:
- Code-aware tokenizer (BPE/Unigram) with identifier rules and reserved relation/graph tokens
- Leafy Chain Graph sequence packer with sampling strategies (k leaves per anchor, positives/negatives)
- Tests: tokenization invariants, packer correctness, span alignment
- Validation checkpoint: identifier split ≥ 95%; span alignment ≥ 99%

### M3: Model & Training (Weeks 8–12)
Deliverables:
- Encoder-only Transformer (~80M) and HGAT module (relation-aware attention)
- Objectives: MLM (code/doc), MNM (leaf tails), negatives, ontology regularizers
- Training loop with mixed precision, checkpointing, logging (MLflow/W&B)
- Smoke run on small subset; learning curves; ablations for HGAT/negatives
- Mid-checkpoint: MLM token acc ≥ 55%; MNM tail token acc ≥ 40%; constraint-aware negatives +3 pts vs. naive

### M4: Evaluation & Baselines (Weeks 10–14)
Deliverables:
- Intrinsic eval: link prediction (MRR/Hits@k), disambiguation accuracy, ontology-constraint satisfaction
- Extrinsic eval: code search reranking, call-graph completion, dependency inference
- Baselines: CodeBERT/GraphCodeBERT/UniXcoder; node2vec/DeepWalk/TransE/RotatE; static analysis baselines
- Reports: tables/plots; win/loss analyses, error buckets

### M5: Packaging & Release (Weeks 14–16)
Deliverables:
- Model registry entry + model card with datasets and metrics
- Inference API (FastAPI) with endpoints for tail completion and reranking
- Monitoring plan: drift, latency, cost, ontology-constraint violations
- Roadmap for v1 (new languages, larger corpora, stronger negatives)

## Ontology Examples (Appendix)
- Entities: Repository, Module/Package, File, Class, Interface, Function/Method, Variable, Type, API, Library, Test, BuildTarget.
- Relations (directional, typed with domain→range):
  - defines: File→(Class|Function|Variable)
  - declares: (Class|Interface)→(Method|Attribute)
  - implements: Class→Interface
  - inherits_from: Class→Class (acyclic)
  - overrides: Method→Method (same signature, parent class chain)
  - calls: (Function|Method)→(Function|Method)
  - imports: Module→Module
  - depends_on: Module→Library | Project→Library
  - uses: (Function|Method|Class)→API
  - reads_from / writes_to: (Function|Method)→Variable
  - raises / catches: (Function|Method)→ExceptionType
  - tested_by: (Function|Class|Module)→Test

Examples (Python)
```python
class B: pass
class A(B):
    def foo(self):
        bar()

def bar():
    pass
```
Triples:
- A inherits_from B
- A.foo overrides? (no, unless B.foo exists)
- A.foo calls bar
- module imports <other_module> (if present)

Examples (Java)
```java
class B {}
class A extends B {
  void foo() { Utils.bar(); }
}
class Utils { static void bar() {} }
```
Triples:
- A inherits_from B
- A#foo calls Utils#bar
- File defines A, defines Utils
- Package imports <other.package> (if present)

## Architecture & Components
- src/ontology/: schemas (pydantic/jsonschema), validators, constraint rules
- src/parsing/: language adapters, AST/CFG extractors, build metadata parsers
- src/kg/: triple builders, dedup, provenance, export to parquet/graph DB
- src/encoding/: leafy_chain_packer.py, samplers.py, special token registry
- src/models/: encoder.py (RoBERTa-like), hgat.py, heads.py (MLM/MNM)
- src/training/: objectives.py, losses.py (constraint regularizers), loop.py
- src/evaluation/: intrinsic.py, downstream/ (search, call-graph, deps)
- configs/: tokenizer.yaml, encode.yaml, model.yaml, train.yaml, eval.yaml
- scripts/: build_tokenizer.py, build_kg.py, pack_sequences.py, train.py, evaluate.py

## Data & Governance
- Licensing: include only MIT, Apache-2.0, BSD-2/3, MPL-2.0 by default; exclude GPL/AGPL/LGPL unless explicitly approved; CI license scanning (e.g., scancode-toolkit); maintain allowlist and per-repo license snapshot.
- Provenance: repo URL, commit, file path, language, license, spans; reproducible snapshots and dataset manifests.
- Versioning: semantic versions for ontology, data, model; changelogs and migration notes for ontology updates.
- Privacy: PII scanning (even in code) and secret detection; redaction policy with audit logs.

## Risks & Mitigations
- Ontology drift → lock versions, CI schema gates, migration scripts
- KG noise → static analysis + heuristics; human-reviewed seed; active error mining
- Tokenization issues → identifier-aware rules; vocab audits; regression tests
- Shortcut learning → type-consistent/hard negatives; ontology regularizers; ablations
- Licensing → license filters; provenance tracking; model card disclosures

## Success Criteria
- Intrinsic: MRR ≥ 0.52, Hits@10 ≥ 0.78 on link prediction (Python/Java held-out); ontology-constraint violations ≤ 1%.
- Disambiguation: namespace/type disambiguation top-1 accuracy ≥ 92%.
- Extrinsic: code search reranking MRR@10 ≥ 0.44 and ≥ +10% over CodeBERT baseline; call-graph completion F1 ≥ 0.63 and +8–12% over static-analysis baseline; dependency inference F1 ≥ 0.70 and +10% over heuristics.
- Stability: reproducible training; inference P50 ≤ 25 ms, P95 ≤ 60 ms per 512 tokens; constraint violations < 1%.

## Timeline (indicative)
- Weeks 1–3: M0
- Weeks 3–7: M1
- Weeks 6–8: M2
- Weeks 8–12: M3
- Weeks 10–14: M4
- Weeks 14–16: M5

## Resource Planning
- Team: 5–7 contributors — 1 Ontology/KG, 2 Parsing/Static Analysis (Python, Java), 1–2 ML (modeling/training), 1 MLOps (infra/CI/serving), 0.5 Tech PM.
- Compute (small-model-first):
  - Primary: CPU training (configs/train_cpu.yaml) — fp32, activation checkpointing, seq_len 384→512, GA=64
  - Secondary: TPU (configs/train_tpu.yaml) — bf16, per-core micro-batch 2, seq_len 512→768
  - Optional: GPU only for ablations; not required for core training
  - Storage 2–4 TB object store; MLflow/W&B for tracking; DVC/LakeFS for data.

## Next Steps
1. Confirm initial languages and priority downstream tasks.
2. Approve ontology entity/relation list and constraints for v0.
3. Greenlight scaffold creation and CI setup.
4. Select baseline models and evaluation datasets.
