# GraphMER-SE Architecture Documentation

## Overview

GraphMER-SE is a neurosymbolic encoder that combines code tokens with knowledge graph triples using the Leafy Chain Graph Encoding algorithm from the GraphMER paper.

## Core Components

### 1. Leafy Chain Graph Encoding (`src/encoding/leafy_chain.py`)
- **Purpose**: Linearizes knowledge graph triples while preserving graph structure
- **Algorithm**: DFS-based chain extraction with special tokens ([ENT], [REL], [CHAIN])
- **Integration**: Used in dataset builder to convert KG triples to token sequences

### 2. Graph Positional Encoding (`src/models/graph_positional.py`)
- **Purpose**: Preserves graph structure in transformer attention
- **Components**: Sequence positions, chain positions, depth encoding, role encoding
- **Integration**: Replaces standard positional encoding in TinyEncoder

### 3. Multi-hop Attention (`src/models/multihop_attention.py`)
- **Purpose**: Enables reasoning over multi-hop graph paths
- **Features**: Path-aware attention biases, configurable hop lengths (1-3)
- **Integration**: Optional replacement for standard attention layers

### 4. Relation-Aware Attention (`src/models/encoder.py`)
- **Purpose**: Adds relation-specific biases to attention computation
- **Implementation**: Separate bias embeddings for same-relation and cross-relation attention
- **Integration**: Core component of TinyEncoder

## Training Pipeline

### Dataset Builder (`src/training/dataset_v2.py`)
1. **Input**: Code snippets + KG triples
2. **Processing**: 
   - Leafy Chain encoding for KG triples
   - BPE tokenization for code
   - MLM/MNM masking
3. **Output**: Unified sequences with relation IDs

### Training Script (`scripts/train_v2.py`)
1. **Model**: TinyEncoder with all GraphMER features
2. **Objectives**: Joint MLM + MNM training
3. **Features**: Constraint regularizers, curriculum learning, negative sampling
4. **Checkpointing**: Automatic cleanup (keeps latest 2)

## Model Architecture

```
TinyEncoder (85M parameters)
├── Token Embedding (vocab_size=8000, d_model=768)
├── Graph Positional Encoding (multi-component)
├── Relation Embedding (num_relations=13)
├── 12x Encoder Layers
│   ├── Multi-head Attention (with relation biases)
│   ├── Feed-forward Network
│   └── Layer Normalization
└── Output Layer Normalization
```

## Configuration

### Standard Training (`configs/train_cpu.yaml`)
```yaml
model:
  d_model: 768
  n_heads: 12
  n_layers: 12
  use_rel_attention_bias: true
  use_multihop: false

training_data:
  max_seq_len: 512
  mlm_prob: 0.15
  mnm_prob: 0.2
```

### Multi-hop Training
```yaml
model:
  use_multihop: true
  max_hops: 3
```

## Data Flow

1. **Input**: Multi-language code (Python, Java, JavaScript) + KG triples
2. **Leafy Chain**: Linearize graph structure
3. **Tokenization**: BPE encoding
4. **Masking**: MLM + MNM objectives
5. **Encoding**: Graph-aware positional + relation embeddings
6. **Attention**: Relation-aware or multi-hop attention
7. **Output**: Contextualized representations

## Key Features

### GraphMER Paper Compliance
- ✅ Neurosymbolic architecture
- ✅ Leafy Chain graph encoding
- ✅ Relation-aware attention
- ✅ Graph positional encoding
- ✅ Multi-hop reasoning
- ✅ Joint MLM/MNM training

### Advanced Features (Beyond Paper)
- ✅ Multi-language support (Python, Java, JavaScript)
- ✅ Constraint regularizers
- ✅ Curriculum learning
- ✅ Negative sampling
- ✅ Production infrastructure

## Performance

### Training Results (1,000 steps)
- **Loss Reduction**: 57% (16.4 → 6.999)
- **MLM Accuracy**: 33% validation
- **Training Time**: ~1.5 hours (CPU)
- **Memory**: <8GB RAM

### Model Scale
- **Parameters**: 85M (configurable)
- **Vocabulary**: 8,000 BPE tokens
- **Relations**: 13 types
- **Sequence Length**: Up to 512 tokens
- **Knowledge Graph**: 29,274 triples (3 languages)
