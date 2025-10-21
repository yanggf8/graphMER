#!/bin/bash
# End-to-end automation: Package, validate, and upload to Google Drive for Colab TPU training

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                                                              ║${NC}"
echo -e "${BLUE}║     GraphMER-SE: Automated Colab TPU Deployment Pipeline    ║${NC}"
echo -e "${BLUE}║                                                              ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Step 1: Validate Knowledge Graph
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Step 1: Validate Knowledge Graph${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

if [ -f "data/kg/enhanced_multilang.jsonl" ]; then
    TRIPLE_COUNT=$(wc -l < data/kg/enhanced_multilang.jsonl)
    echo -e "${GREEN}✅ Knowledge graph found: $TRIPLE_COUNT triples${NC}"
    
    if [ "$TRIPLE_COUNT" -lt 30000 ]; then
        echo -e "${RED}❌ Insufficient triples (need ≥30,000)${NC}"
        exit 1
    fi
    
    echo -e "${BLUE}Running quality validation...${NC}"
    python src/ontology/kg_validator.py \
        data/kg/enhanced_multilang.jsonl \
        data/kg/enhanced_multilang.entities.jsonl \
        docs/specs/ontology_spec.yaml | tee /tmp/kg_validation.txt
    
    if grep -q "inherits_acyclic.*True" /tmp/kg_validation.txt; then
        echo -e "${GREEN}✅ Quality validation passed${NC}"
    else
        echo -e "${RED}❌ Quality validation failed${NC}"
        exit 1
    fi
else
    echo -e "${RED}❌ Knowledge graph not found. Build it first:${NC}"
    echo "  python scripts/build_kg_enhanced.py --source_dir data/raw"
    exit 1
fi

echo ""

# Step 2: Create Deployment Package
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Step 2: Create Deployment Package${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Clean old package
rm -rf colab_deploy/ graphmer_colab.tar.gz 2>/dev/null || true

# Create deployment directory
mkdir -p colab_deploy
echo -e "${BLUE}Creating deployment package...${NC}"

# Copy essential files
cp -r src/ colab_deploy/
cp -r data/kg/ colab_deploy/data/
cp -r configs/ colab_deploy/
cp -r scripts/ colab_deploy/
cp pyproject.toml colab_deploy/ 2>/dev/null || echo "pyproject.toml not found, skipping"

# Copy documentation
mkdir -p colab_deploy/docs/specs
cp docs/specs/ontology_spec.yaml colab_deploy/docs/specs/ 2>/dev/null || echo "ontology_spec.yaml not found"
cp COLAB_TPU_SETUP.md colab_deploy/ 2>/dev/null || echo "COLAB_TPU_SETUP.md not found"

# Create Colab-optimized config
cat > colab_deploy/configs/train_colab.yaml << 'YAML_EOF'
# Training config for Google Colab TPU v2-8
run:
  seed: 1337
  epochs: 5
  gradient_accumulation_steps: 16
  log_interval: 50
  eval_interval_steps: 500
  save_interval_steps: 500  # Frequent saves for 12h session limit
  mixed_precision: bf16
  deterministic: true

hardware:
  device: tpu
  tpu_cores: 8
  num_workers: 2  # Reduced for Colab CPU limits

model:
  hidden_size: 768
  num_layers: 12
  num_heads: 12
  intermediate_size: 3072
  dropout: 0.1
  positional_encoding: alibi
  norm: rmsnorm
  activation: swiglu
  hgat:
    enabled: true
    relation_bias: true
  use_rel_attention_bias: true

optimizer:
  name: adamw
  lr: 3.0e-4
  weight_decay: 0.01
  betas: [0.9, 0.98]
  eps: 1e-8
  scheduler:
    name: cosine
    warmup_steps: 1500

training_data:
  max_seq_len: 768
  micro_batch_size: 2
  pack_sequences: true
  short_to_long_curriculum:
    enabled: true
    schedule:
      - {steps: 0, max_seq_len: 512}
      - {steps: 15000, max_seq_len: 768}

objectives:
  mlm:
    mask_prob: 0.15
    span_mask_identifiers: true
  mnm:
    mask_prob: 0.20
    type_consistent_negatives: 2
    hard_negatives: 2

encoding:
  leaves_per_anchor:
    positive: 2
    negatives: 2
  max_leaves_per_sequence: 10

regularizers:
  ontology_constraints:
    antisymmetry_weight: 0.2
    acyclicity_weight: 0.2
  contrastive:
    enabled: true
    temperature: 0.07

checkpointing:
  activation_checkpointing: true

paths:
  kg_data: "/content/colab_deploy/data/enhanced_multilang.jsonl"
  kg_entities: "/content/colab_deploy/data/enhanced_multilang.entities.jsonl"
  output_dir: "/content/drive/MyDrive/GraphMER/outputs"
  checkpoint_dir: "/content/drive/MyDrive/GraphMER/checkpoints"
YAML_EOF

# Create tarball
echo -e "${BLUE}Creating tarball...${NC}"
tar -czf graphmer_colab.tar.gz colab_deploy/

PACKAGE_SIZE=$(du -h graphmer_colab.tar.gz | cut -f1)
echo -e "${GREEN}✅ Package created: graphmer_colab.tar.gz ($PACKAGE_SIZE)${NC}"
echo ""

# Step 3: Upload to Google Drive
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Step 3: Upload to Google Drive${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

if command -v rclone &> /dev/null; then
    if rclone listremotes | grep -q "^gdrive:"; then
        echo -e "${BLUE}Uploading to Google Drive...${NC}"
        ./scripts/upload_to_drive.sh
    else
        echo -e "${YELLOW}⚠️  rclone not configured${NC}"
        echo ""
        echo "To upload automatically, configure rclone:"
        echo "  1. Run: rclone config"
        echo "  2. Add 'gdrive' remote for Google Drive"
        echo "  3. Then run: ./scripts/upload_to_drive.sh"
        echo ""
        echo "Or upload manually:"
        echo "  1. Go to: https://drive.google.com"
        echo "  2. Create folder: GraphMER"
        echo "  3. Upload: graphmer_colab.tar.gz"
    fi
else
    echo -e "${YELLOW}⚠️  rclone not installed${NC}"
    echo ""
    echo "For automated upload, install rclone:"
    echo "  ./scripts/setup_rclone.sh"
    echo ""
    echo "Or upload manually:"
    echo "  1. Go to: https://drive.google.com"
    echo "  2. Create folder: GraphMER"
    echo "  3. Upload: graphmer_colab.tar.gz"
fi

echo ""

# Step 4: Generate Colab Notebook Instructions
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Step 4: Next Steps${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

echo -e "${GREEN}✅ Deployment package ready!${NC}"
echo ""
echo "📦 Package: graphmer_colab.tar.gz ($PACKAGE_SIZE)"
echo "📊 Triples: $TRIPLE_COUNT"
echo ""
echo "Next steps:"
echo "  1. Go to: https://colab.research.google.com"
echo "  2. Create new notebook"
echo "  3. Runtime → Change runtime type → TPU → Save"
echo "  4. Follow: UPLOAD_INSTRUCTIONS.md"
echo ""
echo "Quick start cells (copy to Colab):"
echo ""
echo "# Cell 1: Mount Drive and extract"
echo "from google.colab import drive"
echo "drive.mount('/content/drive')"
echo "!tar -xzf /content/drive/MyDrive/GraphMER/graphmer_colab.tar.gz -C /content/"
echo "%cd /content/colab_deploy"
echo ""
echo "# Cell 2: Verify data"
echo "!wc -l data/enhanced_multilang.jsonl"
echo ""
echo "# Cell 3: Run training"
echo "!python scripts/train.py --config configs/train_colab.yaml --steps 1000"
echo ""

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                                                              ║${NC}"
echo -e "${BLUE}║                DEPLOYMENT PIPELINE COMPLETE! 🚀              ║${NC}"
echo -e "${BLUE}║                                                              ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
