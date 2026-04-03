
.PHONY: help test setup verify tree binding mog-dfm clean

# Default help
help:
	@echo "hadsbm-hiv GPU-Ready Pipeline"
	@echo "=============================="
	@echo ""
	@echo "For GPU clusters: See GPU_QUICK_START.md"
	@echo ""
	@echo "Available targets:"
	@echo "  make setup    - Install Python dependencies"
	@echo "  make test     - Run smoke tests (no GPU required)"
	@echo "  make verify   - Verify installation"
	@echo "  make tree     - Build phylogenetic tree (MAFFT + FastTree)"
	@echo "  make binding  - Evaluate random peptides with binding affinity"
	@echo "  make mog-dfm  - Generate peptides optimized with MOG-DFM"
	@echo "  make clean    - Remove generated files"
	@echo ""

# Setup and verification
setup:
	pip install --upgrade pip
	pip install -r requirements.txt
	@if [ ! -d PeptiVerse ]; then \
		echo "Cloning PeptiVerse..."; \
		git clone https://huggingface.co/ChatterjeeLab/PeptiVerse; \
	fi

test:
	@python test_pipeline.py

verify: test
	@echo "✓ Installation verified"

# Tree analysis pipeline
TREE_SRC := tree_analysis/src

tree:
	@echo "Building phylogenetic tree..."
	@cd tree_analysis && \
	python $(TREE_SRC)/tree.py --json ../data/variants/hiv-variants.json --out ../data/sequences/hiv_sequences.fasta --verbose && \
	python $(TREE_SRC)/phylogeny.py && \
	python $(TREE_SRC)/hadsbm_export.py --prob-mode length
	@echo "✓ Tree saved to: data/trees/hadsbm_tree.json"

# Binding affinity (GPU target)
PEPTIDE_SRC := peptide_optimization/src

binding:
	@echo "Computing binding affinity for random peptides (CPU mode)..."
	@echo "For GPU: python $(PEPTIDE_SRC)/binding_affinity_simple.py --device cuda:0"
	python $(PEPTIDE_SRC)/binding_affinity_simple.py \
		--tree-json data/trees/hadsbm_tree.json \
		--num-peptides 3 \
		--device cpu

mog-dfm:
	@echo "Generating peptides with MOG-DFM (tree-weighted binding objective)..."
	@echo "GPU mode by default - override with --device cpu"
	python $(PEPTIDE_SRC)/mog_dfm_binding.py \
		--tree-json data/trees/hadsbm_tree.json \
		--num-peptides 3 \
		--length 12 \
		--device cuda:0

# Cleanup
clean:
	@echo "Removing generated files..."
	rm -rf data/sequences/*.fasta data/trees/*.nwk data/trees/*.json
	rm -rf __pycache__ peptide_optimization/__pycache__ tree_analysis/__pycache__
	@echo "✓ Cleaned"
