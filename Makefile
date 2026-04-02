
.PHONY: fasta phylogeny phylogeny-only visualize basic clean

fasta:
	python3 tree.py

# Regenerate FASTA, align with MAFFT, build tree with FastTree
phylogeny:
	python3 phylogeny.py

# Use existing hiv_sequences.fasta only (no JSON re-export)
phylogeny-only:
	python3 phylogeny.py --skip-fasta

# Plot tree + leaf paths CSV (needs: pip install -r requirements.txt)
visualize:
	python3 visualize_tree.py

# End-to-end basic pipeline (shared paths in pipeline_paths.py — import from other stages)
basic:
	python3 pipeline_basic.py

clean:
	rm -f hiv_sequences_aligned.fasta hiv_tree.nwk hiv_tree.png hiv_tree.svg leaf_paths.csv hadsbm_tree.json
