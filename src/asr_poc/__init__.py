"""AI-assisted Ancestral Sequence Reconstruction (ASR) of enzymes — POC package.

Thin, importable helpers shared by the work-package notebooks. Each module maps
to one work package:

    retrieval  -> WP1  Sequence dataset build
    phylo      -> WP2  Alignment, phylogeny & ASR
    ranking    -> WP3  AI-based candidate prioritization
    structure  -> WP4  Structural modeling & analysis

Configuration and IO are centralised in `config` and `io_utils` so notebooks
never hardcode paths or magic numbers.
"""

__version__ = "0.1.0"
