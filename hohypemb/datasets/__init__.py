"""Dataset loaders and generators: synthetic, cora co-authorship, E. coli.

Nothing is downloaded at import time: the loaders fetch (and cache) their files when called.
"""
from hohypemb.datasets import cora, ecoli, synthetic
from hohypemb.datasets.cora import load_cora_coauthorship
from hohypemb.datasets.ecoli import load_ecoli
from hohypemb.datasets.synthetic import identical_skeleton_pair, planted_hypergraph_sbm, planted_surfaces

__all__ = ["cora", "ecoli", "synthetic", "load_cora_coauthorship", "load_ecoli", "identical_skeleton_pair",
           "planted_hypergraph_sbm", "planted_surfaces"]
