"""Embedders: classical, HHO and the six baselines of the 2x2 design."""
from hohypemb.embedders.base import Embedder, EmbeddingResult
from hohypemb.embedders.baselines import (
    EuclideanIsomapEmbedder,
    EuclideanSpectralEmbedder,
    EuclideanStarExpansionEmbedder,
    StarExpansionEmbedder,
    WeightedCliqueExpansionEmbedder,
    ZhouHypergraphEmbedder,
    build_registry,
)
from hohypemb.embedders.classical import ClassicalEmbedder
from hohypemb.embedders.hho import HHOEmbedder

__all__ = ["Embedder", "EmbeddingResult", "ClassicalEmbedder", "HHOEmbedder",
           "WeightedCliqueExpansionEmbedder", "StarExpansionEmbedder", "ZhouHypergraphEmbedder",
           "EuclideanSpectralEmbedder", "EuclideanIsomapEmbedder", "EuclideanStarExpansionEmbedder",
           "build_registry"]
