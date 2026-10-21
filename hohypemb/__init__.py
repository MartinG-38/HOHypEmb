"""HOHypEmb: higher-order hyperbolic embedding (HHO) of simplicial complexes."""
import importlib

from hohypemb.complex import SimplicialComplex
from hohypemb.embedders import ClassicalEmbedder, EmbeddingResult, HHOEmbedder

__version__ = "0.1.0"
__all__ = ["SimplicialComplex", "ClassicalEmbedder", "HHOEmbedder", "EmbeddingResult",
           "geometry", "presets", "markers", "evaluation", "plotting", "datasets", "baselines"]

_LAZY = {
    "geometry": "hohypemb.geometry",
    "presets": "hohypemb.presets",
    "markers": "hohypemb.markers",
    "evaluation": "hohypemb.evaluation",
    "plotting": "hohypemb.plotting",
    "datasets": "hohypemb.datasets",
    "baselines": "hohypemb.embedders.baselines",
}


def __getattr__(name):
    """Import submodules on first access (keeps ``import hohypemb`` free of matplotlib)."""
    if name in _LAZY:
        module = importlib.import_module(_LAZY[name])
        globals()[name] = module
        return module
    raise AttributeError(f"module 'hohypemb' has no attribute {name!r}")
