"""Synthetic generators with planted higher-order structure."""

import random
from itertools import combinations
from typing import List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from hohypemb.complex import SimplicialComplex


def _connected_background(n_nodes: int, m: int, p_tri: float, seed: int) -> nx.Graph:
    G = nx.powerlaw_cluster_graph(n_nodes, m, p_tri, seed=seed)
    if nx.number_connected_components(G) > 1:
        components = sorted(nx.connected_components(G), key=len, reverse=True)
        hub = min(components[0])
        for comp in components[1:]:
            G.add_edge(hub, min(comp))
    return G


def identical_skeleton_pair(
    n_nodes: int = 300,
    K: int = 4,
    n_planted_triangles: int = 400,
    m_background: int = 2,
    p_background_tri: float = 0.3,
    n_fill: Optional[int] = None,
    seed: int = 0,
) -> Tuple[SimplicialComplex, SimplicialComplex, np.ndarray]:
    """Two complexes with a BIT-IDENTICAL 1-skeleton and different simplicial filling.

    This is the decisive control for claim C2. The paper's headline measurement is a
    difference-in-differences: ``Delta(method) = NMI(method, aligned) - NMI(method,
    random)``. For that quantity to isolate the contribution of the *higher-order*
    layer, the two conditions must be identical in every respect a 1-skeleton-only
    method can see, and differ ONLY in which triangles are declared "filled" as
    2-simplices. If the two graphs were merely statistically similar (e.g. drawn from
    the same generator with different seeds), any Delta could be explained away as
    noise in the pairwise structure. By handing both complexes the SAME graph object
    as ``base_graph``, the adjacency matrices are not just equal but bit-identical:
    a method reading only edges is *provably*, not empirically, blind to the
    difference between the two conditions, so any nonzero Delta it exhibits can only
    be a bug, and any nonzero Delta a higher-order method exhibits can only be
    attributed to information from filled simplices. This is what makes Delta a
    clean measurement instrument rather than a suggestive correlation.

    Construction:

    1. Assign every node a community label at random.
    2. Build a background graph WITHOUT reference to the labels.
    3. Sample ``n_planted_triangles`` triples *within* communities and add their
       three edges, guaranteeing a supply of community-pure triangles.
    4. The 'aligned' complex fills the community-pure triangles; the 'random'
       complex fills an equal number of triangles drawn uniformly.

    Both complexes are handed the SAME graph object as ``base_graph``, so any
    method reading only the 1-skeleton produces identical output on both and
    its difference-in-differences is exactly zero.

    Returns
    -------
    aligned, random_control, labels
    """
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, K, n_nodes)

    G = _connected_background(n_nodes, m_background, p_background_tri, seed)

    by_community: List[np.ndarray] = [np.flatnonzero(labels == k) for k in range(K)]
    for _ in range(n_planted_triangles):
        members = by_community[rng.integers(0, K)]
        if len(members) < 3:
            continue
        trio = rng.choice(members, size=3, replace=False)
        for u, v in combinations(sorted(int(x) for x in trio), 2):
            if not G.has_edge(u, v):
                G.add_edge(u, v, weight=1)
    nx.set_edge_attributes(G, 1.0, "weight")

    triangles = [tuple(sorted(c)) for c in nx.enumerate_all_cliques(G) if len(c) == 3]
    pure = [t for t in triangles if len(set(labels[list(t)])) == 1]
    if not pure:
        raise RuntimeError(
            "No community-pure triangles produced; raise n_planted_triangles or lower K."
        )

    if n_fill is not None and n_fill < len(pure):
        pure = [pure[i] for i in rng.choice(len(pure), size=n_fill, replace=False)]

    size = len(pure)
    control = [
        triangles[i] for i in rng.choice(len(triangles), size=size, replace=False)
    ]

    aligned = SimplicialComplex(
        simplices=pure, n_nodes=n_nodes, name="control_aligned",
        node_labels=labels, base_graph=G,
    )
    random_control = SimplicialComplex(
        simplices=control, n_nodes=n_nodes, name="control_random",
        node_labels=labels, base_graph=G,
    )
    return aligned, random_control, labels


def planted_hypergraph_sbm(
    n_nodes: int = 200,
    K: int = 4,
    n_facets: int = 300,
    facet_sizes: Sequence[int] = (3, 4),
    p_in: float = 0.9,
    m_background: int = 2,
    p_background_tri: float = 0.3,
    seed: int = 0,
) -> SimplicialComplex:
    """Hypergraph stochastic block model: facets drawn preferentially within blocks.

    A conventional higher-order benchmark, used as a second synthetic row in the
    ablation grid and as the substrate for robustness curves. Unlike
    ``identical_skeleton_pair`` the 1-skeleton here DOES carry community signal,
    so pairwise baselines are expected to score above chance.
    """
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, K, n_nodes)
    by_community = [np.flatnonzero(labels == k) for k in range(K)]

    facets: List[Tuple[int, ...]] = []
    for _ in range(n_facets):
        size = int(rng.choice(list(facet_sizes)))
        if rng.random() < p_in:
            members = by_community[rng.integers(0, K)]
            if len(members) < size:
                continue
            chosen = rng.choice(members, size=size, replace=False)
        else:
            chosen = rng.choice(n_nodes, size=size, replace=False)
        facets.append(tuple(sorted(int(x) for x in chosen)))

    G = _connected_background(n_nodes, m_background, p_background_tri, seed)
    for facet in facets:
        for u, v in combinations(facet, 2):
            if not G.has_edge(u, v):
                G.add_edge(u, v)
    nx.set_edge_attributes(G, 1.0, "weight")

    return SimplicialComplex(
        simplices=facets, n_nodes=n_nodes, name="planted_hsbm",
        node_labels=labels, base_graph=G,
    )


_GENERATORS = {
    # model: (default seed, default graph parameters, generator)
    # the paper's configuration (the author's original simulation code, as of 2026-09-25)
    "holme_kim": (1000, {"n": 500, "m": 2, "p": 0.3},
                  lambda rng, n, m, p: nx.powerlaw_cluster_graph(n, m, p, seed=rng)),
    "small_world": (104, {"n": 80, "k": 4, "p": 0.3},
                    lambda rng, n, k, p: nx.watts_strogatz_graph(n, k, p, seed=rng)),
    # erdos_renyi: same size as holme_kim and about as many triangles (175 against 178)
    "erdos_renyi": (103, {"n": 500, "p": 0.02},
                    lambda rng, n, p: nx.erdos_renyi_graph(n, p, seed=rng)),
    "barabasi_albert": (102, {"n": 100, "m": 2},
                        lambda rng, n, m: nx.barabasi_albert_graph(n, m, seed=rng)),
}
MODELS = tuple(_GENERATORS)


def planted_surfaces(model: str = "holme_kim", p_surface: float = 50, seed: Optional[int] = None,
                     **graph_params) -> SimplicialComplex:
    """Random graph whose triangles are each filled with probability ~``p_surface`` %.

    Replicates the author's original simulation code bit for bit: one ``random.Random(seed)``
    drives the networkx generator and then one ``randint(0, 100) < p_surface`` draw per
    triangle, in ``nx.enumerate_all_cliques`` order (so the probability is p_surface / 101,
    as in the notebook). The global ``random`` state is not touched. ``graph_params``
    overrides the model's defaults (e.g. ``n=80, p=0.25`` for holme_kim).
    """
    if model not in _GENERATORS:
        raise ValueError(f"model must be one of {MODELS}, got {model!r}")
    default_seed, defaults, make = _GENERATORS[model]
    unknown = sorted(set(graph_params) - set(defaults))
    if unknown:
        raise TypeError(f"{model} takes graph parameters {sorted(defaults)}, got {unknown}")
    rng = random.Random(default_seed if seed is None else seed)
    G = make(rng, **{**defaults, **graph_params})
    triangles = []
    for clique in nx.enumerate_all_cliques(G):
        if len(clique) > 3:
            break
        if len(clique) == 3:
            triangles.append(clique)
    surfaces = [t for t in triangles if rng.randint(0, 100) < p_surface]
    return SimplicialComplex.from_graph(G, surfaces, name=f"{model}_p{p_surface}")
