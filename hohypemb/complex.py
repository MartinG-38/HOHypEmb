"""Simplicial complexes over nodes 0..n-1, as consumed by every HOHypEmb embedder.

Ported from the author's earlier research code (``facets`` renamed ``simplices``) and new helpers for filled/empty triangles and layer-2 views.
"""
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

Simplex = Tuple[int, ...]


@dataclass(eq=False)
class SimplicialComplex:
    """The distinct simplices given as input, over nodes ``0 .. n_nodes - 1``.

    ``simplices`` are deduplicated and sorted on construction but NOT reduced to maximal
    simplices: an input simplex strictly contained in another one keeps its own layer-2
    point in HHO's default mode. Identity equality (``eq=False``) because the
    object carries caches and a possibly shared ``base_graph``.

    Parameters
    ----------
    simplices
        Node-index tuples, any dimension >= 1.
    n_nodes
        Number of nodes; every index must lie in ``0 .. n_nodes - 1``.
    base_graph
        When given, this graph IS the 1-skeleton (weighted); otherwise the skeleton is the
        clique expansion of ``simplices`` with weight = number of simplices containing the pair.
    """

    simplices: List[Simplex]
    n_nodes: int
    name: str = ""
    node_labels: Optional[np.ndarray] = None
    node_names: Optional[List[Hashable]] = None
    base_graph: Optional[nx.Graph] = None
    _skeleton: Optional[nx.Graph] = field(default=None, init=False, repr=False)
    _by_node: Optional[Dict[int, List[frozenset]]] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.n_nodes = int(self.n_nodes)
        label = self.name or "complex"
        cleaned = set()
        for s in self.simplices:
            nodes = tuple(sorted({int(v) for v in s}))
            if not nodes:
                continue
            bad = [v for v in nodes if not 0 <= v < self.n_nodes]
            if bad:
                raise ValueError(
                    f"{label}: simplex {tuple(s)} has node ids {bad} outside 0..{self.n_nodes - 1}; "
                    "node ids must be 0-based and contiguous (use SimplicialComplex.from_graph for "
                    "arbitrary node ids)"
                )
            cleaned.add(nodes)
        self.simplices = sorted(cleaned)
        self._simplex_set = {frozenset(s) for s in self.simplices}
        if self.node_labels is not None:
            self.node_labels = np.asarray(self.node_labels)
        if self.base_graph is not None and set(self.base_graph.nodes()) != set(range(self.n_nodes)):
            raise ValueError(f"{label}: base_graph nodes must be exactly 0..{self.n_nodes - 1}")

    @classmethod
    def from_graph(cls, G: nx.Graph, simplices: Iterable[Iterable[Hashable]], name: str = "",
                   node_labels: Optional[Mapping[Hashable, object]] = None) -> "SimplicialComplex":
        """Build a complex whose 1-skeleton is ``G`` and whose filled simplices are ``simplices``.

        Nodes are relabelled ``0..n-1`` in sorted order of their ids (insertion order when the
        ids cannot be sorted); the original ids are kept in ``node_names``. Edge weights are
        ``G``'s ``"weight"`` attribute, 1.0 when absent. Every simplex must be a clique of ``G``.
        """
        label = name or "complex"
        nodes = list(G.nodes())
        try:
            order = sorted(nodes)
        except TypeError:
            order = nodes
        index = {v: i for i, v in enumerate(order)}
        H = nx.Graph()
        H.add_nodes_from(range(len(order)))
        for u, v, data in G.edges(data=True):
            if u != v:
                H.add_edge(index[u], index[v], weight=float(data.get("weight", 1.0)))
        converted = []
        for s in simplices:
            members = list(dict.fromkeys(s))
            missing = [v for v in members if v not in index]
            if missing:
                raise ValueError(f"{label}: simplex {tuple(members)} contains nodes {missing} that are not in G")
            ids = [index[v] for v in members]
            for a, b in combinations(ids, 2):
                if not H.has_edge(a, b):
                    raise ValueError(
                        f"{label}: simplex {tuple(members)} is not a clique of G (missing edge "
                        f"{order[a]!r}-{order[b]!r}); a filled simplex needs all its edges"
                    )
            converted.append(tuple(ids))
        labels = None
        if node_labels is not None:
            labels = np.array([node_labels.get(v) for v in order], dtype=object)
        return cls(simplices=converted, n_nodes=len(order), name=name, node_labels=labels,
                   node_names=list(order), base_graph=H)

    # ---------------------------------------------------------------- skeleton

    @property
    def one_skeleton(self) -> nx.Graph:
        """Weighted 1-skeleton (``base_graph`` by reference when given; treat as read-only)."""
        if self.base_graph is not None:
            return self.base_graph
        if self._skeleton is None:
            G = nx.Graph()
            G.add_nodes_from(range(self.n_nodes))
            for s in self.simplices:
                for u, v in combinations(s, 2):
                    if G.has_edge(u, v):
                        G[u][v]["weight"] += 1
                    else:
                        G.add_edge(u, v, weight=1)
            self._skeleton = G
        return self._skeleton

    @property
    def adjacency(self) -> np.ndarray:
        return nx.to_numpy_array(self.one_skeleton, nodelist=range(self.n_nodes), weight="weight")

    # -------------------------------------------------------------- simplices

    def _simplices_by_node(self) -> Dict[int, List[frozenset]]:
        if self._by_node is None:
            index: Dict[int, List[frozenset]] = defaultdict(list)
            for s in self.simplices:
                fs = frozenset(s)
                for v in s:
                    index[v].append(fs)
            self._by_node = dict(index)
        return self._by_node

    def higher_order_simplices(self, min_dim: int = 2, max_dim: Optional[int] = None) -> List[List[int]]:
        """Input simplices with ``min_dim <= dim <= max_dim`` (``max_dim=None``: no upper bound)."""
        out = []
        for s in self.simplices:
            dim = len(s) - 1
            if dim < min_dim or (max_dim is not None and dim > max_dim):
                continue
            out.append(list(s))
        return out

    def maximal_simplices(self, min_dim: int = 2, max_dim: Optional[int] = None) -> List[List[int]]:
        """Input simplices of dim >= ``min_dim`` not strictly inside another input simplex, then
        filtered by ``max_dim``."""
        by_node = self._simplices_by_node()
        out = []
        for s in self.simplices:
            dim = len(s) - 1
            if dim < min_dim:
                continue
            fs = frozenset(s)
            if any(len(t) > len(fs) and fs < t for t in by_node[s[0]]):
                continue
            if max_dim is not None and dim > max_dim:
                continue
            out.append(list(s))
        return out

    def faces(self, min_dim: int = 2, *, max_dim: Optional[int]) -> List[Simplex]:
        """Every face of dimension ``min_dim..max_dim`` of the input simplices, sorted."""
        if max_dim is None:
            raise ValueError("faces() needs max_dim: the number of faces grows combinatorially "
                             "with simplex size")
        out = set()
        for s in self.simplices:
            for k in range(min_dim, min(max_dim, len(s) - 1) + 1):
                out.update(combinations(s, k + 1))
        return sorted(out)

    def is_input_simplex(self, group: Sequence[int]) -> bool:
        return frozenset(int(v) for v in group) in self._simplex_set

    def is_filled(self, nodes: Sequence[int]) -> bool:
        """True iff ``nodes`` (3 or more) lies inside an input simplex of dimension >= 2."""
        fs = frozenset(int(v) for v in nodes)
        if len(fs) < 3:
            return False
        first = next(iter(fs))
        return any(len(t) >= 3 and fs <= t for t in self._simplices_by_node().get(first, []))

    def incidence_matrix(self) -> csr_matrix:
        """Sparse (n_simplices, n_nodes) 0/1 incidence matrix."""
        rows, cols = [], []
        for i, s in enumerate(self.simplices):
            for node in s:
                rows.append(i)
                cols.append(node)
        data = np.ones(len(rows), dtype=np.float64)
        return coo_matrix((data, (rows, cols)), shape=(len(self.simplices), self.n_nodes)).tocsr()

    def shared_count_matrix(self) -> np.ndarray:
        """(n_nodes, n_nodes): number of input simplices containing both nodes; zero diagonal."""
        M = self.incidence_matrix()
        S = (M.T @ M).toarray().astype(float)
        np.fill_diagonal(S, 0.0)
        return S

    # -------------------------------------------------------------- triangles

    def triangles_in_skeleton(self) -> List[Tuple[int, int, int]]:
        """Every 3-clique of the 1-skeleton. Stops before larger cliques (they are generated
        in increasing size): enumerating them all never finishes on dense graphs."""
        out = []
        for c in nx.enumerate_all_cliques(self.one_skeleton):
            if len(c) > 3:
                break
            if len(c) == 3:
                out.append(tuple(sorted(c)))
        return out

    def filled_triangles(self) -> List[Tuple[int, int, int]]:
        """Triangles of the skeleton lying inside an input simplex of dim >= 2, sorted."""
        return sorted(t for t in self.triangles_in_skeleton() if self.is_filled(t))

    def empty_triangles(self) -> List[Tuple[int, int, int]]:
        """Triangles of the skeleton that are not filled, sorted."""
        return sorted(t for t in self.triangles_in_skeleton() if not self.is_filled(t))

    # ------------------------------------------------------------------ names

    def indices_of(self, names: Iterable[Hashable]) -> List[int]:
        """Indices of the given original node ids, silently dropping ids absent from the complex."""
        if self.node_names is None:
            raise ValueError(f"{self.name or 'complex'} has no node_names")
        lookup = {n: i for i, n in enumerate(self.node_names)}
        return [lookup[n] for n in names if n in lookup]

    # -------------------------------------------------------------- components

    def giant_component(self) -> "SimplicialComplex":
        """Restrict to the largest connected component of the 1-skeleton, reindexed to 0..m-1."""
        G = self.one_skeleton
        keep = sorted(max(nx.connected_components(G), key=len))
        remap = {old: new for new, old in enumerate(keep)}
        keep_set = set(keep)
        new_simplices = [tuple(remap[v] for v in s) for s in self.simplices if set(s) <= keep_set]
        new_base = None
        if self.base_graph is not None:
            new_base = nx.relabel_nodes(G.subgraph(keep).copy(), remap, copy=True)
        return SimplicialComplex(
            simplices=new_simplices,
            n_nodes=len(keep),
            name=f"{self.name}_giant",
            node_labels=None if self.node_labels is None else self.node_labels[keep],
            node_names=None if self.node_names is None else [self.node_names[i] for i in keep],
            base_graph=new_base,
        )
