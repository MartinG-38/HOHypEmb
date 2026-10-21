"""E. coli simplicial complex: STRING v12 interactions + EMBL-EBI Complex Portal.

Faithful port of the author's original E. coli analysis (construction and COG labels). Filled simplices are
the maximal cliques, inside the subnetwork, of each annotated complex's members; their faces of
dimension 2..k are exactly that analysis's filled simplices.
"""
import csv
import gzip
import http.client
import os
import re
import time
import urllib.request
from typing import Dict, FrozenSet, List, Optional, Set, Tuple, Union

import networkx as nx
import pandas as pd

from hohypemb.complex import SimplicialComplex
from hohypemb.datasets._download import USER_AGENT, PathLike, fetch, resolve_data_dir, warn_if_changed
from hohypemb.datasets.reference import REFERENCE_SHA256

TAXON = "511145."
STRING_LINKS = ("https://stringdb-downloads.org/download/protein.links.v12.0/511145.protein.links.v12.0.txt.gz",
                "511145.protein.links.v12.0.txt.gz")
STRING_ALIASES = ("https://stringdb-downloads.org/download/protein.aliases.v12.0/"
                  "511145.protein.aliases.v12.0.txt.gz", "511145.protein.aliases.v12.0.txt.gz")
COMPLEX_PORTAL = ("https://ftp.ebi.ac.uk/pub/databases/intact/complex/current/complextab/83333.tsv",
                  "83333.complextab.tsv")
COG_MAPPINGS_URL = "https://stringdb-downloads.org/download/COG.mappings.v12.0.txt.gz"
COG_MAPPINGS_FILE = "511145.cog.mappings.txt"
COG_DEF = ("https://ftp.ncbi.nih.gov/pub/COG/COG2020/data/cog-20.def.tab", "cog-20.def.tab")
COG_FUN = ("https://ftp.ncbi.nih.gov/pub/COG/COG2020/data/fun-20.tab", "fun-20.tab")
_ID_PATTERN = re.compile(r"([^()|]+)\((?:[^()]*)\)")


def _get(source: Tuple[str, str], data_dir: Optional[PathLike]):
    url, filename = source
    return fetch(url, filename, sha256=REFERENCE_SHA256.get(filename), data_dir=data_dir)


def _string_graph(data_dir, score_threshold: int) -> nx.Graph:
    links = pd.read_csv(_get(STRING_LINKS, data_dir), sep=" ")
    links = links[links["combined_score"] >= score_threshold]
    G = nx.Graph()
    for p1, p2 in zip(links["protein1"], links["protein2"]):   # file order = insertion order of the original analysis
        G.add_edge(p1.replace(TAXON, ""), p2.replace(TAXON, ""))
    return G


def _subnetwork(G: nx.Graph, max_nodes: int, kcore_k: int) -> nx.Graph:
    """Exact loop semantics of the original analysis's subnetwork extraction."""
    core_nodes = list(nx.k_core(G, k=kcore_k).nodes())
    if len(core_nodes) > max_nodes:
        return G.subgraph(core_nodes[:max_nodes]).copy()
    nodes_subset = set(core_nodes)
    for n in core_nodes:
        for neighbor in G.neighbors(n):
            nodes_subset.add(neighbor)
            if len(nodes_subset) >= max_nodes:
                break
        if len(nodes_subset) >= max_nodes:
            break
    return G.subgraph(list(nodes_subset)).copy()


def _uniprot_to_string(data_dir) -> Dict[str, Set[str]]:
    mapping: Dict[str, Set[str]] = {}
    with gzip.open(_get(STRING_ALIASES, data_dir), "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            string_id, alias, source = line.rstrip("\n").split("\t")
            if source == "UniProt_AC":
                mapping.setdefault(alias, set()).add(string_id.replace(TAXON, ""))
    return mapping


def _complexes(data_dir, uniprot: Dict[str, Set[str]]) -> List[Tuple[str, FrozenSet[str]]]:
    """(recommended name, STRING members) for every complex with >= 3 mapped proteins, file order."""
    with open(_get(COMPLEX_PORTAL, data_dir), encoding="utf-8") as f:
        rows = list(csv.reader(f, delimiter="\t"))
    header, rows = rows[0], rows[1:]
    col_participants = header.index("Expanded participant list")
    col_name = header.index("Recommended name")
    out = []
    for row in rows:
        members: Set[str] = set()
        for token in _ID_PATTERN.findall(row[col_participants]):
            members |= uniprot.get(token.split("-")[0], set())
        if len(members) >= 3:
            out.append((row[col_name], frozenset(members)))
    return out


def _fetch_cog_mappings(data_dir) -> str:
    """STRING's all-organism COG table (~750 MB) streamed and filtered to TaxID 511145."""
    target = resolve_data_dir(data_dir) / COG_MAPPINGS_FILE
    if not target.exists():
        part = target.with_name(target.name + ".part")
        last_error = None
        for attempt in range(1, 4):
            try:
                request = urllib.request.Request(COG_MAPPINGS_URL, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(request, timeout=300) as response, \
                        gzip.GzipFile(fileobj=response) as gz, open(part, "w", encoding="utf-8") as out:
                    for raw in gz:
                        if raw.startswith(TAXON.encode()):
                            out.write(raw.decode("utf-8"))
                os.replace(part, target)
                break
            except (OSError, EOFError, UnicodeError, http.client.HTTPException) as exc:
                last_error = exc
                if part.exists():
                    part.unlink()
                if attempt < 3:
                    time.sleep(5)
        else:
            raise RuntimeError(f"Could not download {COG_MAPPINGS_URL} ({last_error}); place the "
                               f"TaxID-511145 lines of it as {target}.") from last_error
    warn_if_changed(target, REFERENCE_SHA256[COG_MAPPINGS_FILE])
    return str(target)


def _cog_labels(nodes: List[str], data_dir) -> Dict[str, str]:
    protein_to_cog: Dict[str, str] = {}
    with open(_fetch_cog_mappings(data_dir), encoding="utf-8") as f:
        for line in f:
            fields = line.rstrip("\n").split("\t")
            protein_to_cog[fields[0].replace(TAXON, "")] = fields[3]
    cog_to_letter: Dict[str, str] = {}
    with open(_get(COG_DEF, data_dir), encoding="latin-1") as f:
        for line in f:
            fields = line.rstrip("\n").split("\t")
            cog_to_letter[fields[0]] = fields[1][0]
    letter_to_name: Dict[str, str] = {}
    with open(_get(COG_FUN, data_dir), encoding="latin-1") as f:
        for line in f:
            fields = line.rstrip("\n").split("\t")
            letter_to_name[fields[0]] = fields[2]
    labels = {}
    for node in nodes:
        name = letter_to_name.get(cog_to_letter.get(protein_to_cog.get(node, ""), ""), None)
        labels[node] = name if name else "Unknown category"
    return labels


def load_ecoli(max_nodes: int = 300, score_threshold: int = 700, kcore_k: int = 4, with_cog: bool = False,
               return_complexes: bool = False, data_dir: Optional[PathLike] = None
               ) -> Union[SimplicialComplex, Tuple[SimplicialComplex, Dict[str, FrozenSet[str]]]]:
    """E. coli K-12 protein complex network as a simplicial complex.

    Parameters
    ----------
    max_nodes, score_threshold, kcore_k
        Subnetwork size, STRING combined-score threshold, k of the seed k-core.
    with_cog
        Node labels = first COG functional category (downloads ~750 MB once if not cached).
    return_complexes
        Also return ``{recommended name: STRING ids of mapped members}`` for complexes with >= 3
        mapped proteins (same-name complexes merged, not restricted to the subnetwork).
    data_dir
        Folder holding (or receiving) the data files; nothing is downloaded when they are present.

    The caller applies ``giant_component()``.
    """
    G_sub = _subnetwork(_string_graph(data_dir, score_threshold), max_nodes, kcore_k)
    complexes = _complexes(data_dir, _uniprot_to_string(data_dir))
    in_sub = set(G_sub.nodes())
    simplices = set()
    for _, members in complexes:
        for clique in nx.find_cliques(G_sub.subgraph(members & in_sub)):
            if len(clique) >= 3:
                simplices.add(frozenset(clique))
    labels = _cog_labels(sorted(in_sub), data_dir) if with_cog else None
    sc = SimplicialComplex.from_graph(G_sub, [sorted(s) for s in simplices], name="ecoli", node_labels=labels)
    if not return_complexes:
        return sc
    named: Dict[str, Set[str]] = {}
    for name, members in complexes:
        named.setdefault(name, set()).update(members)
    return sc, {name: frozenset(members) for name, members in named.items()}
