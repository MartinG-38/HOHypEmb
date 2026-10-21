"""cora co-authorship (AHORN 0.3 format) loader."""
import gzip
import json
import os
import shutil
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np

from hohypemb.complex import SimplicialComplex
from hohypemb.datasets._download import PathLike, fetch, resolve_data_dir, warn_if_changed
from hohypemb.datasets.reference import REFERENCE_SHA256

CORA_URL = "https://zenodo.org/records/19232933/files/cora-coauthorship.txt.gz?download=1"
"""AHORN cora-coauthorship, revision 2 (Zenodo record 19232933). Licence listed as "Unknown" by
AHORN, hence downloaded, never redistributed."""


def cora_path(data_dir: Optional[PathLike] = None) -> Path:
    """Local path of ``cora-coauthorship.txt``, downloading and decompressing it if needed."""
    txt = resolve_data_dir(data_dir) / "cora-coauthorship.txt"
    if not txt.exists():
        gz = fetch(CORA_URL, "cora-coauthorship.txt.gz", data_dir=data_dir)
        part = txt.with_name(txt.name + ".part")
        try:
            with gzip.open(gz, "rb") as src, open(part, "wb") as out:
                shutil.copyfileobj(src, out)
            os.replace(part, txt)
        except (OSError, EOFError) as exc:
            if part.exists():
                part.unlink()
            raise RuntimeError(
                f"Could not decompress {gz} ({exc}). The cached archive is corrupt or truncated: "
                f"delete it to re-download, or pass data_dir= a folder that already contains "
                f"cora-coauthorship.txt.") from exc
    warn_if_changed(txt, REFERENCE_SHA256["cora-coauthorship.txt"])
    return txt


def load_cora_coauthorship(path: Optional[Union[str, Path]] = None, max_dim: int = 5,
                           data_dir: Optional[PathLike] = None) -> SimplicialComplex:
    """Load the cora co-authorship complex (AHORN 0.3 format).

    ``path=None`` downloads AHORN revision 2 from Zenodo into the data cache (see ``cora_path``).
    Line 1 of the file is a JSON header. Each following line is ``<ids> <json>``, split on the
    FIRST space. A line whose JSON has ``category`` or ``name`` declares an author; every other
    line is an article. The 1-skeleton is built from ALL articles (weight = number of articles
    joining the pair), while ``simplices`` keeps only articles with ``1 <= dim <= max_dim``.
    """
    p = Path(path) if path is not None else cora_path(data_dir)
    if not p.exists():
        raise FileNotFoundError(f"Not found: {p.resolve()}")

    node_attrs: Dict[int, Dict[str, str]] = {}
    articles: List[Tuple[int, ...]] = []
    with p.open("r", encoding="utf-8") as f:
        json.loads(f.readline())
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            left, right = line.split(" ", 1)
            attrs = json.loads(right)
            ids = tuple(sorted({int(tok) for tok in left.split(",") if tok}))
            if "category" in attrs or "name" in attrs:
                node_attrs[ids[0]] = {"name": attrs.get("name", "?"),
                                      "category": attrs.get("category", "Unknown")}
            else:
                articles.append(ids)

    author_ids = sorted(node_attrs)
    remap = {a: i for i, a in enumerate(author_ids)}
    n_nodes = len(author_ids)

    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    for art in articles:
        idx = [remap[a] for a in art if a in remap]
        for u, v in combinations(idx, 2):
            if G.has_edge(u, v):
                G[u][v]["weight"] += 1
            else:
                G.add_edge(u, v, weight=1)

    simplices = [tuple(remap[a] for a in art if a in remap)
                 for art in articles if 1 <= len(art) - 1 <= max_dim]
    simplices = [s for s in simplices if len(s) >= 2]

    return SimplicialComplex(
        simplices=simplices,
        n_nodes=n_nodes,
        name="cora-coauthorship",
        node_labels=np.array([node_attrs[a]["category"] for a in author_ids]),
        node_names=[node_attrs[a]["name"] for a in author_ids],
        base_graph=G,
    )
