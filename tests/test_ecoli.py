import gzip
import io
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import hohypemb.datasets.ecoli as ecoli_module
from hohypemb.datasets.ecoli import _fetch_cog_mappings, load_ecoli
from tests._data import SNAPSHOT_DIR, load_golden

ROOT = Path(__file__).resolve().parents[1]
HEADER = ["Complex ac", "Recommended name", "Aliases", "Taxonomy identifier",
          "Identifiers (and stoichiometry) of molecules in complex", "Expanded participant list"]


def _write_tiny(folder: Path) -> None:
    """K6 on b1..b6 (4-core), a pendant b7, complexes {b1..b4}, {b4,b5,b6}, {b1,b7,b8}."""
    nodes = [f"b{i}" for i in range(1, 7)]
    lines = ["protein1 protein2 combined_score"]
    for i, u in enumerate(nodes):
        for v in nodes[i + 1:]:
            lines += [f"511145.{u} 511145.{v} 900", f"511145.{v} 511145.{u} 900"]
    lines += ["511145.b1 511145.b7 900", "511145.b1 511145.b8 100"]
    with gzip.open(folder / "511145.protein.links.v12.0.txt.gz", "wt") as f:
        f.write("\n".join(lines) + "\n")
    aliases = ["#string_protein_id\talias\tsource"]
    aliases += [f"511145.b{i}\tP{i:05d}\tUniProt_AC" for i in range(1, 9)]
    with gzip.open(folder / "511145.protein.aliases.v12.0.txt.gz", "wt") as f:
        f.write("\n".join(aliases) + "\n")
    rows = ["\t".join(HEADER),
            "\t".join(["CPX-1", "tetra", "-", "83333", "-", "P00001(1)|P00002(1)|P00003(1)|P00004-2(1)"]),
            "\t".join(["CPX-2", "trio", "-", "83333", "-", "P00004(1)|P00005(1)|P00006(1)"]),
            "\t".join(["CPX-3", "open", "-", "83333", "-", "P00001(1)|P00007(1)|P00008(1)"])]
    (folder / "83333.complextab.tsv").write_text("\n".join(rows) + "\n")


def test_tiny_offline_dataset(tmp_path):
    _write_tiny(tmp_path)
    with pytest.warns(UserWarning):          # tiny files differ from the paper's checksums
        sc, complexes = load_ecoli(max_nodes=300, data_dir=tmp_path, return_complexes=True)
    assert sc.node_names == ["b1", "b2", "b3", "b4", "b5", "b6", "b7"]   # b8 only has a weak link
    ids = {n: i for i, n in enumerate(sc.node_names)}
    assert set(sc.simplices) == {tuple(sorted(ids[n] for n in c)) for c in (["b1", "b2", "b3", "b4"],
                                                                            ["b4", "b5", "b6"])}
    assert complexes == {"tetra": frozenset({"b1", "b2", "b3", "b4"}), "trio": frozenset({"b4", "b5", "b6"}),
                         "open": frozenset({"b1", "b7", "b8"})}


def test_max_nodes_truncates_the_core(tmp_path):
    _write_tiny(tmp_path)
    with pytest.warns(UserWarning):
        sc = load_ecoli(max_nodes=4, data_dir=tmp_path)
    assert sc.n_nodes == 4


@pytest.mark.slow
def test_matches_the_original_analysis_on_the_paper_snapshot():
    if SNAPSHOT_DIR is None:
        pytest.skip("needs HOHYPEMB_SNAPSHOT_DIR")
    d = load_golden("G4_ecoli_300_structure")
    sc = load_ecoli(max_nodes=300, data_dir=SNAPSHOT_DIR).giant_component()
    assert sc.node_names == [str(x) for x in d["nodelist"]]
    assert {tuple(sorted(e)) for e in sc.one_skeleton.edges()} == {tuple(e) for e in d["edges"]}
    assert sc.faces(max_dim=2) == [tuple(t) for t in d["triangles"]]
    assert sc.filled_triangles() == [tuple(t) for t in d["triangles"]]
    assert len(sc.empty_triangles()) == int(d["counts"][3])


@pytest.mark.slow
def test_builder_does_not_depend_on_python_hash_seed():
    if SNAPSHOT_DIR is None:
        pytest.skip("needs HOHYPEMB_SNAPSHOT_DIR")
    script = textwrap.dedent(f"""
        from hohypemb.datasets.ecoli import load_ecoli
        sc = load_ecoli(max_nodes=300, data_dir={str(SNAPSHOT_DIR)!r})
        print(repr((sc.node_names, sc.simplices)))
    """)
    outs = []
    for seed in ("0", "1"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        outs.append(subprocess.run([sys.executable, "-c", script], cwd=ROOT, env=env, capture_output=True,
                                   text=True, check=True).stdout.splitlines()[-1])
    assert outs[0] == outs[1]


def test_cog_labels(tmp_path):
    _write_tiny(tmp_path)
    (tmp_path / "511145.cog.mappings.txt").write_text(
        "511145.b1\tx\tx\tCOG0001\n511145.b2\tx\tx\tCOG0002\n")
    (tmp_path / "cog-20.def.tab").write_text("COG0001\tJK\tx\nCOG0002\tE\tx\n", encoding="latin-1")
    (tmp_path / "fun-20.tab").write_text("J\tx\tTranslation\nE\tx\tAmino acid transport\n",
                                         encoding="latin-1")
    with pytest.warns(UserWarning):
        sc = load_ecoli(data_dir=tmp_path, with_cog=True)
    labels = dict(zip(sc.node_names, sc.node_labels))
    assert labels["b1"] == "Translation" and labels["b2"] == "Amino acid transport"
    assert labels["b3"] == "Unknown category"


def test_fetch_cog_mappings_cleans_up_on_truncated_gzip(tmp_path, monkeypatch):
    data = b"".join(f"511145.b{i}\tx\tx\tCOG000{i}\n".encode() for i in range(1, 5))
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as f:
        f.write(data)
    full = buf.getvalue()
    truncated_gz = tmp_path / "truncated.cog.gz"
    truncated_gz.write_bytes(full[: len(full) // 2])

    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(ecoli_module, "COG_MAPPINGS_URL", truncated_gz.as_uri())
    monkeypatch.setattr(ecoli_module.time, "sleep", lambda *_: None)

    with pytest.raises(RuntimeError):
        _fetch_cog_mappings(cache_dir)

    assert not (cache_dir / "511145.cog.mappings.txt.part").exists()
    assert not (cache_dir / "511145.cog.mappings.txt").exists()


def test_fetch_cog_mappings_retries_a_truncated_stream(tmp_path, monkeypatch):
    from tests.test_download import TruncatedResponse
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as f:
        f.write(b"".join(f"511145.b{i}\tx\tx\tCOG000{i}\n".encode() for i in range(1, 5)))
    head = buf.getvalue()[:-8]          # valid gzip data; the stream breaks before the trailer
    calls, sleeps = [], []

    def truncated(*args, **kwargs):
        calls.append(1)
        return TruncatedResponse(head)

    monkeypatch.setattr(ecoli_module.urllib.request, "urlopen", truncated)
    monkeypatch.setattr(ecoli_module.time, "sleep", sleeps.append)
    cache_dir = tmp_path / "cache"
    with pytest.raises(RuntimeError, match="Could not download"):
        _fetch_cog_mappings(cache_dir)
    assert len(calls) == 3 and len(sleeps) == 2
    assert not (cache_dir / "511145.cog.mappings.txt.part").exists()
    assert not (cache_dir / "511145.cog.mappings.txt").exists()
