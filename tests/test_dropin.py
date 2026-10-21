"""HOHypEmb works without installation: copy the `hohypemb/` folder next to your code and import it."""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCRIPT = """
import networkx as nx
import hohypemb as hh
print(hh.__file__)
G = nx.Graph([("ana", "ben"), ("ana", "cleo"), ("ben", "cleo"), ("cleo", "dan"),
              ("dan", "eve"), ("cleo", "eve"), ("eve", "fay"), ("dan", "fay")])
sc = hh.SimplicialComplex.from_graph(G, [("ana", "ben", "cleo")])
res = hh.HHOEmbedder(mu=15, gamma=0.5, **hh.presets.EMBEDDING_PARAMS["default"]).fit(sc, seed=0)
print(res.coords.shape)
"""


def test_copied_package_folder_works_without_installation(tmp_path):
    shutil.copytree(ROOT / "hohypemb", tmp_path / "hohypemb")
    out = subprocess.run([sys.executable, "-c", SCRIPT], cwd=tmp_path, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    imported_from, shape = out.stdout.splitlines()
    assert Path(imported_from).resolve().is_relative_to(tmp_path.resolve()), imported_from
    assert shape == "(6, 2)"


def test_copied_package_folder_carries_the_license():
    assert (ROOT / "hohypemb" / "LICENSE").read_text() == (ROOT / "LICENSE").read_text()
