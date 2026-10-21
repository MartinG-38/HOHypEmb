"""The example notebooks: in sync with their builder, executed, public API only, and they run."""
import ast
import importlib
import re
import sys
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

from tests._data import SNAPSHOT_DIR

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
COMMITTED = sorted(p.name for p in EXAMPLES.glob("*.ipynb"))
FAST = ["01_quickstart_synthetic.ipynb", "04_synthetic_markers.ipynb"]
NEEDS_DATA = ["02_cora.ipynb", "03_ecoli.ipynb"]


def _builder():
    """``NOTEBOOKS`` of examples/build_examples.py (the examples folder is not a package)."""
    sys.path.insert(0, str(EXAMPLES))
    try:
        return importlib.import_module("build_examples").NOTEBOOKS
    finally:
        sys.path.remove(str(EXAMPLES))


def _cells(name):
    return nbformat.read(EXAMPLES / name, as_version=4).cells


def _code_cells(name):
    return [c for c in _cells(name) if c.cell_type == "code"]


def _run(name):
    nb = nbformat.read(EXAMPLES / name, as_version=4)
    NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(EXAMPLES)}}).execute()
    errors = [o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.output_type == "error"]
    assert not errors, errors


def test_builder_lists_every_committed_notebook():
    assert sorted(_builder()) == COMMITTED


@pytest.mark.parametrize("name", COMMITTED)
def test_committed_notebook_matches_builder(name):
    built = _builder()[name]
    assert [(c.cell_type, c.source) for c in _cells(name)] == [(c.cell_type, c.source) for c in built], (
        f"{name} differs from examples/_notebooks; run `python examples/build_examples.py --execute --only {name}`")


@pytest.mark.parametrize("name", COMMITTED)
def test_committed_notebook_is_executed(name):
    counts = [c.get("execution_count") for c in _code_cells(name)]
    assert counts == list(range(1, len(counts) + 1)), (
        f"{name}: execution counts {counts} are not 1..n in order; build it with --execute")


@pytest.mark.parametrize("name", COMMITTED)
def test_examples_use_only_the_public_api(name):
    for cell in _code_cells(name):
        assert not re.search(r"^\s*(from|import)\s+paper\b", cell.source, re.M), cell.source
        assert not re.search(r"\b(hohypemb|hh)\._", cell.source), cell.source


@pytest.mark.parametrize("name", COMMITTED)
def test_cells_parse_as_python_39(name):
    for cell in _code_cells(name):
        ast.parse(cell.source, feature_version=(3, 9))


@pytest.mark.parametrize("name", FAST)
def test_fast_examples_run(name):
    _run(name)


@pytest.mark.slow
@pytest.mark.parametrize("name", NEEDS_DATA)
def test_dataset_examples_run(name):
    if SNAPSHOT_DIR is None:
        pytest.skip("needs HOHYPEMB_SNAPSHOT_DIR (or network access)")
    _run(name)


BRITISH = (r"\bcolour", r"\bnormalis", r"\bvisualis", r"\bbehaviour", r"\banalys(e|ed|es|ing)\b",
           r"\bcentre", r"\bneighbour", r"\blabelled", r"\bmodelling", r"\bfavour", r"\borganis(e|ed|es|ing|ation)",
           r"\boptimis", r"\brecognis", r"\bminimis", r"\bmaximis", r"\bartefact", r"\blicence", r"\bgrey\b")


@pytest.mark.parametrize("name", COMMITTED)
def test_notebooks_use_american_spelling(name):
    text = "\n".join(c.source for c in _cells(name)).lower()
    found = sorted({pattern for pattern in BRITISH if re.search(pattern, text)})
    assert not found, f"{name}: British spellings {found}; the examples are in American English"
