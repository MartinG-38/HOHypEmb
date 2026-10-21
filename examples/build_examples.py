"""Write (and optionally execute) the example notebooks.

    python examples/build_examples.py                                  # write them, unexecuted
    python examples/build_examples.py --execute                        # write and execute them
    python examples/build_examples.py --execute --only 02_cora.ipynb   # one notebook

The cells live in ``examples/_notebooks/``, one module per notebook. Executing the dataset notebooks
needs network access or ``HOHYPEMB_SNAPSHOT_DIR``.
"""
import argparse
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_notebook

from _notebooks import cora, ecoli, quickstart, synthetic_markers

HERE = Path(__file__).resolve().parent
NOTEBOOKS = {
    "01_quickstart_synthetic.ipynb": quickstart.CELLS,
    "02_cora.ipynb": cora.CELLS,
    "03_ecoli.ipynb": ecoli.CELLS,
    "04_synthetic_markers.ipynb": synthetic_markers.CELLS,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--only", nargs="+", choices=sorted(NOTEBOOKS), help="notebook file names to build")
    args = parser.parse_args()
    if args.execute:
        # The kernel subprocess inherits this process's environment. MPLBACKEND=Agg (set by the
        # project's shell setup so headless test/CI runs never pop up a GUI window) would force
        # matplotlib's non-interactive Agg backend inside the notebook kernel too, making
        # plt.show() a no-op and dropping the committed figures. Drop it here so ipykernel's
        # normal inline-backend autodetection runs and PNGs are captured as usual.
        os.environ.pop("MPLBACKEND", None)
    for name in args.only or NOTEBOOKS:
        nb = new_notebook(cells=NOTEBOOKS[name], metadata={"kernelspec": {"name": "python3", "display_name": "Python 3",
                                                                          "language": "python"}})
        if args.execute:
            NotebookClient(nb, timeout=3600, kernel_name="python3",
                           resources={"metadata": {"path": str(HERE)}}).execute()
        nbformat.write(nb, HERE / name)
        print("wrote", name)


if __name__ == "__main__":
    main()
