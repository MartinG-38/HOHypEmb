import hohypemb


def test_version():
    assert hohypemb.__version__ == "0.1.0"


def test_datasets_namespace_exposes_loaders_and_generators():
    import hohypemb as hh
    for name in ("planted_surfaces", "identical_skeleton_pair", "planted_hypergraph_sbm",
                 "load_cora_coauthorship", "load_ecoli"):
        assert callable(getattr(hh.datasets, name)), name
    assert hh.datasets.synthetic.planted_surfaces is hh.datasets.planted_surfaces
    assert hh.datasets.ecoli.load_ecoli is hh.datasets.load_ecoli
    assert hh.datasets.cora.load_cora_coauthorship is hh.datasets.load_cora_coauthorship


def test_importing_the_package_and_datasets_pulls_no_matplotlib_and_no_network():
    import subprocess
    import sys
    import textwrap
    script = textwrap.dedent("""
        import sys, urllib.request
        def refuse(*args, **kwargs):
            raise AssertionError("network access at import time")
        urllib.request.urlopen = refuse
        urllib.request.OpenerDirector.open = refuse
        import hohypemb
        assert "matplotlib" not in sys.modules, "import hohypemb imported matplotlib"
        hohypemb.datasets.load_ecoli
        import hohypemb.datasets.synthetic, hohypemb.datasets.cora, hohypemb.datasets.ecoli
        assert "matplotlib" not in sys.modules, "hohypemb.datasets imported matplotlib"
    """)
    subprocess.run([sys.executable, "-c", script], check=True)
