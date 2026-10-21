import gzip
import warnings

import pytest

from hohypemb.datasets import cora
from hohypemb.datasets.reference import REFERENCE_SHA256

AHORN = (
    '{"name": "cora-coauthorship", "format-version": "0.3", "revision": 2}\n'
    '1 {"category": "Theory", "name": "A"}\n'
    '2 {"category": "Theory", "name": "B"}\n'
    '3 {"category": "Neural_Networks", "name": "C"}\n'
    "1,2,3 {}\n"
    "1,2 {}\n"
)


def test_downloads_and_parses_when_no_path(tmp_path, monkeypatch):
    gz = tmp_path / "remote.txt.gz"
    with gzip.open(gz, "wt", encoding="utf-8") as f:
        f.write(AHORN)
    monkeypatch.setattr(cora, "CORA_URL", gz.as_uri())
    with pytest.warns(UserWarning, match="differs"):
        sc = cora.load_cora_coauthorship(data_dir=tmp_path / "cache")
    assert sc.n_nodes == 3 and sc.simplices == [(0, 1), (0, 1, 2)]
    assert (tmp_path / "cache" / "cora-coauthorship.txt").exists()


def test_corrupt_cached_archive_raises_clear_error_and_leaves_no_partial_file(tmp_path):
    (tmp_path / "cora-coauthorship.txt.gz").write_bytes(b"not actually gzip")
    with pytest.raises(RuntimeError, match="cora-coauthorship.txt.gz"):
        cora.cora_path(tmp_path)
    assert not (tmp_path / "cora-coauthorship.txt.part").exists()
    assert not (tmp_path / "cora-coauthorship.txt").exists()


@pytest.mark.network
def test_zenodo_revision_2_matches_the_paper_file(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        path = cora.cora_path(tmp_path)
    from hohypemb.datasets._download import sha256_of
    assert sha256_of(path) == REFERENCE_SHA256["cora-coauthorship.txt"]
