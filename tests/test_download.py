import http.client
import io
import urllib.error

import pytest

from hohypemb.datasets import _download


def test_fetch_downloads_once_and_caches(tmp_path):
    src = tmp_path / "remote.txt"
    src.write_text("payload")
    cache = tmp_path / "cache"
    path = _download.fetch(src.as_uri(), "local.txt", data_dir=cache)
    assert path.read_text() == "payload"
    src.write_text("changed")
    assert _download.fetch(src.as_uri(), "local.txt", data_dir=cache).read_text() == "payload"


def test_fetch_warns_on_checksum_mismatch(tmp_path):
    src = tmp_path / "remote.txt"
    src.write_text("payload")
    with pytest.warns(UserWarning, match="differs from the version used in the HOHypEmb paper"):
        _download.fetch(src.as_uri(), "x.txt", sha256="0" * 64, data_dir=tmp_path / "c")


def test_fetch_failure_explains_what_to_do(tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(_download.urllib.request, "urlopen", broken)
    with pytest.raises(RuntimeError, match=r"place it as .*missing\.txt.*data_dir"):
        _download.fetch("https://example.invalid/f", "missing.txt", data_dir=tmp_path, pause=0)
    assert not (tmp_path / "missing.txt.part").exists()


def test_data_dir_resolution(tmp_path, monkeypatch):
    monkeypatch.setenv("HOHYPEMB_DATA", str(tmp_path / "env"))
    assert _download.resolve_data_dir() == tmp_path / "env"
    assert _download.resolve_data_dir(tmp_path / "explicit") == tmp_path / "explicit"


class TruncatedResponse:
    """A response whose body breaks off mid-stream, as a dropped connection does."""

    def __init__(self, head: bytes = b"partial body"):
        self.body = io.BytesIO(head)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, size=-1):
        chunk = self.body.read(size)
        if not chunk:
            raise http.client.IncompleteRead(b"")
        return chunk


def test_fetch_retries_a_truncated_body_and_cleans_up(tmp_path, monkeypatch):
    calls, sleeps = [], []

    def truncated(*args, **kwargs):
        calls.append(1)
        return TruncatedResponse()

    monkeypatch.setattr(_download.urllib.request, "urlopen", truncated)
    monkeypatch.setattr(_download.time, "sleep", sleeps.append)
    with pytest.raises(RuntimeError, match="after 3 attempts"):
        _download.fetch("https://example.invalid/f", "cut.txt", data_dir=tmp_path, pause=7.0)
    assert len(calls) == 3 and sleeps == [7.0, 7.0]
    assert not (tmp_path / "cut.txt.part").exists()
    assert not (tmp_path / "cut.txt").exists()
