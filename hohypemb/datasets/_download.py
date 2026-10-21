"""Cached, checksummed downloads of public data files."""
import hashlib
import http.client
import os
import shutil
import time
import urllib.error
import urllib.request
import warnings
from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, Path]
USER_AGENT = "hohypemb (https://github.com/MartinG-38/HOHypEmb)"


def resolve_data_dir(data_dir: Optional[PathLike] = None) -> Path:
    """``data_dir``, else ``$HOHYPEMB_DATA``, else ``~/.cache/hohypemb`` (created if needed)."""
    chosen = data_dir or os.environ.get("HOHYPEMB_DATA") or Path.home() / ".cache" / "hohypemb"
    path = Path(chosen).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_of(path: PathLike) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def warn_if_changed(path: Path, sha256: Optional[str]) -> None:
    if sha256 is not None and sha256_of(path) != sha256:
        warnings.warn(f"{path.name}: data differs from the version used in the HOHypEmb paper "
                      "(SHA-256 mismatch); results may differ from the published ones.",
                      UserWarning, stacklevel=3)


def fetch(url: str, filename: str, sha256: Optional[str] = None, data_dir: Optional[PathLike] = None,
          attempts: int = 3, pause: float = 5.0) -> Path:
    """Return the local path of ``filename``, downloading it from ``url`` if it is not cached.

    The download goes to ``<filename>.part`` and is renamed only when complete. After
    ``attempts`` failures a ``RuntimeError`` explains where to place the file by hand.
    """
    target = resolve_data_dir(data_dir) / filename
    if not target.exists():
        part = target.with_name(target.name + ".part")
        last_error: Optional[BaseException] = None
        for attempt in range(1, attempts + 1):
            try:
                request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(request, timeout=300) as response, open(part, "wb") as out:
                    shutil.copyfileobj(response, out)
                os.replace(part, target)
                break
            except (OSError, urllib.error.URLError, http.client.HTTPException) as exc:
                last_error = exc
                if part.exists():
                    part.unlink()
                if attempt < attempts:
                    time.sleep(pause)
        else:
            raise RuntimeError(
                f"Could not download {url} after {attempts} attempts ({last_error}). Download it "
                f"manually and place it as {target}, or pass data_dir= a folder that contains "
                f"{filename}.") from last_error
    warn_if_changed(target, sha256)
    return target
