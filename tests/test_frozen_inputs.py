import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_frozen_input_hashes_match_manifest():
    manifest = ROOT / "provenance" / "frozen_input_sha256.txt"
    rows = [line.split("  ", 1) for line in manifest.read_text().splitlines()]
    assert rows
    for expected, relative in rows:
        path = ROOT / relative
        assert path.is_file(), relative
        assert sha256(path) == expected, relative
