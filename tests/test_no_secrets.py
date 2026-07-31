import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECRET = re.compile(r"sk-(?:proj-)?[A-Za-z0-9_-]{20,}")
SECRET_PATTERNS = [
    SECRET,
    re.compile(r"(?:ghp|github_pat|hf)_[A-Za-z0-9_-]{20,}"),
]


def test_repository_contains_no_openai_keys():
    findings = []
    names = subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    for name in names:
        path = ROOT / name
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(pattern.search(text) for pattern in SECRET_PATTERNS):
            findings.append(name)
    assert findings == []
