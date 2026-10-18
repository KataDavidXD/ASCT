"""Verify the exact public allowlist without training dependencies."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    manifest = json.loads((ROOT / "MANIFEST.json").read_text(encoding="utf-8"))
    rows = manifest["files"]
    paths = [r["path"] for r in rows]
    assert len(paths) == len(set(paths)), "Duplicate publication paths"
    assert manifest["paper"] == "https://arxiv.org/abs/2609.35215v1"
    forbidden_secrets = re.compile(rb"(?:sk-(?:proj-)?[A-Za-z0-9_-]{24,}|gh[pousr]_[A-Za-z0-9]{30,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----)")
    for row in rows:
        path = (ROOT / row["path"]).resolve()
        assert path.is_relative_to(ROOT), row["path"]
        data = path.read_bytes()
        assert len(data) == row["bytes"], f"Size changed: {row['path']}"
        assert hashlib.sha256(data).hexdigest() == row["sha256"], f"Hash changed: {row['path']}"
        assert not forbidden_secrets.search(data), f"Credential-shaped content: {row['path']}"
        assert path.name not in {".env", "checkpoint.pt"}
        if path.suffix == ".py":
            ast.parse(data.decode("utf-8-sig"), filename=row["path"])

    # Avoid resolving an enclosing workspace repository before this release is initialized.
    if (ROOT / ".git").exists():
        tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
        tracked = {p for p in tracked if p}
        expected = set(paths) | {"MANIFEST.json"}
        assert tracked == expected, f"Tracked publication paths differ: extra={sorted(tracked - expected)}, missing={sorted(expected - tracked)}"

    splits = json.loads((ROOT / "new_controls/actor_cf/metadata/splits.json").read_text())
    expected_sizes = {"train": 2000, "valid": 400, "test": 400}
    for split, size in expected_sizes.items():
        assert len(splits[split]) == len(set(splits[split])) == size
    assert len(set().union(*(set(splits[s]) for s in expected_sizes))) == 2800
    print(f"Public release PASS: {len(paths) + 1} files; exact hashes; paper QIDs 2000/400/400.")


if __name__ == "__main__":
    main()
