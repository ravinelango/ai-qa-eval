"""Fail if the built wheel does not contain every declared top-level package."""

from __future__ import annotations

import sys
import zipfile

EXPECTED = ("apps/__init__.py", "evals/__init__.py", "redteam/__init__.py", "dashboard/__init__.py")


def main(path: str) -> int:
    names = set(zipfile.ZipFile(path).namelist())
    missing = [name for name in EXPECTED if name not in names]
    if missing:
        print(f"{path}: missing {missing}", file=sys.stderr)
        return 1
    print(f"{path}: contains {', '.join(EXPECTED)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
