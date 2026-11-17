#!/usr/bin/env python3
"""Link checker for local HTML report pages.

Validates that every href/src target in a report resolves to an existing local
file (relative to the page's own directory — the exact way file:// resolves it).
External URLs (http/https) are skipped; anchors (#...) are skipped.

Usage:
  .venv/bin/python scripts/check_report_links.py artifacts/2026-11-16-GATE-NIGHT-REPORT.html
Exit 0 = all local links resolve; exit 1 = broken links listed.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

ATTR_RE = re.compile(r'(?:href|src)="([^"]+)"')


def check(page: Path) -> list[str]:
    html = page.read_text(encoding="utf-8")
    broken = []
    for raw in ATTR_RE.findall(html):
        target = raw.split("#")[0].strip()
        if not target or target.startswith(("http://", "https://", "mailto:", "data:")):
            continue
        resolved = (page.parent / unquote(urlparse(target).path)).resolve()
        if not resolved.exists():
            broken.append(f"{raw}  ->  {resolved}")
    return broken


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    failures = 0
    for arg in sys.argv[1:]:
        page = Path(arg).resolve()
        if not page.is_file():
            print(f"FAIL page not found: {page}")
            failures += 1
            continue
        broken = check(page)
        if broken:
            failures += 1
            print(f"FAIL {page}:")
            for b in broken:
                print(f"  broken: {b}")
        else:
            links = len([t for t in ATTR_RE.findall(page.read_text(encoding='utf-8'))
                         if not t.startswith(("http://", "https://", "mailto:", "data:")) and t and not t.startswith("#")])
            print(f"PASS {page.name}: {links} local links all resolve")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
