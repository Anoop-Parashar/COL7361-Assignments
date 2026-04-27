"""
verify_requirements.py
----------------------
Compares packages listed in a requirements.txt file against
what is actually installed in the current Python environment.

Usage:
    python verify_requirements.py                         # uses requirements.txt in cwd
    python verify_requirements.py path/to/requirements.txt
"""

import sys
import re
import importlib.metadata as meta
from pathlib import Path


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_requirements(filepath: Path) -> list[dict]:
    """
    Parse a requirements.txt file and return a list of
    {"name": str, "specifier": str | None, "raw": str} dicts.
    Skips blank lines, comments, and unsupported directives (-r, -c, etc.).
    """
    entries = []
    # Matches:  package_name[extras]>=1.2,<2  (all version specifiers)
    pattern = re.compile(
        r"^(?P<name>[A-Za-z0-9_\-\.]+)"
        r"(?:\[.*?\])?"                        # optional extras
        r"(?P<specifier>[^;#\s]*)?"            # optional version specifier(s)
    )

    for raw_line in filepath.read_text().splitlines():
        line = raw_line.strip()
        # Skip blanks, comments, and pip options/flags
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        # Strip inline comments
        line = line.split("#")[0].strip()

        m = pattern.match(line)
        if m:
            entries.append({
                "name":      m.group("name"),
                "specifier": (m.group("specifier") or "").strip(),
                "raw":       raw_line.strip(),
            })
    return entries


def normalize(name: str) -> str:
    """Normalize package names per PEP 503 (case + punctuation)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def installed_version(name: str) -> str | None:
    """Return the installed version string, or None if not installed."""
    try:
        return meta.version(name)
    except meta.PackageNotFoundError:
        return None


def check_specifier(installed_ver: str, specifier_str: str) -> bool:
    """
    Return True if installed_ver satisfies the specifier string.
    Uses packaging.specifiers if available, falls back to simple ==.
    """
    if not specifier_str:
        return True  # no version constraint → any version is fine

    try:
        from packaging.specifiers import SpecifierSet
        return installed_ver in SpecifierSet(specifier_str, prereleases=True)
    except ImportError:
        # Fallback: only handle the common == case
        if specifier_str.startswith("=="):
            return installed_ver == specifier_str[2:].strip()
        # Can't evaluate other specifiers without packaging; treat as pass
        return True


# ── Main ──────────────────────────────────────────────────────────────────────

def verify(req_file: Path) -> None:
    if not req_file.exists():
        print(f"[ERROR] File not found: {req_file}")
        sys.exit(1)

    entries = parse_requirements(req_file)
    if not entries:
        print("No packages found in requirements file.")
        return

    # Column widths
    W_PKG  = max(len(e["name"])      for e in entries) + 2
    W_REQ  = max(len(e["specifier"]) for e in entries) + 2
    W_INST = 15

    header = (
        f"{'Package':<{W_PKG}} "
        f"{'Required':<{W_REQ}} "
        f"{'Installed':<{W_INST}} "
        f"Status"
    )
    separator = "-" * len(header)

    ok_count      = 0
    missing       = []
    mismatch      = []
    not_specified = []

    print(f"\nVerifying: {req_file.resolve()}\n")
    print(header)
    print(separator)

    for entry in entries:
        name      = entry["name"]
        specifier = entry["specifier"]
        inst_ver  = installed_version(name)

        if inst_ver is None:
            status = "❌  MISSING"
            missing.append(name)
        elif not specifier:
            status = "✅  OK (any version)"
            not_specified.append(name)
            ok_count += 1
        elif check_specifier(inst_ver, specifier):
            status = "✅  OK"
            ok_count += 1
        else:
            status = f"⚠️   VERSION MISMATCH"
            mismatch.append((name, specifier, inst_ver))

        print(
            f"{name:<{W_PKG}} "
            f"{specifier or 'any':<{W_REQ}} "
            f"{inst_ver or 'not installed':<{W_INST}} "
            f"{status}"
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    print(separator)
    total = len(entries)
    print(f"\nSummary: {ok_count}/{total} packages OK")

    if missing:
        print(f"\n[MISSING — {len(missing)}]")
        for pkg in missing:
            print(f"  • {pkg}")
        print("\n  Install with:")
        print(f"  pip install {' '.join(missing)}")

    if mismatch:
        print(f"\n[VERSION MISMATCH — {len(mismatch)}]")
        for pkg, req_spec, inst_ver in mismatch:
            print(f"  • {pkg}: requires {req_spec}, found {inst_ver}")
        print("\n  Fix with:")
        print(f"  pip install " + " ".join(
            f'"{p}{s}"' for p, s, _ in mismatch
        ))

    if not missing and not mismatch:
        print("\n✅  All packages match requirements.txt!")
    else:
        sys.exit(1)  # non-zero exit for CI/CD pipelines


if __name__ == "__main__":
    req_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("requirements.txt")
    verify(req_path)
