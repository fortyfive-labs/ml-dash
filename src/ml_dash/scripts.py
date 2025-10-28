#!/usr/bin/env python3
"""Development scripts for building and previewing documentation."""
import subprocess
import sys
from pathlib import Path


def get_docs_dir():
    """Get the absolute path to the docs directory."""
    package_dir = Path(__file__).parent.parent.parent
    return package_dir / "docs"


def docs():
    """Build the documentation using Sphinx."""
    docs_dir = get_docs_dir()
    build_dir = docs_dir / "_build" / "html"

    print(f"Building documentation from {docs_dir}...")
    result = subprocess.run(
        ["sphinx-build", "-b", "html", str(docs_dir), str(build_dir)],
        check=False
    )

    if result.returncode == 0:
        print(f"\nDocumentation built successfully!")
        print(f"Open {build_dir / 'index.html'} to view the docs.")

    sys.exit(result.returncode)


def preview():
    """Build and preview the documentation with live reload."""
    docs_dir = get_docs_dir()
    build_dir = docs_dir / "_build" / "html"

    print(f"Starting documentation preview server from {docs_dir}...")
    print("Documentation will open in your browser automatically.")
    print("Press Ctrl+C to stop the server.\n")

    result = subprocess.run(
        [
            "sphinx-autobuild",
            str(docs_dir),
            str(build_dir),
            "--open-browser"
        ],
        check=False
    )

    sys.exit(result.returncode)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "preview":
        preview()
    else:
        docs()
