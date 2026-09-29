"""Record the Python and package versions used for a run."""
import _common  # noqa: F401  (must come first)

import platform
import sys
from importlib.metadata import PackageNotFoundError, version

from src import config

PACKAGES = ["scikit-learn", "numpy", "scipy", "pandas", "matplotlib", "psutil", "datasets"]


def main():
    _common.parse_args(__doc__)
    lines = [f"python {sys.version.split()[0]}", f"platform {platform.platform()}"]
    for name in PACKAGES:
        try:
            lines.append(f"{name} {version(name)}")
        except PackageNotFoundError:
            lines.append(f"{name} not installed")

    text = "\n".join(lines)
    print(text)
    config.ENV_FILE.write_text(text + "\n")
    print(f"\nWritten to {config.ENV_FILE}")
    print("Copy these versions into requirements.txt to pin the environment.")


if __name__ == "__main__":
    main()
