"""Print rankings by macro-F1 and efficiency index, and write the Pareto frontier."""
import _common  # noqa: F401  (must come first)

from src.analysis import main

if __name__ == "__main__":
    _common.parse_args(__doc__)
    main()
