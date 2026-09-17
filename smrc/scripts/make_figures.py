"""Generate all report figures from results/results.csv."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.plots import make_all
if __name__ == "__main__":
    make_all()
