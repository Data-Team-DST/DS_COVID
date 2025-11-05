import sys
from pathlib import Path

# Ajouter src/ au sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
