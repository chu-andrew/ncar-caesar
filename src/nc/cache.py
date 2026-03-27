from joblib import Memory
from pathlib import Path

from nc.loader import PROJECT_ROOT

MEMORY = Memory(location=Path(PROJECT_ROOT, "cache"), verbose=0)
