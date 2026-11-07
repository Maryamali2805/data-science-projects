import sys
from pathlib import Path

# Allow `src.*` imports from the project root when running pytest
sys.path.insert(0, str(Path(__file__).parent))
