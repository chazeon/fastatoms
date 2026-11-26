"""Pytest configuration for the fastatoms parity suite.

Keeps bytecode and caches out of the source tree, and makes the helper module
in this directory importable from the test files.
"""

import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
