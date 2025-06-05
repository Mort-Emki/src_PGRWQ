"""Compatibility wrapper to maintain the old package name ``PGRWQI``."""

import importlib
import sys
from pathlib import Path

# Path of the directory containing the real package
package_dir = Path(__file__).resolve().parent.parent
# Add its parent directory to sys.path so the package can be imported
parent_dir = package_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Import the actual package and register it under the old name
module = importlib.import_module(package_dir.name)
sys.modules[__name__] = module

