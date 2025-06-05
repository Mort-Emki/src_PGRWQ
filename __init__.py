"""PG-RWQ package initialization.

Provides backward compatibility so modules can be imported using the
original package name ``PGRWQI``.
"""

import sys

# Alias this package as ``PGRWQI`` if it hasn't been defined already
sys.modules.setdefault("PGRWQI", sys.modules[__name__])
