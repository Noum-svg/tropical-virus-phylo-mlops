"""Pytest configuration.

Ensure the project root is on ``sys.path`` so tests can import the ``src``
package as ``src.<module>`` regardless of the working directory pytest is
launched from.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
