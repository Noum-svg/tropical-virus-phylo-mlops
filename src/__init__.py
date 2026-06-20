"""Scientific core for tropical-virus-phylo-mlops.

All mathematical and business logic lives here as pure, typed, documented
functions. The delivery layers (``api/`` and ``app/``) orchestrate these
functions only -- they must never duplicate the math.
"""

__all__ = [
    "utils",
    "data_loader",
    "distances",
]
