"""
_bootstrap.py
-------------
Import this FIRST in every runnable script:

    import _bootstrap  # noqa: F401

It adds this project's own folder to Python's import path, so that
`from model import ...` works no matter which directory you launch from.
This is what prevents the classic "ModuleNotFoundError: No module named
'model'" that happens when you run the script from a different folder or
from an IDE with a mismatched working directory.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
