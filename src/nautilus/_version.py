"""The package version, in a leaf module of its own.

Kept separate from ``nautilus/__init__.py`` so a module can read the version without importing the whole
top-level package (which pulls in the DSL, and through its lazy ``deploy`` terminal the control plane).
:mod:`nautilus.driver.meta` reads it from here, so the data path never reaches ``nautilus.cluster``.
"""

__version__ = "0.0.1"
