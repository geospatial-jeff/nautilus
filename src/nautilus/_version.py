"""The package version, in a leaf module of its own.

This module imports nothing from ``nautilus``, so a boundary module like :mod:`nautilus.driver.meta` can
read the version via ``from nautilus._version import __version__`` without the import graph recording an
edge into ``nautilus``'s top-level ``__init__``. That ``__init__`` imports the DSL, which through its lazy
``deploy`` terminal reaches ``nautilus.cluster`` — so a ``from nautilus import __version__`` in a data-path
module would add ``… -> nautilus -> dsl -> cluster`` and break the "data path must not import the control
plane" import-linter contract. Reading the version from this leaf keeps that contract held.
"""

__version__ = "0.0.1"
