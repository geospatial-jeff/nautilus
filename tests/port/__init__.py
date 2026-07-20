"""Characterization tests that pin nautilus's observable behavior at its stable interfaces, so a future
Python->Rust rewrite that passes this suite is provably faithful. Two kinds sit side by side: golden /
example tests fix the output for hand-picked inputs (``test_*.py``), and ``test_prop_*.py`` property
tests assert the *invariant* over a whole space of fixed-seed random inputs — the contract a port must
hold everywhere, which a golden point can miss. A package (not flat files) so module names stay distinct
from the top-level suite (e.g. ``port.test_dsl`` vs ``test_dsl``)."""
