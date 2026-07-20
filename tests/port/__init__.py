"""Characterization / golden tests that pin nautilus's observable behavior at its stable interfaces,
so a future Python->Rust rewrite that passes this suite is provably faithful. A package (not flat files)
so module names stay distinct from the top-level suite (e.g. ``port.test_dsl`` vs ``test_dsl``)."""
