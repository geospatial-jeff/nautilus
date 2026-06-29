"""The runtime data path: channels, mailboxes, partitioners and the operator-instance actors.

Nothing in this package may import :mod:`nautilus.cluster` (enforced by an import-linter contract):
the control plane is bootstrap-only and never sits on the data path.
"""
