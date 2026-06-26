"""The control plane: placement, launcher, membership, rendezvous and the startup barrier.

Everything here runs during the one-time **control phase** (compile + deploy + bootstrap) or at job
boundaries — never on the per-record data path. The "no central scheduler" guarantee is enforced
mechanically by an import-linter contract: data-path packages (``nautilus.runtime``,
``nautilus.core``, ``nautilus.transport``) may not import ``nautilus.cluster``.

Fleshed out in Stage 2 (deterministic placement, two-phase bootstrap launcher, static membership,
decentralized startup barrier).
"""
