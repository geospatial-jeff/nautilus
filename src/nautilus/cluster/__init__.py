"""The control plane: it brings a cluster up and tears it down around a job, but does no per-record work.

Everything here runs during the one-time **control phase** (compile + deploy + bootstrap) or at job
boundaries — never on the per-record data path. The "no central scheduler" guarantee is enforced
mechanically by an import-linter contract: no data-path package (``nautilus.runtime``, ``core``,
``transport``, ``telemetry``, ``api``, ``compile``) may import ``nautilus.cluster``.

:func:`~nautilus.cluster.coordinator.deploy` is the entry point: it runs a :class:`LogicalGraph` across
spawned worker processes and returns a :class:`RunResult`.
"""

from nautilus.cluster.coordinator import WorkerCrashed, WorkerError, deploy

__all__ = ["deploy", "WorkerError", "WorkerCrashed"]
