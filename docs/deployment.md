# Deployment

`run(workers=4)` spreads a pipeline across processes on **one** machine. To run it across **many** machines
you deploy two roles yourself — a set of long-lived **worker daemons** and a **coordinator** that dials
them — and run the same graph across them. Both roles are the one image the repo `Dockerfile` builds, and
the compiled plan is a serializable value, so the graph you ran locally runs unchanged on the cluster (see
[Concepts](design.md)).

## The two roles

- **`nautilus worker`** — a long-lived daemon, one per container, listening on a control port. When a job
  arrives it runs its slice of the plan and shuffles data directly to its peer daemons (Arrow-IPC over TCP).
- **`nautilus run --daemons <roster>`** — the coordinator. It compiles your graph, hands each daemon a
  serialized slice of the plan to run, and folds the telemetry they return into one report.

Every daemon authenticates with a shared secret (`NAUTILUS_CLUSTER_SECRET`) and refuses to bind a
non-loopback interface without one; add the TLS env vars to encrypt the wire on an untrusted network. Flags
for both commands are in the [CLI reference](cli-reference.md).

## On Kubernetes

`deploy/kubernetes/` is a kustomize setup for exactly this shape: a headless **Service** and a
**StatefulSet** of daemons, so each pod gets a stable DNS name the coordinator can list in `--daemons`
before the pods start. Only the control port needs the Service — the shuffle's data edges use an ephemeral
pod-to-pod port each daemon advertises by its pod IP.

Placement is a scheduler decision, not a code change: `placement.py` maps operator instances to daemons
round-robin and knows nothing about physical nodes, so the overlays choose the layout — `cross-node` (one
daemon per host, shuffle over the network) or `intra-node` (all on one host, shuffle over loopback).

You supply the image (point the kustomization's `images:` at your registry) and the cluster secret, then
apply an overlay:

```bash
kubectl apply -k deploy/kubernetes/overlays/cross-node
```

The step-by-step — building the image, creating the secret, running a benchmark across the daemons, tuning,
and security — is in
[`deploy/kubernetes/README.md`](https://github.com/geospatial-jeff/nautilus/blob/main/deploy/kubernetes/README.md).
For what crossing the network costs, see [Performance](performance.md#distributed-performance).
