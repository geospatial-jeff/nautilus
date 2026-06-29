# The Stage 4 multi-node image. One image runs every role — a worker daemon or the coordinator — because
# a daemon cloudpickle.loads the plan and any operator pickled by reference must be importable wherever it
# lands, so every container needs the same installed package. pyarrow/numpy/psutil ship manylinux wheels,
# so the install needs no compiler on the slim base.
FROM python:3.12-slim

WORKDIR /app
# Only what the wheel build (hatchling) reads: the metadata, the README it references, and the package
# (which carries the bundled dashboard.html artifact).
COPY pyproject.toml README.md ./
COPY src ./src
# The `system` extra pulls in psutil so each worker reports its container's CPU / memory / fd usage — the
# per-host resource telemetry a multi-node run needs to tell which container is hot or running out of room.
RUN pip install --no-cache-dir ".[system]"

# `nautilus worker ...` for a daemon, `nautilus run ...` for the coordinator.
ENTRYPOINT ["nautilus"]
