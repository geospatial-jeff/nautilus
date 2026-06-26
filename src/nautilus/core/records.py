"""The ``Frame`` model: the sealed set of frame types that flow on a channel.

Every edge in a Nautilus dataflow carries two kinds of frame:

* **data** frames — only :class:`Batch` (an Arrow ``RecordBatch``), and
* **control** frames — :class:`Watermark`, :class:`EOS`, :class:`StatusIdle`,
  :class:`StatusActive`, and (reserved) :class:`Barrier`.

Data frames are routed by the edge's partitioner; control frames are always *broadcast* to every
downstream instance. The class-level :attr:`Frame.is_control` flag distinguishes them.

Event time is represented as an integer number of **microseconds** since the Unix epoch. This is a
deliberate, stable choice (it survives serialization and a future multi-node / cross-language wire
format without float rounding). Real event times must be strictly less than
:data:`MAX_LEGAL_EVENT_TIME` so a genuine timestamp can never collide with the
:data:`WATERMARK_MAX` "stream complete" sentinel.
"""

from __future__ import annotations

import dataclasses
from typing import ClassVar, Final, TypeGuard

import pyarrow as pa

# --- Event-time domain (integer microseconds since the Unix epoch) -----------------------------

#: Smallest representable watermark; a fresh stream starts here ("nothing seen yet").
WATERMARK_MIN: Final[int] = -(2**62)
#: Watermark value meaning "stream complete" — advanced past every possible real event time.
WATERMARK_MAX: Final[int] = 2**62
#: Real event times must be strictly below this so they never collide with ``WATERMARK_MAX``.
MAX_LEGAL_EVENT_TIME: Final[int] = WATERMARK_MAX - 1


def check_event_time(t: int) -> int:
    """Validate a real event time, returning it unchanged. Raises on the reserved sentinel range."""
    if not (WATERMARK_MIN <= t <= MAX_LEGAL_EVENT_TIME):
        raise ValueError(
            f"event time {t} out of legal range "
            f"[{WATERMARK_MIN}, {MAX_LEGAL_EVENT_TIME}] (microseconds since epoch)"
        )
    return t


# --- The sealed Frame union --------------------------------------------------------------------


class Frame:
    """Base class for everything that flows on a channel.

    Sealed: the only subclasses are :class:`Batch`, :class:`Watermark`, :class:`EOS`,
    :class:`StatusIdle`, :class:`StatusActive`, and :class:`Barrier`. Use :func:`is_data` /
    :func:`is_control` (or the :attr:`is_control` class flag) to discriminate; the concrete
    subclasses support structural pattern matching.
    """

    __slots__ = ()
    #: Overridden by every concrete subclass. True for control frames.
    is_control: ClassVar[bool]


@dataclasses.dataclass(frozen=True, slots=True)
class Batch(Frame):
    """A micro-batch of data records, carried as an Arrow ``RecordBatch``.

    This is the only data frame. Keying, windowing and partitioning operate on *columns* of
    ``data`` (Arrow-first), so the batch itself carries no separate key — the operator and the
    edge's partitioner know which columns form the key.
    """

    data: pa.RecordBatch
    is_control: ClassVar[bool] = False

    @property
    def num_rows(self) -> int:
        return int(self.data.num_rows)


@dataclasses.dataclass(frozen=True, slots=True)
class Watermark(Frame):
    """Event-time watermark: a promise that no future record on this channel has event time < ``t``.

    Watermarks are monotonically non-decreasing on any single channel and are broadcast to all
    downstream instances.
    """

    t: int
    is_control: ClassVar[bool] = True


@dataclasses.dataclass(frozen=True, slots=True)
class EOS(Frame):
    """End of stream. The sole terminal frame.

    Semantics: receiving ``EOS`` on a channel advances that channel's watermark to
    :data:`WATERMARK_MAX`. An operator forwards ``EOS`` downstream only after it has received
    ``EOS`` on *all* of its inputs and flushed all pending state.
    """

    is_control: ClassVar[bool] = True


@dataclasses.dataclass(frozen=True, slots=True)
class StatusIdle(Frame):
    """Marks a channel as temporarily idle so it is excluded from the watermark minimum.

    Without this, a silent input would pin the combined watermark at ``WATERMARK_MIN`` forever and
    stop all event-time progress downstream.
    """

    is_control: ClassVar[bool] = True


@dataclasses.dataclass(frozen=True, slots=True)
class StatusActive(Frame):
    """Marks a previously idle channel as active again; it rejoins the watermark minimum."""

    is_control: ClassVar[bool] = True


@dataclasses.dataclass(frozen=True, slots=True)
class Barrier(Frame):
    """Checkpoint barrier. **Reserved** for future aligned (exactly-once) checkpointing; the slot
    exists now so adding it later is not a breaking wire-format change."""

    checkpoint_id: int
    is_control: ClassVar[bool] = True


# Convenience singletons for the zero-field control frames. Equality also works (frozen, no
# fields), but the singletons avoid needless allocation on hot control paths.
EOS_FRAME: Final[EOS] = EOS()
IDLE_FRAME: Final[StatusIdle] = StatusIdle()
ACTIVE_FRAME: Final[StatusActive] = StatusActive()


def is_data(frame: Frame) -> TypeGuard[Batch]:
    """True iff ``frame`` is a data frame (i.e. is a :class:`Batch`)."""
    return not frame.is_control


def is_control(frame: Frame) -> bool:
    """True iff ``frame`` is a control frame."""
    return frame.is_control
