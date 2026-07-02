"""The ``Frame`` model: the sealed set of frame types that flow on a channel.

Every edge in a Nautilus dataflow carries two kinds of frame:

* **data** frames — only :class:`Batch` (an Arrow ``RecordBatch``), and
* **control** frames — :class:`EOS` and (reserved) :class:`Barrier`.

Data frames are routed by the edge's partitioner; control frames are always *broadcast* to every
downstream instance. The class-level :attr:`Frame.is_control` flag distinguishes them.
"""

from __future__ import annotations

import dataclasses
from typing import ClassVar, Final, TypeGuard

import pyarrow as pa

# --- The sealed Frame union --------------------------------------------------------------------


class Frame:
    """Base class for everything that flows on a channel.

    Sealed: the only subclasses are :class:`Batch`, :class:`EOS`, and :class:`Barrier`. Use
    :func:`is_data` / :func:`is_control` (or the :attr:`is_control` class flag) to discriminate; the
    concrete subclasses support structural pattern matching.
    """

    __slots__ = ()
    #: Overridden by every concrete subclass. True for control frames.
    is_control: ClassVar[bool]


@dataclasses.dataclass(frozen=True, slots=True)
class Batch(Frame):
    """A micro-batch of data records, carried as an Arrow ``RecordBatch``.

    This is the only data frame. Keying and partitioning operate on *columns* of ``data``
    (Arrow-first), so the batch itself carries no separate key — the operator and the edge's
    partitioner know which columns form the key.
    """

    data: pa.RecordBatch
    is_control: ClassVar[bool] = False

    @property
    def num_rows(self) -> int:
        return int(self.data.num_rows)


@dataclasses.dataclass(frozen=True, slots=True)
class EOS(Frame):
    """End of stream. The sole terminal frame.

    An operator forwards ``EOS`` downstream only after it has received ``EOS`` on *all* of its inputs
    and flushed all pending state (its :meth:`~nautilus.core.operator.OneInputOperator.on_eos` hook).
    """

    is_control: ClassVar[bool] = True


@dataclasses.dataclass(frozen=True, slots=True)
class Barrier(Frame):
    """Checkpoint barrier. **Reserved** for future aligned (exactly-once) checkpointing; the slot
    exists now so adding it later is not a breaking wire-format change."""

    checkpoint_id: int
    is_control: ClassVar[bool] = True


# Convenience singleton for the zero-field terminal frame. Equality also works (frozen, no fields),
# but the singleton avoids needless allocation on the hot control path.
EOS_FRAME: Final[EOS] = EOS()


def is_data(frame: Frame) -> TypeGuard[Batch]:
    """True iff ``frame`` is a data frame (i.e. is a :class:`Batch`)."""
    return not frame.is_control


def is_control(frame: Frame) -> bool:
    """True iff ``frame`` is a control frame."""
    return frame.is_control
