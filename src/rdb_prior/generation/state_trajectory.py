"""Interval-valued private temporal-state trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from rdb_prior.priors.model import StateSpacePlan


_INITIAL_ORDINAL = -(2**63)
_CUTOFF_ORDINAL = -(2**62)
_FINAL_ORDINAL = 2**62


@dataclass(frozen=True, slots=True, order=True)
class StatePoint:
    timestamp: int
    ordinal: int

    def __post_init__(self) -> None:
        if isinstance(self.timestamp, bool) or not isinstance(self.timestamp, int):
            raise TypeError("timestamp must be an integer")
        if isinstance(self.ordinal, bool) or not isinstance(self.ordinal, int):
            raise TypeError("ordinal must be an integer")

    def to_dict(self) -> dict[str, int]:
        return {"timestamp": self.timestamp, "ordinal": self.ordinal}

    @classmethod
    def from_dict(cls, data: Mapping[str, int]) -> "StatePoint":
        return cls(timestamp=data["timestamp"], ordinal=data["ordinal"])


@dataclass(frozen=True, slots=True)
class StateInterval:
    state: str
    start: StatePoint
    end: StatePoint

    def __post_init__(self) -> None:
        if not isinstance(self.state, str) or not self.state:
            raise ValueError("state must be a non-empty string")
        if not isinstance(self.start, StatePoint) or not isinstance(self.end, StatePoint):
            raise TypeError("interval boundaries must be StatePoint values")
        if self.start >= self.end:
            raise ValueError("state intervals must be non-empty and ordered")

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state,
            "start": self.start.to_dict(),
            "end": self.end.to_dict(),
        }


class StateTrajectory:
    """A contiguous, half-open state trajectory over logical event positions."""

    def __init__(
        self,
        *,
        calendar_start: int,
        calendar_end: int,
        initial_state: str,
        state_space: StateSpacePlan,
    ) -> None:
        if calendar_end <= calendar_start:
            raise ValueError("calendar_end must be after calendar_start")
        if initial_state not in state_space.values:
            raise ValueError("initial_state must belong to state-space")
        self.calendar_start = calendar_start
        self.calendar_end = calendar_end
        self.state_space = state_space
        self.initial_state = initial_state
        self._changes: list[tuple[StatePoint, str]] = [
            (StatePoint(calendar_start, _INITIAL_ORDINAL), initial_state)
        ]

    @property
    def is_terminal(self) -> bool:
        return self._changes[-1][1] in self.state_space.terminal_states

    @property
    def intervals(self) -> tuple[StateInterval, ...]:
        end = StatePoint(self.calendar_end + 1, _INITIAL_ORDINAL)
        result: list[StateInterval] = []
        for index, (start, state) in enumerate(self._changes):
            next_point = (
                self._changes[index + 1][0]
                if index + 1 < len(self._changes)
                else end
            )
            result.append(StateInterval(state=state, start=start, end=next_point))
        return tuple(result)

    def state_before(self, timestamp: int, ordinal: int | None = None) -> str:
        point = StatePoint(
            timestamp,
            _CUTOFF_ORDINAL if ordinal is None else ordinal,
        )
        return self._state_at(point, inclusive=False)

    def state_after(self, timestamp: int, ordinal: int | None = None) -> str:
        point = StatePoint(
            timestamp,
            _FINAL_ORDINAL if ordinal is None else ordinal,
        )
        return self._state_at(point, inclusive=True)

    def transition(self, *, timestamp: int, ordinal: int, state: str) -> None:
        point = StatePoint(timestamp, ordinal)
        if state not in self.state_space.values:
            raise ValueError("transition state must belong to state-space")
        if self.is_terminal:
            raise ValueError("terminal state cannot transition")
        final = StatePoint(self.calendar_end + 1, _INITIAL_ORDINAL)
        if point <= self._changes[-1][0] or point >= final:
            raise ValueError("transition must advance inside the calendar")
        if state != self._changes[-1][1]:
            self._changes.append((point, state))

    def to_dict(self) -> dict[str, object]:
        return {
            "calendar_start": self.calendar_start,
            "calendar_end": self.calendar_end,
            "initial_state": self.initial_state,
            "state_space": self.state_space.to_dict(),
            "changes": [
                {"point": point.to_dict(), "state": state}
                for point, state in self._changes[1:]
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "StateTrajectory":
        state_space = StateSpacePlan.from_dict(data["state_space"])
        trajectory = cls(
            calendar_start=data["calendar_start"],
            calendar_end=data["calendar_end"],
            initial_state=data["initial_state"],
            state_space=state_space,
        )
        for change in data.get("changes", ()):
            trajectory.transition(
                timestamp=change["point"]["timestamp"],
                ordinal=change["point"]["ordinal"],
                state=change["state"],
            )
        return trajectory

    def _state_at(self, point: StatePoint, *, inclusive: bool) -> str:
        if not self.calendar_start <= point.timestamp <= self.calendar_end:
            raise ValueError("state query lies outside the calendar")
        selected: str | None = None
        for changed_at, state in self._changes:
            if changed_at < point or (inclusive and changed_at == point):
                selected = state
            else:
                break
        if selected is None:
            return self.initial_state
        return selected


class TemporalStateRegistry:
    """Private entity-indexed trajectories; never a database table."""

    def __init__(self, values: Mapping[str, tuple[StateTrajectory, ...]]) -> None:
        self._values = {state_id: tuple(items) for state_id, items in values.items()}

    def trajectory(self, state_id: str, entity_row_id: int) -> StateTrajectory:
        try:
            return self._values[state_id][entity_row_id]
        except KeyError as error:
            raise KeyError(f"No temporal state for {state_id!r}") from error
        except IndexError as error:
            raise IndexError("entity row is outside temporal-state registry") from error

    def trajectories(self, state_id: str) -> tuple[StateTrajectory, ...]:
        try:
            return self._values[state_id]
        except KeyError as error:
            raise KeyError(f"No temporal state for {state_id!r}") from error

    def to_dict(self) -> dict[str, object]:
        return {
            "states": [
                {
                    "state_id": state_id,
                    "trajectories": [item.to_dict() for item in trajectories],
                }
                for state_id, trajectories in sorted(self._values.items())
            ]
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "TemporalStateRegistry":
        return cls(
            {
                item["state_id"]: tuple(
                    StateTrajectory.from_dict(trajectory)
                    for trajectory in item.get("trajectories", ())
                )
                for item in data.get("states", ())
            }
        )

    def values(self) -> Mapping[str, tuple[StateTrajectory, ...]]:
        return MappingProxyType(self._values)


__all__ = [
    "StatePoint",
    "StateInterval",
    "StateTrajectory",
    "TemporalStateRegistry",
]
