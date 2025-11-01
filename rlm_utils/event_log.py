from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
import time


@dataclass
class Event:
    kind: str
    t: float
    data: Dict[str, Any]


class EventLogger:
    def __init__(self) -> None:
        self.events: List[Event] = []

    def add(self, kind: str, **data: Any) -> None:
        self.events.append(Event(kind=kind, t=time.time(), data=data))

    def dump(self) -> List[Dict[str, Any]]:
        return [dict(kind=e.kind, t=e.t, **e.data) for e in self.events]


_LOGGER: Optional[EventLogger] = None


def get_logger() -> EventLogger:
    global _LOGGER
    if _LOGGER is None:
        _LOGGER = EventLogger()
    return _LOGGER


def reset_logger() -> None:
    global _LOGGER
    _LOGGER = EventLogger()

