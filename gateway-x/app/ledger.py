"""Transparency ledger for structured logging of consensus events."""

import json
import logging
import time
from typing import Dict, Any, List


logger = logging.getLogger("gatewayx")
logging.basicConfig(level=logging.INFO)


class TransparencyLedger:
    """Structured logging of all key events (claims, duels, rounds, final result)."""

    def __init__(self):
        self.events: List[Dict[str, Any]] = []

    def log(self, event_type: str, payload: Dict[str, Any]):
        """Log an event to the transparency ledger."""
        evt = {
            "ts": round(time.time(), 3),
            "type": event_type,
            **payload,
        }
        self.events.append(evt)
        logger.info("ledger_event=%s", json.dumps(evt, separators=(",", ":")))

    def get_events(self, event_type: str = None) -> List[Dict[str, Any]]:
        """Get events, optionally filtered by type."""
        if event_type is None:
            return self.events
        return [evt for evt in self.events if evt.get("type") == event_type]

    def clear(self):
        """Clear all events from the ledger."""
        self.events.clear()


# Global ledger instance
LEDGER = TransparencyLedger()
