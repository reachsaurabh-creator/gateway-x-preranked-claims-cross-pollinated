"""
Transparency Module - Full logging and auditability
"""

import json
import logging
import time
from typing import Dict, Any, List, Optional
from pathlib import Path
import aiofiles

from .settings import settings

logger = logging.getLogger(__name__)


class TransparencyLogger:
    """Comprehensive logging for full auditability"""
    
    def __init__(self):
        self.log_file = Path(settings.response_log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.in_memory_logs = []
        self.max_memory_logs = settings.max_responses_in_memory
    
    async def log_query(self, query_id: str, query: str, 
                       responses: List[Dict[str, Any]], 
                       result: Dict[str, Any],
                       metadata: Dict[str, Any] = None):
        """Log a complete query and its processing"""
        log_entry = {
            "timestamp": time.time(),
            "query_id": query_id,
            "query": query,
            "responses": responses,
            "result": result,
            "metadata": metadata or {},
            "version": "32.1.0"
        }
        
        # Add to in-memory logs
        self.in_memory_logs.append(log_entry)
        if len(self.in_memory_logs) > self.max_memory_logs:
            self.in_memory_logs = self.in_memory_logs[-self.max_memory_logs:]
        
        # Persist to file if enabled
        if settings.persist_responses:
            await self._persist_log(log_entry)
    
    async def _persist_log(self, log_entry: Dict[str, Any]):
        """Persist log entry to file"""
        try:
            async with aiofiles.open(self.log_file, 'a') as f:
                await f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            logger.error(f"Failed to persist log: {e}")
    
    async def get_query_history(self, query_id: Optional[str] = None, 
                              limit: int = 100) -> List[Dict[str, Any]]:
        """Get query history"""
        if query_id:
            return [log for log in self.in_memory_logs if log["query_id"] == query_id]
        else:
            return self.in_memory_logs[-limit:]
    
    async def replay_query(self, query_id: str) -> Optional[Dict[str, Any]]:
        """Replay a specific query"""
        history = await self.get_query_history(query_id)
        if history:
            return history[0]
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics"""
        return {
            "total_logs": len(self.in_memory_logs),
            "log_file": str(self.log_file),
            "persist_enabled": settings.persist_responses,
            "max_memory_logs": self.max_memory_logs
        }
