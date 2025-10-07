"""
System Monitors - Health monitoring and metrics collection
"""

import asyncio
import logging
import time
from typing import Dict, Any, List
from collections import deque, defaultdict

logger = logging.getLogger(__name__)


class SystemMonitor:
    """System health monitoring and metrics collection"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics_history = deque(maxlen=max_history)
        self.health_checks = {}
        self.alerts = []
        self.start_time = time.time()
    
    async def record_metric(self, metric_name: str, value: float, 
                          tags: Dict[str, str] = None):
        """Record a metric"""
        metric = {
            "name": metric_name,
            "value": value,
            "tags": tags or {},
            "timestamp": time.time()
        }
        self.metrics_history.append(metric)
    
    async def check_health(self, component: str, check_func) -> bool:
        """Perform health check for a component"""
        try:
            result = await check_func()
            self.health_checks[component] = {
                "status": "healthy" if result else "unhealthy",
                "last_check": time.time(),
                "result": result
            }
            return result
        except Exception as e:
            logger.error(f"Health check failed for {component}: {e}")
            self.health_checks[component] = {
                "status": "error",
                "last_check": time.time(),
                "error": str(e)
            }
            return False
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        current_time = time.time()
        uptime = current_time - self.start_time
        
        # Calculate recent metrics (last 5 minutes)
        recent_cutoff = current_time - 300
        recent_metrics = [
            m for m in self.metrics_history 
            if m["timestamp"] > recent_cutoff
        ]
        
        # Group metrics by name
        metric_groups = defaultdict(list)
        for metric in recent_metrics:
            metric_groups[metric["name"]].append(metric["value"])
        
        # Calculate statistics
        metrics_summary = {}
        for name, values in metric_groups.items():
            if values:
                metrics_summary[name] = {
                    "count": len(values),
                    "average": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "latest": values[-1]
                }
        
        return {
            "uptime": uptime,
            "total_metrics": len(self.metrics_history),
            "recent_metrics": len(recent_metrics),
            "health_checks": self.health_checks,
            "metrics_summary": metrics_summary,
            "alerts": self.alerts[-10:]  # Last 10 alerts
        }
    
    async def add_alert(self, severity: str, message: str, component: str = None):
        """Add an alert"""
        alert = {
            "severity": severity,
            "message": message,
            "component": component,
            "timestamp": time.time()
        }
        self.alerts.append(alert)
        
        # Keep only recent alerts
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
        
        logger.warning(f"Alert [{severity}]: {message}")
    
    async def cleanup(self):
        """Cleanup monitor resources"""
        self.metrics_history.clear()
        self.health_checks.clear()
        self.alerts.clear()
