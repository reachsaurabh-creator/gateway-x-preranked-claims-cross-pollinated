"""
Gateway X - Multi-Engine AI Consensus System

A production-ready system for achieving consensus among multiple AI models
using statistical methods including Bradley-Terry ranking, bootstrap confidence
intervals, and adaptive strategy selection.
"""

__version__ = "32.1.0"
__author__ = "Gateway X Team"
__email__ = "team@gatewayx.ai"

from .server import app
from .settings import Settings

__all__ = ["app", "Settings"]
