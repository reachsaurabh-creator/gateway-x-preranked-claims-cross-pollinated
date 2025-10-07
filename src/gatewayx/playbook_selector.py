"""
Playbook Selector - UCB-based strategy selection
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional
import time

logger = logging.getLogger(__name__)


class PlaybookSelector:
    """UCB-based playbook selection for adaptive strategy"""
    
    def __init__(self):
        self.strategies = {
            "weighted": {"count": 0, "total_reward": 0.0, "last_used": 0},
            "round_robin": {"count": 0, "total_reward": 0.0, "last_used": 0},
            "least_loaded": {"count": 0, "total_reward": 0.0, "last_used": 0}
        }
        self.exploration_factor = 2.0
    
    def select_strategy(self, available_strategies: Optional[List[str]] = None) -> str:
        """Select strategy using UCB algorithm"""
        if available_strategies is None:
            available_strategies = list(self.strategies.keys())
        
        if not available_strategies:
            return "weighted"  # Default fallback
        
        if len(available_strategies) == 1:
            return available_strategies[0]
        
        # Calculate UCB scores
        ucb_scores = {}
        total_count = sum(self.strategies[s]["count"] for s in available_strategies)
        
        for strategy in available_strategies:
            stats = self.strategies[strategy]
            
            if stats["count"] == 0:
                # First time using this strategy
                ucb_scores[strategy] = float('inf')
            else:
                # Calculate UCB score
                avg_reward = stats["total_reward"] / stats["count"]
                exploration = self.exploration_factor * np.sqrt(
                    np.log(total_count) / stats["count"]
                )
                ucb_scores[strategy] = avg_reward + exploration
        
        # Select strategy with highest UCB score
        selected = max(ucb_scores.items(), key=lambda x: x[1])[0]
        
        # Update last used time
        self.strategies[selected]["last_used"] = time.time()
        
        return selected
    
    def update_reward(self, strategy: str, reward: float):
        """Update reward for a strategy"""
        if strategy in self.strategies:
            self.strategies[strategy]["count"] += 1
            self.strategies[strategy]["total_reward"] += reward
    
    def get_strategy_stats(self) -> Dict[str, Any]:
        """Get statistics for all strategies"""
        stats = {}
        for name, data in self.strategies.items():
            if data["count"] > 0:
                stats[name] = {
                    "count": data["count"],
                    "average_reward": data["total_reward"] / data["count"],
                    "total_reward": data["total_reward"],
                    "last_used": data["last_used"]
                }
            else:
                stats[name] = {
                    "count": 0,
                    "average_reward": 0.0,
                    "total_reward": 0.0,
                    "last_used": 0
                }
        return stats
