"""UCB-based playbook selection for duel strategies."""

import math
from typing import Dict

from .config import Config


class PlaybookSelector:
    """Upper Confidence Bound (UCB) based selector for duel playbooks."""
    
    def __init__(self, config: Config):
        self.config = config
        self.arms = ["SelfConsistency", "Debate", "EvidenceFirst", "FocusOnDisputes"]
        self.n = {a: 0 for a in self.arms}  # Number of times each arm was selected
        self.mu = {a: 0.0 for a in self.arms}  # Average reward for each arm
        self.t = 0  # Total number of selections

    def choose_playbook(self, state: Dict[str, float], round_num: int, total_budget: int) -> str:
        """Choose a playbook using UCB algorithm with phase-based priors."""
        self.t += 1
        phase = round_num / max(1, total_budget)
        
        # Phase-based priors (different strategies work better at different phases)
        pri = {
            "SelfConsistency": 0.25 if phase < 0.5 else 0.30,
            "Debate": 0.40 if phase < 0.5 else 0.25,
            "EvidenceFirst": 0.20,
            "FocusOnDisputes": 0.15 if phase < 0.7 else 0.25,
        }
        
        scores = {}
        for a in self.arms:
            # Ensure each arm is explored at least once
            if self.n[a] == 0:
                return a
            
            # UCB formula: mean + confidence bound
            bonus = math.sqrt(math.log(self.t + 1.0) / (self.n[a] + 1.0))
            scores[a] = pri[a] + self.config.UCB_WEIGHT * (self.mu[a] + bonus)
        
        return max(scores, key=scores.get)

    def update_performance(self, arm: str, reward: float):
        """Update the performance statistics for a playbook."""
        self.n[arm] += 1
        # Running average update
        self.mu[arm] += (reward - self.mu[arm]) / self.n[arm]
