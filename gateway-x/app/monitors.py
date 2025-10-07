"""Stop condition monitoring for consensus convergence."""

from .config import Config
from .btl_ranker import BTLRanker


class StopConditionEvaluator:
    """Evaluates various stop conditions for consensus convergence."""
    
    def __init__(self, config: Config):
        self.config = config
        self.last_reason = "running"

    def should_stop(self, btl: BTLRanker, target_confidence: float) -> bool:
        """
        Evaluate if consensus should stop based on various conditions.
        
        Stop conditions (in order of priority):
        1. Maximum rounds reached
        2. CI separation achieved (statistical significance)
        3. Confidence threshold reached
        """
        # Check max rounds
        if btl.rounds() >= self.config.MAX_ROUNDS:
            self.last_reason = "max_rounds"
            return True
        
        # Check CI separation (requires minimum rounds)
        if (btl.rounds() >= self.config.CI_MIN_ROUNDS and 
            btl.ci_gap() >= self.config.CI_SEPARATION_MIN):
            self.last_reason = "ci_separation"
            return True
        
        # Check confidence threshold
        if btl.get_confidence_proxy() >= target_confidence:
            self.last_reason = "confidence_threshold"
            return True
        
        self.last_reason = "running"
        return False

    def get_stop_reason(self) -> str:
        """Get the reason for the last stop decision."""
        return self.last_reason
