"""Bradley-Terry-Luce ranking with bootstrap confidence intervals."""

import math
import random
from itertools import combinations
from typing import Dict, List, Tuple, Any

from .config import Config


class BTLRanker:
    """Bradley-Terry-Luce model with MM updates and bootstrap confidence intervals."""
    
    def __init__(self, config: Config):
        self.config = config
        self.theta: Dict[str, float] = {}  # BTL scores
        self.wins: Dict[Tuple[str, str], int] = {}  # Win counts
        self.duels: List[Tuple[str, str, str]] = []  # Duel history
        self._rounds = 0
        self.cis: Dict[str, Tuple[float, float]] = {}  # Confidence intervals
        self._best_claim_history: List[str] = []  # Track best claim over rounds

    def add_claims(self, claims: List[str]):
        """Initialize claims with uniform scores."""
        for c in claims:
            self.theta.setdefault(c, 1.0 / max(1, len(claims)))
        self._normalize()

    def update(self, duel: Dict[str, Any]):
        """Update BTL scores based on duel result."""
        a, b, w = duel["a"], duel["b"], duel["result"]["winner"]
        if w not in ("A", "B"):
            return
            
        # Record win
        key = (a, b) if w == "A" else (b, a)
        self.wins[key] = self.wins.get(key, 0) + 1
        self.duels.append((a, b, w))
        
        # Update scores using MM algorithm
        self._mm_update()
        self._rounds += 1
        
        # Track best claim history for stability analysis
        if self.theta:
            current_best = self.best_claim()
            self._best_claim_history.append(current_best)

    def _mm_update(self):
        """Minorization-Maximization update with strong regularization to prevent extreme values."""
        for _ in range(self.config.BTL_ITERS):
            denom = {i: 0.0 for i in self.theta}
            wins = {i: 0.0 for i in self.theta}
            
            for (i, j), w_ij in self.wins.items():
                # Skip if claims no longer exist in theta (due to claim replacement)
                if i not in self.theta or j not in self.theta:
                    continue
                    
                n_ij = w_ij + self.wins.get((j, i), 0)
                s = self.theta[i] + self.theta[j]
                if s <= 0:
                    continue
                    
                denom[i] += n_ij / s
                denom[j] += n_ij / s
                wins[i] += w_ij
                wins[j] += self.wins.get((j, i), 0)
            
            # STRONG REGULARIZATION: Prevent extreme winner-take-all scenarios
            for i in self.theta:
                if denom[i] > 0:
                    # Strong regularization to keep scores reasonable
                    regularization = 0.5  # Much stronger regularization
                    raw_update = self.theta[i] * (wins[i] / denom[i])
                    # Smooth the update heavily to prevent extreme values
                    self.theta[i] = max(
                        self.config.BTL_FLOOR, 
                        raw_update * (1 - regularization) + self.theta[i] * regularization
                    )
        
        # ADDITIONAL CONSTRAINTS: Ensure no single claim dominates
        self._apply_score_constraints()
        self._normalize()
    
    def _apply_score_constraints(self):
        """Apply additional constraints to prevent extreme score distributions."""
        if not self.theta:
            return
        
        # Find the maximum score
        max_score = max(self.theta.values())
        
        # If any score is too dominant (>80%), redistribute some weight
        if max_score > 0.8:
            # Find the dominant claim
            dominant_claim = max(self.theta.keys(), key=lambda k: self.theta[k])
            
            # Redistribute 20% of the dominant score to others
            redistribution = max_score * 0.2
            self.theta[dominant_claim] -= redistribution
            
            # Distribute evenly among other claims
            other_claims = [k for k in self.theta.keys() if k != dominant_claim]
            if other_claims:
                per_claim = redistribution / len(other_claims)
                for claim in other_claims:
                    self.theta[claim] += per_claim

    def _normalize(self):
        """Normalize BTL scores to sum to 1."""
        z = sum(self.theta.values()) or 1.0
        for k in self.theta:
            self.theta[k] /= z

    def select_k_informative_pairs(self, claims: List[str], k: int) -> List[Tuple[str, str]]:
        """Select k most informative pairs based on entropy."""
        pairs = list(combinations(claims, 2))
        random.shuffle(pairs)
        sample = pairs[:min(len(pairs), self.config.PAIR_SAMPLE_LIMIT)]
        
        # Score pairs by entropy (uncertainty)
        scored: List[Tuple[Tuple[str, str], float]] = []
        for a, b in sample:
            p = self._p_i_gt_j(a, b)
            h = -(p * math.log(p + 1e-12) + (1 - p) * math.log(1 - p + 1e-12))
            scored.append(((a, b), h))
        
        # Sort by entropy (descending) and select most informative pairs
        # ALLOW OVERLAPPING PAIRS to ensure comprehensive testing
        scored.sort(key=lambda x: x[1], reverse=True)
        chosen = []
        
        for (a, b), _ in scored:
            if len(chosen) >= k:
                break
            # Allow overlapping pairs for better coverage
            chosen.append((a, b))
        
        return chosen or ([(claims[0], claims[-1])] if len(claims) >= 2 else [])

    def compute_cis(self, n_bootstrap: int):
        """Compute bootstrap confidence intervals for BTL scores."""
        if len(self.duels) < 2:
            self.cis = {c: (0.0, 1.0) for c in self.theta}
            return
        
        # Bootstrap sampling
        samples: List[Dict[str, float]] = []
        for _ in range(n_bootstrap):
            tmp = BTLRanker(self.config)
            tmp.add_claims(list(self.theta.keys()))
            
            # Resample duels with replacement
            for _ in range(len(self.duels)):
                a, b, w = random.choice(self.duels)
                tmp.update({"a": a, "b": b, "result": {"winner": w}})
            
            samples.append(tmp.theta.copy())
        
        # Compute 95% confidence intervals
        self.cis = {}
        for c in self.theta:
            vals = sorted(s[c] for s in samples)
            lo = vals[int(0.025 * (len(vals) - 1))]
            hi = vals[int(0.975 * (len(vals) - 1))]
            self.cis[c] = (lo, hi)

    def ci_gap(self) -> float:
        """Compute gap between top two claims' confidence intervals."""
        if len(self.theta) < 2 or not self.cis:
            return 0.0
        
        ranked = sorted(self.theta.items(), key=lambda kv: kv[1], reverse=True)
        top, second = ranked[0][0], ranked[1][0]
        
        top_lo, _ = self.cis.get(top, (0.0, 1.0))
        _, sec_hi = self.cis.get(second, (0.0, 1.0))
        
        return max(0.0, top_lo - sec_hi)

    def get_confidence_proxy(self) -> float:
        """Get reasonable confidence proxy that reflects actual convergence."""
        vals = sorted(self.theta.values(), reverse=True)
        if len(vals) < 2:
            return 1.0
        
        # FIXED MARGIN CALCULATION: Use log-odds for better sensitivity
        # Convert to log-odds to prevent extreme values
        top_score = vals[0]
        second_score = vals[1]
        
        # Use log-odds ratio for more reasonable margin calculation
        if top_score <= 0 or second_score <= 0:
            margin = 0.0
        else:
            log_odds = math.log(top_score / second_score)
            # Normalize log-odds to [0, 1] range
            margin = min(1.0, max(0.0, log_odds / 3.0))  # log(20) ≈ 3 for 95% vs 5%
        
        # ROUND FACTOR: Gradual increase with more evidence
        rfac = min(1.0, self._rounds / max(4.0, float(self.config.CI_MIN_ROUNDS)))
        
        # BASE CONFIDENCE: Reasonable starting point
        base_confidence = margin * rfac
        
        # STABILITY FACTOR: Reward consistent winners
        stability = self.get_convergence_stability()
        stability_bonus = stability * 0.2  # Max 20% bonus for stability
        
        # COMBINE FACTORS
        final_confidence = base_confidence + stability_bonus
        
        # CONVERGENCE PRESERVATION: Prevent dramatic drops but don't force high confidence
        if not hasattr(self, '_peak_confidence'):
            self._peak_confidence = 0.0
        
        if final_confidence > self._peak_confidence:
            self._peak_confidence = final_confidence
        
        # Only preserve if we had meaningful convergence (>0.3)
        if self._peak_confidence > 0.3:
            min_confidence = self._peak_confidence * 0.6  # Allow 40% drop
            final_confidence = max(final_confidence, min_confidence)
        
        return max(0.0, min(1.0, final_confidence))

    def get_convergence_stability(self) -> float:
        """Calculate how stable the best claim has been over recent rounds."""
        if len(self._best_claim_history) < 3:
            return 0.0
        
        # Look at last 3 rounds for stability
        recent_claims = self._best_claim_history[-3:]
        current_best = recent_claims[-1]
        
        # Count how many times the current best claim appeared in recent rounds
        stability_count = sum(1 for claim in recent_claims if claim == current_best)
        
        # Return stability score (0.0 to 1.0)
        return stability_count / len(recent_claims)

    def best_claim(self) -> str:
        """Get the highest-scoring claim."""
        return max(self.theta, key=self.theta.get) if self.theta else ""

    def snapshot_state(self) -> Dict[str, float]:
        """Get current BTL scores."""
        return dict(self.theta)

    def rounds(self) -> int:
        """Get number of rounds completed."""
        return self._rounds

    def _p_i_gt_j(self, i: str, j: str) -> float:
        """Probability that claim i beats claim j."""
        t = self.theta[i] + self.theta[j]
        return self.theta[i] / t if t > 0 else 0.5
