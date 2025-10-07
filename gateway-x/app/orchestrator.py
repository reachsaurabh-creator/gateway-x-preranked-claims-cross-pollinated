"""Main orchestrator for consensus rounds and timeline management."""

import asyncio
import hashlib
import time
from typing import Dict, Any, List

from .config import Config
from .schemas import TimelineItem, ClaimScore
from .duel_scheduler import DuelScheduler
from .btl_ranker import BTLRanker
from .playbook_selector import PlaybookSelector
from .monitors import StopConditionEvaluator
from .ledger import LEDGER


class Orchestrator:
    """
    Runs consensus rounds until governance thresholds are met.
    
    For each round:
      1) Select informative pairs; run parallel duels.
      2) Update BTL; compute convergence score (truth score).
      3) Build a "truth response" (summary) + record top claims.
    
    Stores a per-run timeline for /timeline and /timeline/report.
    """

    def __init__(self, config: Config):
        self.config = config
        self.scheduler = DuelScheduler(config)
        self.btl = BTLRanker(config)
        self.selector = PlaybookSelector(config)
        self.stopper = StopConditionEvaluator(config)
        self.runs: Dict[str, List[TimelineItem]] = {}  # run_id -> timeline
        self.current_engines: List[str] = [
            "engine.alpha", "engine.beta", "engine.gamma", "engine.delta"
        ]

    async def run(self, query: str, budget: int, confidence_threshold: float) -> Dict[str, Any]:
        """Run the complete consensus process."""
        run_id = hashlib.sha256(f"{time.time()}|{query}".encode()).hexdigest()[:12]
        timeline: List[TimelineItem] = []
        self.runs[run_id] = timeline

        # Create fresh BTL ranker for this query
        btl = BTLRanker(self.config)
        selector = PlaybookSelector(self.config)
        stopper = StopConditionEvaluator(self.config)

        # Round 0: initial claims
        claims = await self.scheduler.initial_round(
            run_id, query, engines=self.current_engines, num_claims=5
        )
        claims = self._filter_invalid_claims(claims)
        btl.add_claims(claims)

        total_duels = 0
        prev_conf = 0.0
        stop_reason = "budget_exhausted"

        # Main consensus loop
        convergence_preserved = False
        for round_idx in range(1, budget + 1):
            # ITERATIVE REFINEMENT: Generate new claims based on previous round
            # BUT ONLY if we haven't reached good convergence yet
            current_confidence = btl.get_confidence_proxy()
            
            if round_idx > 1 and round_idx <= 4 and not convergence_preserved:
                # Get ALL claims from previous round for comprehensive analysis
                prev_all_claims = sorted(
                    btl.snapshot_state().items(), 
                    key=lambda kv: kv[1], 
                    reverse=True
                )  # ALL claims for comprehensive context
                
                # Generate refined claims using ALL previous round context
                refined_claims = await self.scheduler.generate_refined_claims(
                    run_id, query, prev_all_claims, round_idx
                )
                
                # Filter and replace original claims with refined ones
                refined_claims = self._filter_invalid_claims(refined_claims)
                if refined_claims:
                    # REPLACE original claims with refined ones for critical feedback loop
                    # Remove original claims from BTL and clean up related data
                    original_claims = list(btl.theta.keys())
                    for claim in original_claims:
                        if claim in btl.theta:
                            del btl.theta[claim]
                    
                    # Clean up wins dictionary to remove stale references
                    btl.wins = {k: v for k, v in btl.wins.items() 
                               if k[0] not in original_claims and k[1] not in original_claims}
                    
                    # Add refined claims to BTL (they'll start with neutral scores)
                    btl.add_claims(refined_claims)
                    
                    # Replace the claims list with refined ones
                    claims = refined_claims
                    
                    LEDGER.log("refined_claims_replacement", {
                        "run_id": run_id, 
                        "round": round_idx,
                        "original_claims": len(original_claims),
                        "refined_claims": len(refined_claims),
                        "confidence": current_confidence,
                        "action": "replaced_original_with_refined"
                    })
            elif current_confidence >= 0.8 and not convergence_preserved:
                # Only stop adding new claims at very high confidence
                convergence_preserved = True
                LEDGER.log("convergence_preserved", {
                    "run_id": run_id,
                    "round": round_idx,
                    "confidence": current_confidence,
                    "reason": "Stopped adding new claims to preserve convergence"
                })

            # Select playbook and informative pairs from current claims
            playbook = selector.choose_playbook(
                btl.snapshot_state(), round_idx, budget
            )
            pairs = btl.select_k_informative_pairs(claims, k=self.config.DUELS_PER_ROUND)

            # Run duels in parallel
            tasks = [
                self.scheduler.schedule_duel(run_id, query, a, b, playbook) 
                for a, b in pairs
            ]
            results = []
            if tasks:
                results = await asyncio.gather(*tasks)

            # Update BTL scores
            for duel in results:
                btl.update(duel)
            total_duels += len(results)

            # Compute confidence intervals if enough rounds elapsed
            if btl.rounds() >= self.config.CI_MIN_ROUNDS:
                btl.compute_cis(self.config.CI_BOOTSTRAP_SAMPLES)

            # Update UCB performance (reward = Δ confidence proxy)
            cur_conf = btl.get_confidence_proxy()
            selector.update_performance(playbook, max(0.0, cur_conf - prev_conf))
            prev_conf = cur_conf

            # Assemble truth response (summary) for this round
            best = btl.best_claim()
            ranked = sorted(
                btl.snapshot_state().items(), 
                key=lambda kv: kv[1], 
                reverse=True
            )
            
            # Top-5 claims with confidence intervals
            top = [
                ClaimScore(
                    cid=cid,
                    score=score,
                    ci_low=btl.cis.get(cid, (0.0, 1.0))[0],
                    ci_high=btl.cis.get(cid, (0.0, 1.0))[1],
                )
                for cid, score in ranked[:5]
            ]
            
            truth_summary = self._build_truth_summary(best, top)

            # Create timeline item
            item = TimelineItem(
                run_id=run_id,
                round_index=round_idx,
                convergence_score=cur_conf,
                best_claim_cid=best,
                best_claim_text=best,
                summary=truth_summary,
                top_claims=top,
            )
            timeline.append(item)
            LEDGER.log("round_summary", item.model_dump())

            # Check stop conditions
            if stopper.should_stop(btl, confidence_threshold):
                stop_reason = stopper.get_stop_reason()
                break

        # Final result
        result = {
            "run_id": run_id,
            "query": query,
            "best_claim": btl.best_claim(),
            "confidence": btl.get_confidence_proxy(),
            "rounds": len(timeline),
            "total_duels": total_duels,
            "stop_reason": stop_reason,
        }
        LEDGER.log("final_result", result)
        return result

    def get_timeline(self, run_id: str) -> List[TimelineItem]:
        """Get timeline for a specific run."""
        return self.runs.get(run_id, [])

    def _filter_invalid_claims(self, claims: List[str]) -> List[str]:
        """Filter out invalid, duplicate, or error claims."""
        seen = set()
        clean = []
        
        for c in claims:
            # Skip error messages
            if any(error_indicator in c.lower() for error_indicator in [
                "error", "failed to generate", "api error", "not found", 
                "404", "500", "timeout", "exception"
            ]):
                continue
                
            # Normalize for duplicate detection
            s = "".join(ch for ch in c.lower().strip() if ch.isalnum() or ch.isspace())
            if len(s) < self.config.MIN_CLAIM_CHARS or s in seen:
                continue
            seen.add(s)
            clean.append(c.strip())
        
        # If we filtered out too many, keep some original claims
        if len(clean) < 2 and len(claims) > 0:
            # Take the first few non-error claims
            for c in claims:
                if not any(error_indicator in c.lower() for error_indicator in [
                    "error", "failed to generate", "api error", "not found", 
                    "404", "500", "timeout", "exception"
                ]):
                    clean.append(c.strip())
                    if len(clean) >= 2:
                        break
        
        return clean or claims

    def _build_truth_summary(self, best_cid: str, top: List[ClaimScore]) -> str:
        """
        Build a deterministic summary that:
          - names current best claim,
          - cites up to 3 runner-up IDs (with scores),
          - notes divergence (CI overlaps) when present.
        """
        if not best_cid:
            return "No consensus yet."
        
        others = [cs for cs in top if cs.cid != best_cid]
        diverge = [
            cs for cs in others 
            if cs.ci_high >= next((t.ci_low for t in top if t.cid == best_cid), 0.0)
        ]
        
        parts = [f"Truth response selects {best_cid} as current best."]
        
        if others:
            parts.append(
                "Next contenders: " + 
                ", ".join(f"{cs.cid} (Tseek Score {cs.score:.2f})" for cs in others[:3]) + 
                "."
            )
        
        if diverge:
            parts.append("Some divergence remains (CI overlap with best).")
        
        return " ".join(parts)
