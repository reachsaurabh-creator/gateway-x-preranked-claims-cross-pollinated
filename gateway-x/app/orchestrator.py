"""Main orchestrator for preranked claims cross-pollination consensus."""

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
from .cross_pollination import CrossPollinationEngine, CrossPollinationResult
from .ai_engines import MultiEngineClient


class Orchestrator:
    """
    Runs preranked claims cross-pollination consensus until governance thresholds are met.
    
    For each round:
      1) Generate responses with extracted claims from all engines
      2) Cross-pollinate claims between engines for improved responses
      3) Update BTL scores based on extracted claims
      4) Compute convergence score based on claims similarity and BTL scores
    
    Stores a per-run timeline for /timeline and /timeline/report.
    """

    def __init__(self, config: Config):
        self.config = config
        self.scheduler = DuelScheduler(config)
        self.btl = BTLRanker(config)
        self.selector = PlaybookSelector(config)
        self.stopper = StopConditionEvaluator(config)
        self.cross_pollination = CrossPollinationEngine(config)
        self.multi_engine_client = MultiEngineClient(config)
        self.runs: Dict[str, List[TimelineItem]] = {}  # run_id -> timeline

    async def run(self, query: str, budget: int, confidence_threshold: float) -> Dict[str, Any]:
        """Run the complete preranked claims cross-pollination consensus process."""
        run_id = hashlib.sha256(f"{time.time()}|{query}".encode()).hexdigest()[:12]
        timeline: List[TimelineItem] = []
        self.runs[run_id] = timeline

        # Create fresh components for this query
        btl = BTLRanker(self.config)
        selector = PlaybookSelector(self.config)
        stopper = StopConditionEvaluator(self.config)
        
        # Reset cross-pollination state
        self.cross_pollination.reset()

        # Round 1: Initial responses with claims extraction
        LEDGER.log("cross_pollination_start", {
            "run_id": run_id,
            "query": query,
            "budget": budget,
            "confidence_threshold": confidence_threshold,
            "engines": list(self.multi_engine_client.engines.keys())
        })
        
        initial_results = await self.cross_pollination.run_initial_round(
            query, self.multi_engine_client.engines
        )
        
        # Extract all initial claims and add to BTL
        initial_claims = []
        for result in initial_results:
            initial_claims.extend([claim.text for claim in result.extracted_claims])
        
        if initial_claims:
            btl.add_claims(initial_claims)
        
        total_duels = 0
        stop_reason = "budget_exhausted"

        # Main cross-pollination consensus loop
        for round_idx in range(2, budget + 1):
            # Run cross-pollination round
            cross_pollination_results = await self.cross_pollination.run_cross_pollination_round(
                query, self.multi_engine_client.engines, round_idx
            )
            
            # Extract new claims and update BTL
            new_claims = []
            for result in cross_pollination_results:
                new_claims.extend([claim.text for claim in result.extracted_claims])
            
            if new_claims:
                btl.add_claims(new_claims)

            # Run duels on current claims pool for BTL updates
            all_claims = list(btl.theta.keys())
            if len(all_claims) >= 2:
                playbook = selector.choose_playbook(
                    btl.snapshot_state(), round_idx, budget
                )
                pairs = btl.select_k_informative_pairs(all_claims, k=self.config.DUELS_PER_ROUND)

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

            # Calculate hybrid confidence (BTL + claims convergence)
            btl_confidence = btl.get_confidence_proxy()
            claims_convergence = self.cross_pollination.calculate_claims_convergence()
            hybrid_confidence = (btl_confidence + claims_convergence) / 2

            # Assemble truth response for this round
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
                convergence_score=hybrid_confidence,
                best_claim_cid=best,
                best_claim_text=best,
                summary=truth_summary,
                top_claims=top,
            )
            timeline.append(item)
            LEDGER.log("cross_pollination_round_summary", {
                "run_id": run_id,
                "round": round_idx,
                "btl_confidence": btl_confidence,
                "claims_convergence": claims_convergence,
                "hybrid_confidence": hybrid_confidence,
                "total_claims": len(self.cross_pollination.get_all_claims()),
                "new_claims": len(new_claims),
                "duels": len(results) if 'results' in locals() else 0
            })

            # Check stop conditions using hybrid confidence
            if hybrid_confidence >= confidence_threshold:
                stop_reason = "confidence_threshold"
                break
                
            if round_idx >= budget:
                stop_reason = "budget_exhausted"
                break

        # Final result
        final_btl_confidence = btl.get_confidence_proxy()
        final_claims_convergence = self.cross_pollination.calculate_claims_convergence()
        final_hybrid_confidence = (final_btl_confidence + final_claims_convergence) / 2
        
        result = {
            "run_id": run_id,
            "query": query,
            "best_claim": btl.best_claim(),
            "confidence": final_hybrid_confidence,
            "btl_confidence": final_btl_confidence,
            "claims_convergence": final_claims_convergence,
            "rounds": len(timeline),
            "total_duels": total_duels,
            "total_claims": len(self.cross_pollination.get_all_claims()),
            "stop_reason": stop_reason,
        }
        LEDGER.log("cross_pollination_final_result", result)
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
