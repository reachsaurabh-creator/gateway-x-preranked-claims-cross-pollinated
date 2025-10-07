"""Duel scheduling and referee management."""

import hashlib
import json
import random
import re
from typing import Dict, Any, List

from .config import Config
from .prompts import PromptVault
from .ledger import LEDGER
from .ai_engines import MultiEngineClient


JSON_OBJ_RE = re.compile(r"\{.*\}", re.S)


class DuelScheduler:
    """Manages dueling between claims using multi-engine AI client."""
    
    def __init__(self, config: Config):
        self.config = config
        self.memo: Dict[str, Dict[str, Any]] = {}
        
        if self.config.USE_REAL_LLM:
            self.ai_client = MultiEngineClient(config)
        else:
            self.ai_client = None

    async def initial_round(self, run_id: str, query: str, engines: List[str], num_claims: int = 5) -> List[str]:
        """Generate initial claims using real AI engines or mock data."""
        if self.ai_client:
            try:
                claims = await self.ai_client.generate_initial_claims(query)
                LEDGER.log("initial_claims", {"run_id": run_id, "claims": claims, "source": "real_engines"})
                return claims
            except Exception as e:
                LEDGER.log("initial_claims_error", {"run_id": run_id, "error": str(e)})
                # Fallback to mock claims
        
        # Mock claims for testing or fallback
        claims = [
            f"[{eng}] Answer emphasizing aspect {i} for: {query}" 
            for i, eng in zip(range(1, num_claims + 1), engines * 10)
        ]
        LEDGER.log("initial_claims", {"run_id": run_id, "claims": claims, "source": "mock"})
        return claims

    async def generate_refined_claims(self, run_id: str, query: str, prev_best_claims: List[tuple], round_idx: int) -> List[str]:
        """Generate refined claims based on previous round's best claims."""
        if not self.ai_client:
            # Mock refined claims for testing
            refined_claims = [
                f"[refined-{eng}] Refined answer {i} based on previous round for: {query}" 
                for i, eng in zip(range(1, 4), ["alpha", "beta", "gamma"])
            ]
            LEDGER.log("refined_claims", {"run_id": run_id, "claims": refined_claims, "source": "mock"})
            return refined_claims
        
        try:
            # Create context from previous best claims
            context = self._build_convergence_context(prev_best_claims, round_idx)
            
            # Generate refined claims using convergence context
            refined_claims = await self.ai_client.generate_refined_claims(query, context, round_idx)
            
            LEDGER.log("refined_claims", {
                "run_id": run_id, 
                "claims": refined_claims, 
                "source": "ai_engines",
                "round": round_idx,
                "context_claims": len(prev_best_claims)
            })
            return refined_claims
            
        except Exception as e:
            LEDGER.log("refined_claims_error", {"run_id": run_id, "error": str(e)})
            # Fallback to mock claims
            refined_claims = [
                f"[refined-{eng}] Refined answer {i} (fallback) for: {query}" 
                for i, eng in zip(range(1, 4), ["alpha", "beta", "gamma"])
            ]
            return refined_claims

    def _build_convergence_context(self, prev_best_claims: List[tuple], round_idx: int) -> str:
        """Build comprehensive context from all previous engine responses."""
        if not prev_best_claims:
            return "No previous claims available."
        
        context_parts = [
            f"=== ALL ENGINE RESPONSES FROM ROUND {round_idx - 1} ===",
            "Below are the responses from all AI engines in the previous round:",
            ""
        ]
        
        # Show all engine responses, not just top claims
        for i, (claim_text, score) in enumerate(prev_best_claims, 1):
            # Extract engine name from claim
            engine_name = "Unknown"
            if claim_text.startswith('[') and ']' in claim_text:
                engine_name = claim_text[1:claim_text.index(']')]
            
            context_parts.extend([
                f"--- {engine_name.upper()} ENGINE (Score: {score:.3f}) ---",
                claim_text,
                ""
            ])
        
        context_parts.extend([
            "=== CRITICAL ANALYSIS REQUIRED ===",
            "Your task is to analyze ALL the above responses and provide an improved answer that:",
            "1. Identifies the strengths of each response",
            "2. Addresses the weaknesses and gaps",
            "3. Combines the best insights from all engines",
            "4. Adds new perspectives or corrections",
            "5. Provides a more comprehensive and accurate final answer",
            ""
        ])
        
        return "\n".join(context_parts)

    async def schedule_duel(self, run_id: str, query: str, a: str, b: str, playbook: str) -> Dict[str, Any]:
        """Schedule and execute a duel between two claims."""
        key = self._cache_key(query, a, b, playbook)
        
        # Check cache first
        if key in self.memo:
            duel = self.memo[key]
            LEDGER.log("duel_cached", {"run_id": run_id, **duel})
            return duel

        # Get referee decision
        if self.ai_client:
            try:
                text = await self.ai_client.referee_duel(query, a, b)
            except Exception as e:
                LEDGER.log("referee_error", {"run_id": run_id, "error": str(e)})
                text = self._mock_referee(a, b)
        else:
            text = self._mock_referee(a, b)
        
        # Robust JSON extraction
        m = JSON_OBJ_RE.search(text or "")
        parsed = PromptVault.validate_response(m.group(0) if m else "{}")

        duel = {"a": a, "b": b, "result": parsed, "playbook": playbook}
        self.memo[key] = duel
        LEDGER.log("duel_result", {"run_id": run_id, **duel})
        return duel


    def _mock_referee(self, a: str, b: str) -> str:
        """Mock referee for testing - longer answer wins with 10% flip chance."""
        win = "A" if len(a) >= len(b) else "B"
        if random.random() < 0.1:
            win = "A" if win == "B" else "B"
        return json.dumps({
            "winner": win, 
            "factuality": 0.8, 
            "coherence": 0.85, 
            "note": "mock"
        })

    def _cache_key(self, query: str, a: str, b: str, playbook: str) -> str:
        """Generate cache key for duel results."""
        return hashlib.sha256("|".join([query, a, b, playbook]).encode("utf-8")).hexdigest()
