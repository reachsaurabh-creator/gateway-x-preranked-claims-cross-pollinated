"""Cross-pollination engine for preranked claims algorithm."""

import asyncio
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from .claims_extractor import ExtractedClaim, ClaimsExtractor
from .ai_engines import AIEngine
from .config import Config

logger = logging.getLogger("gatewayx")


@dataclass
class CrossPollinationResult:
    """Result of cross-pollination round."""
    engine: str
    response: str
    extracted_claims: List[ExtractedClaim]
    round_idx: int


class CrossPollinationEngine:
    """Manages cross-pollination between AI engines using extracted claims."""
    
    def __init__(self, config: Config):
        self.config = config
        self.claims_extractor = ClaimsExtractor()
        self.claims_history: List[ExtractedClaim] = []
    
    async def run_initial_round(
        self, 
        query: str, 
        engines: Dict[str, AIEngine]
    ) -> List[CrossPollinationResult]:
        """Run initial round where each engine generates response and extracts claims."""
        logger.info(f"Starting initial cross-pollination round for query: {query}")
        
        tasks = []
        for engine_name, engine in engines.items():
            tasks.append(self._generate_initial_response(query, engine_name, engine))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_results = []
        for i, result in enumerate(results):
            engine_name = list(engines.keys())[i]
            if isinstance(result, Exception):
                logger.error(f"Engine {engine_name} failed in initial round: {result}")
                # Create fallback result
                fallback_result = CrossPollinationResult(
                    engine=engine_name,
                    response=f"[{engine_name}] Error in initial response generation: {str(result)}",
                    extracted_claims=[],
                    round_idx=1
                )
                valid_results.append(fallback_result)
            else:
                valid_results.append(result)
        
        # Extract and store all claims
        for result in valid_results:
            self.claims_history.extend(result.extracted_claims)
        
        logger.info(f"Initial round completed: {len(valid_results)} engines, {len(self.claims_history)} total claims")
        return valid_results
    
    async def run_cross_pollination_round(
        self, 
        query: str, 
        engines: Dict[str, AIEngine],
        round_idx: int
    ) -> List[CrossPollinationResult]:
        """Run cross-pollination round where engines analyze all claims and generate improved responses."""
        logger.info(f"Starting cross-pollination round {round_idx}")
        
        # Format claims for cross-pollination
        claims_context = self.claims_extractor.format_claims_for_cross_pollination(self.claims_history)
        
        tasks = []
        for engine_name, engine in engines.items():
            tasks.append(self._generate_cross_pollinated_response(
                query, engine_name, engine, claims_context, round_idx
            ))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_results = []
        new_claims = []
        
        for i, result in enumerate(results):
            engine_name = list(engines.keys())[i]
            if isinstance(result, Exception):
                logger.error(f"Engine {engine_name} failed in cross-pollination round {round_idx}: {result}")
                # Create fallback result
                fallback_result = CrossPollinationResult(
                    engine=engine_name,
                    response=f"[{engine_name}-r{round_idx}] Cross-pollination failed: {str(result)}",
                    extracted_claims=[],
                    round_idx=round_idx
                )
                valid_results.append(fallback_result)
            else:
                valid_results.append(result)
                new_claims.extend(result.extracted_claims)
        
        # Deduplicate and merge new claims with history
        if new_claims:
            all_claims = self.claims_history + new_claims
            self.claims_history = self.claims_extractor.deduplicate_claims(all_claims)
        
        logger.info(f"Cross-pollination round {round_idx} completed: {len(valid_results)} engines, {len(new_claims)} new claims")
        return valid_results
    
    async def _generate_initial_response(
        self, 
        query: str, 
        engine_name: str, 
        engine: AIEngine
    ) -> CrossPollinationResult:
        """Generate initial response and extract claims."""
        try:
            prompt = f"""You are participating in a multi-AI consensus process. Your task is to:

1. Provide a comprehensive response to the question
2. Extract and rank the key claims/arguments from your response

QUESTION: {query}

Please provide:
1. A detailed, well-reasoned response (2-4 paragraphs)
2. A ranked list of 3-7 key claims extracted from your response

Your response should be factual, comprehensive, and demonstrate deep understanding of the topic.

RESPONSE:"""
            
            response = await engine.generate_response(prompt)
            
            # Extract claims from the response
            extracted_claims = await self.claims_extractor.extract_claims_from_response(
                response, engine_name, 1, engine
            )
            
            return CrossPollinationResult(
                engine=engine_name,
                response=response,
                extracted_claims=extracted_claims,
                round_idx=1
            )
            
        except Exception as e:
            logger.error(f"Error generating initial response for {engine_name}: {e}")
            raise
    
    async def _generate_cross_pollinated_response(
        self, 
        query: str, 
        engine_name: str, 
        engine: AIEngine,
        claims_context: str,
        round_idx: int
    ) -> CrossPollinationResult:
        """Generate cross-pollinated response based on all extracted claims."""
        try:
            prompt = f"""You are participating in a multi-AI consensus process. You have access to claims from all AI engines. Your task is to:

1. Analyze all the claims from other engines
2. Synthesize the best elements while addressing gaps
3. Provide an improved response that builds on all insights
4. Extract and rank new claims from your improved response

ORIGINAL QUESTION: {query}

{claims_context}

CROSS-POLLINATION TASK:
- Critically analyze each engine's claims
- Identify strengths, weaknesses, and gaps
- Synthesize the best elements from all engines
- Add new insights and improvements
- Provide a MORE COMPREHENSIVE and ACCURATE response

Your improved response should demonstrate:
- Deeper understanding than individual engines
- Synthesis of multiple perspectives
- Addressing of limitations in other claims
- New insights not present in original claims

IMPROVED RESPONSE:"""
            
            response = await engine.generate_response(prompt)
            
            # Extract claims from the improved response
            extracted_claims = await self.claims_extractor.extract_claims_from_response(
                response, engine_name, round_idx, engine
            )
            
            return CrossPollinationResult(
                engine=engine_name,
                response=response,
                extracted_claims=extracted_claims,
                round_idx=round_idx
            )
            
        except Exception as e:
            logger.error(f"Error generating cross-pollinated response for {engine_name}: {e}")
            raise
    
    def get_all_claims(self) -> List[ExtractedClaim]:
        """Get all extracted claims from all rounds."""
        return self.claims_history.copy()
    
    def get_claims_by_engine(self, engine: str) -> List[ExtractedClaim]:
        """Get all claims from a specific engine."""
        return [claim for claim in self.claims_history if claim.engine == engine]
    
    def get_top_claims(self, limit: int = 10) -> List[ExtractedClaim]:
        """Get top claims by confidence score."""
        return sorted(
            self.claims_history, 
            key=lambda x: x.confidence, 
            reverse=True
        )[:limit]
    
    def get_claims_by_round(self, round_idx: int) -> List[ExtractedClaim]:
        """Get all claims from a specific round."""
        return [claim for claim in self.claims_history if claim.round_idx == round_idx]
    
    def calculate_claims_convergence(self) -> float:
        """Calculate convergence score based on claim similarity."""
        if len(self.claims_history) < 2:
            return 0.0
        
        # Get top claims by confidence
        top_claims = self.get_top_claims(5)
        
        if len(top_claims) < 2:
            return 0.0
        
        # Calculate average similarity between top claims
        similarities = []
        for i in range(len(top_claims)):
            for j in range(i + 1, len(top_claims)):
                similarity = self.claims_extractor._similarity(
                    self.claims_extractor._normalize_for_comparison(top_claims[i].text),
                    self.claims_extractor._normalize_for_comparison(top_claims[j].text)
                )
                similarities.append(similarity)
        
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def reset(self):
        """Reset the cross-pollination state."""
        self.claims_history = []
