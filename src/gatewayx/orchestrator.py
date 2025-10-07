"""
Orchestrator - Main coordination logic for consensus building
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
import time
import random

from .engine_pool import EnginePool
from .btl_ranker import BTLRanker
from .playbook_selector import PlaybookSelector
from .settings import settings

logger = logging.getLogger(__name__)


class Orchestrator:
    """Main orchestrator for consensus building"""
    
    def __init__(self, engine_pool: EnginePool):
        self.engine_pool = engine_pool
        self.btl_ranker = BTLRanker()
        self.playbook_selector = PlaybookSelector()
        self.query_count = 0
        self.total_processing_time = 0.0
        self.excluded_engines = set()  # Track engines that failed to provide direct answers
    
    async def process_query(self, query: str, budget: int = None, 
                          confidence_threshold: float = None,
                          engines: Optional[List[str]] = None) -> Dict[str, Any]:
        """Process a query and build consensus"""
        start_time = time.time()
        self.query_count += 1
        
        budget = budget or settings.default_budget
        confidence_threshold = confidence_threshold or settings.confidence_threshold
        
        logger.info(f"Processing query {self.query_count}: {query[:100]}...")
        
        try:
            logger.info(f"Multi-engine mode: {settings.multi_engine_mode}, Consensus enabled: {settings.enable_consensus_judging}")
            logger.info(f"Settings loaded from: {settings.Config.env_file}")
            if settings.multi_engine_mode and settings.enable_consensus_judging:
                logger.info("Using _process_with_consensus method")
                try:
                    result = await self._process_with_consensus(
                        query, budget, confidence_threshold, engines
                    )
                    logger.info(f"_process_with_consensus completed successfully")
                except Exception as e:
                    logger.error(f"_process_with_consensus failed: {e}")
                    logger.info("Falling back to _process_single_engine method")
                    result = await self._process_single_engine(
                        query, budget, engines
                    )
            else:
                logger.info("Using _process_single_engine method")
                result = await self._process_single_engine(
                    query, budget, engines
                )
            
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            
            result.update({
                "query_id": self.query_count,
                "processing_time": processing_time,
                "metadata": {
                    "budget_used": budget,
                    "confidence_threshold": confidence_threshold,
                    "multi_engine_mode": settings.multi_engine_mode,
                    "consensus_enabled": settings.enable_consensus_judging
                }
            })
            
            logger.info(f"Query {self.query_count} completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Query {self.query_count} failed: {e}")
            raise
    
    async def _process_single_engine(self, query: str, budget: int,
                                   engines: Optional[List[str]] = None) -> Dict[str, Any]:
        """Process query with all available engines"""
        # Get all available engines
        if engines:
            available_engines = [e for e in engines if e in self.engine_pool.engines]
        else:
            available_engines = self.engine_pool.get_available_engines()
        
        if not available_engines:
            raise RuntimeError("No available engines")
        
        # Generate responses from all available engines
        try:
            all_responses = await self.engine_pool.generate_multiple_responses(query, available_engines)
            logger.info(f"Generated {len(all_responses)} responses")
            for i, resp in enumerate(all_responses):
                logger.info(f"Response {i}: {type(resp)} - {resp.keys() if isinstance(resp, dict) else 'Not a dict'}")
        except Exception as e:
            logger.error(f"Failed to generate responses: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Use BTL ranking to find the best response
        if len(all_responses) > 1:
            try:
                ranking_result = await self.btl_ranker.rank_responses(query, all_responses)
                best_response = ranking_result["best_response"]
                confidence = ranking_result["confidence"]
                consensus_data = ranking_result
            except Exception as e:
                logger.error(f"BTL ranking failed: {e}")
                import traceback
                traceback.print_exc()
                # Fallback to simple selection
                best_response = max(all_responses, key=lambda r: len(r["text"]))
                confidence = 0.8
                consensus_data = None
        else:
            best_response = all_responses[0] if all_responses else None
            confidence = 1.0
            consensus_data = None
        
        # Create detailed round data for single engine mode
        detailed_rounds = []
        if all_responses:
            round_data = await self._create_round_data(
                round_number=1,
                responses=all_responses,
                query=query,
                confidence_threshold=0.95  # Default threshold for single engine
            )
            detailed_rounds.append(round_data)
        
        return {
            "best_claim": best_response["text"] if best_response else "No response generated",
            "confidence": confidence,
            "rounds_used": 1,
            "engines_used": [r["engine"] for r in all_responses],
            "total_cost": sum(r["cost"] for r in all_responses),
            "responses": all_responses,
            "consensus_data": consensus_data,
            "detailed_rounds": detailed_rounds
        }
    
    async def _process_with_consensus(self, query: str, budget: int,
                                    confidence_threshold: float,
                                    engines: Optional[List[str]] = None) -> Dict[str, Any]:
        """Process query with multi-engine consensus"""
        rounds_used = 0
        total_cost = 0.0
        all_responses = []
        engines_used = set()
        detailed_rounds = []  # Track detailed round data
        
        # Determine which engines to use
        if engines:
            available_engines = [e for e in engines if e in self.engine_pool.engines]
        else:
            available_engines = self.engine_pool.get_available_engines()
        
        if not available_engines:
            raise RuntimeError("No available engines")
        
        # Generate initial responses from all available engines
        initial_responses = await self.engine_pool.generate_multiple_responses(
            query, available_engines
        )
        
        # Handle clarifying questions by re-querying those engines
        initial_responses = await self._handle_clarifying_questions(
            query, initial_responses, available_engines
        )
        
        rounds_used += 1
        all_responses.extend(initial_responses)
        engines_used.update(r["engine"] for r in initial_responses)
        total_cost += sum(r["cost"] for r in initial_responses)
        
        # Don't create round data here - we'll create it after consensus building
        # to ensure we use the same responses and confidence calculation
        
        # If we have enough responses, try to build consensus
        if len(initial_responses) >= 2:
            logger.info(f"Building consensus with {len(initial_responses)} responses, threshold: {confidence_threshold:.3f}")
            consensus_result = await self._build_consensus(
                query, initial_responses, budget - total_cost, confidence_threshold
            )
            
            if consensus_result:
                additional_rounds = consensus_result.get("additional_rounds", 0)
                additional_responses = consensus_result.get("additional_responses", [])
                
                rounds_used += additional_rounds
                all_responses.extend(additional_responses)
                engines_used.update(consensus_result.get("additional_engines", []))
                total_cost += consensus_result.get("additional_cost", 0.0)
                
                # Create detailed round data after consensus building
                # Use the final confidence from consensus_result for consistency
                final_confidence = consensus_result["confidence"]
                
                # Create detailed round data using the proper method
                round_1_data = await self._create_round_data(
                    round_number=1,
                    responses=all_responses,
                    query=query,
                    confidence_threshold=confidence_threshold
                )
                detailed_rounds.append(round_1_data)
                
                # Round 2: Additional responses if any
                if additional_responses:
                    round_2_data = await self._create_round_data(
                        round_number=2,
                        responses=additional_responses,
                        query=query,
                        confidence_threshold=confidence_threshold
                    )
                    detailed_rounds.append(round_2_data)
                
                return {
                    "best_claim": consensus_result["best_claim"],
                    "confidence": consensus_result["confidence"],
                    "rounds_used": rounds_used,
                    "engines_used": list(engines_used),
                    "total_cost": total_cost,
                    "responses": all_responses,
                    "consensus_data": consensus_result.get("consensus_data", {}),
                    "detailed_rounds": detailed_rounds
                }
        
        # Fallback to best single response - use BTL ranking for proper confidence
        try:
            ranking_result = await self.btl_ranker.rank_responses(query, all_responses)
            best_response = ranking_result["best_response"]
            actual_confidence = ranking_result["confidence"]
        except Exception as e:
            logger.error(f"BTL ranking failed in fallback: {e}")
            best_response = max(initial_responses, key=lambda r: len(r["text"]))
            actual_confidence = 0.5  # Default confidence for fallback
        
        # Create detailed round data for fallback case
        if not detailed_rounds:
            round_data = await self._create_round_data(
                round_number=1,
                responses=all_responses,
                query=query,
                confidence_threshold=confidence_threshold
            )
            detailed_rounds.append(round_data)
        
        return {
            "best_claim": best_response["text"],
            "confidence": actual_confidence,  # Use actual BTL confidence
            "rounds_used": rounds_used,
            "engines_used": list(engines_used),
            "total_cost": total_cost,
            "responses": all_responses,
            "detailed_rounds": detailed_rounds
        }
    
    async def _build_consensus(self, query: str, responses: List[Dict[str, Any]],
                             remaining_budget: float, confidence_threshold: float) -> Optional[Dict[str, Any]]:
        """Build consensus from responses using BTL ranking"""
        try:
            # Use BTL ranker to evaluate responses
            ranking_result = await self.btl_ranker.rank_responses(query, responses)
            
            logger.info(f"Initial confidence: {ranking_result['confidence']:.3f}, threshold: {confidence_threshold:.3f}")
            if ranking_result["confidence"] >= confidence_threshold:
                logger.info(f"Confidence {ranking_result['confidence']:.3f} meets threshold {confidence_threshold:.3f}, stopping at 1 round")
                return {
                    "best_claim": ranking_result["best_response"]["text"],
                    "confidence": ranking_result["confidence"],
                    "consensus_data": ranking_result,
                    "additional_rounds": 0,
                    "additional_responses": [],
                    "additional_engines": [],
                    "additional_cost": 0.0
                }
            
            # If confidence is too low, try to get more responses
            if remaining_budget > 0 and len(responses) < settings.max_budget:
                additional_responses = await self._get_additional_responses(
                    query, remaining_budget, responses
                )
                
                if additional_responses:
                    all_responses = responses + additional_responses
                    new_ranking = await self.btl_ranker.rank_responses(query, all_responses)
                    
                    # Check if the new confidence meets the threshold
                    if new_ranking["confidence"] >= confidence_threshold:
                        return {
                            "best_claim": new_ranking["best_response"]["text"],
                            "confidence": new_ranking["confidence"],
                            "additional_rounds": 1,
                            "additional_responses": additional_responses,
                            "additional_engines": [r["engine"] for r in additional_responses],
                            "additional_cost": sum(r["cost"] for r in additional_responses),
                            "consensus_data": new_ranking
                        }
                    else:
                        # Still below threshold, but return what we have
                        logger.info(f"Confidence {new_ranking['confidence']:.3f} still below threshold {confidence_threshold:.3f}")
                        return {
                            "best_claim": new_ranking["best_response"]["text"],
                            "confidence": new_ranking["confidence"],
                            "additional_rounds": 1,
                            "additional_responses": additional_responses,
                            "additional_engines": [r["engine"] for r in additional_responses],
                            "additional_cost": sum(r["cost"] for r in additional_responses),
                            "consensus_data": new_ranking
                        }
            
            return None
            
        except Exception as e:
            logger.error(f"Consensus building failed: {e}")
            return None
    
    async def _handle_clarifying_questions(self, query: str, responses: List[Dict[str, Any]], 
                                         available_engines: List[str]) -> List[Dict[str, Any]]:
        """Handle engines that returned clarifying questions by re-querying them"""
        updated_responses = []
        excluded_engines = set()  # Track engines that failed to provide direct answers
        
        for response in responses:
            # Check if this is a clarifying question
            if self.btl_ranker._is_clarifying_question(response["text"]):
                logger.info(f"Engine {response['engine']} returned clarifying question, re-querying...")
                
                # Try multiple re-query attempts with different prompts
                new_response = None
                for attempt in range(3):  # Try up to 3 times
                    if attempt == 0:
                        specific_prompt = f"""Answer this question directly: "{query}"

Do NOT ask for clarification. Do NOT ask for more context. Provide your best answer based on reasonable assumptions. If you need to make assumptions, state them briefly and then give your answer.

Example format:
"Assuming you want [specific assumption], here's my answer: [your answer]"

Question: {query}"""
                    elif attempt == 1:
                        specific_prompt = f"""Give me a direct answer to: "{query}"

Provide a specific recommendation with reasoning. Do not ask questions."""
                    else:
                        specific_prompt = f"""Question: {query}

Answer: [Provide a direct, specific answer without asking for clarification]"""
                    
                    try:
                        # Re-query the same engine with the specific prompt
                        new_response = await self.engine_pool.generate_response(
                            specific_prompt, response["engine"]
                        )
                        
                        # Check if the new response is also a clarifying question
                        if not self.btl_ranker._is_clarifying_question(new_response["text"]):
                            logger.info(f"Engine {response['engine']} provided direct answer after re-query (attempt {attempt + 1})")
                            break
                        else:
                            logger.info(f"Engine {response['engine']} still returned clarifying question (attempt {attempt + 1})")
                            new_response = None
                            
                    except Exception as e:
                        logger.warning(f"Failed to re-query {response['engine']} (attempt {attempt + 1}): {e}")
                        new_response = None
                
                # Use the new response if we got one, otherwise exclude the engine
                if new_response:
                    updated_responses.append(new_response)
                else:
                    logger.warning(f"Engine {response['engine']} failed to provide direct answer after 3 attempts, excluding from consensus")
                    excluded_engines.add(response["engine"])
                    # Don't include the original clarifying question response
            else:
                # Keep non-clarifying responses as-is
                updated_responses.append(response)
        
        # Store excluded engines for future rounds
        self.excluded_engines = excluded_engines
        return updated_responses

    async def _get_additional_responses(self, query: str, budget: float,
                                      existing_responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get additional responses within budget"""
        used_engines = {r["engine"] for r in existing_responses}
        available_engines = [
            name for name in self.engine_pool.get_available_engines()
            if name not in used_engines and name not in self.excluded_engines
        ]
        
        if not available_engines:
            return []
        
        # Estimate cost per response (rough approximation)
        estimated_cost_per_response = 0.01
        max_additional = min(
            len(available_engines),
            int(budget / estimated_cost_per_response),
            settings.batch_size
        )
        
        if max_additional <= 0:
            return []
        
        # Select engines for additional responses
        selected_engines = random.sample(available_engines, max_additional)
        
        try:
            additional_responses = await self.engine_pool.generate_multiple_responses(
                query, selected_engines
            )
            return additional_responses
        except Exception as e:
            logger.warning(f"Failed to get additional responses: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics"""
        avg_processing_time = (
            self.total_processing_time / self.query_count 
            if self.query_count > 0 else 0
        )
        
        return {
            "query_count": self.query_count,
            "total_processing_time": self.total_processing_time,
            "average_processing_time": avg_processing_time,
            "engine_pool_stats": self.engine_pool.get_pool_stats()
        }
    
    async def _create_round_data(self, round_number: int, responses: List[Dict[str, Any]], 
                                query: str, confidence_threshold: float) -> Dict[str, Any]:
        """Create detailed round data for frontend display"""
        try:
            # Use BTL ranking to get scores and rankings
            ranking_result = await self.btl_ranker.rank_responses(query, responses)
            
            # Convert responses to EngineResponse format
            engine_responses = []
            for i, response in enumerate(responses):
                # Get score and ranking from BTL result
                response_text = response.get("text", "")
                score = None
                ranking = None
                
                # Find matching response in BTL results
                if "btl_scores" in ranking_result:
                    btl_scores = ranking_result["btl_scores"]
                    if i < len(btl_scores):
                        score = btl_scores[i]
                
                # Calculate ranking based on BTL scores
                if "btl_scores" in ranking_result:
                    btl_scores = ranking_result["btl_scores"]
                    # Sort responses by score to get ranking
                    sorted_indices = sorted(range(len(btl_scores)), key=lambda x: btl_scores[x], reverse=True)
                    ranking = sorted_indices.index(i) + 1  # 1-based ranking
                
                engine_response = {
                    "engine": response.get("engine", "unknown"),
                    "text": response_text,
                    "cost": response.get("cost", 0.0),
                    "tokens": response.get("tokens", 0),
                    "response_time": response.get("response_time", 0.0),
                    "score": score,
                    "ranking": ranking if ranking is not None else None
                }
                engine_responses.append(engine_response)
            
            # Determine if threshold was met
            round_confidence = ranking_result.get("confidence", 0.0)
            threshold_met = round_confidence >= confidence_threshold
            
            return {
                "round_number": round_number,
                "engines_used": [r["engine"] for r in responses],
                "responses": engine_responses,
                "consensus_data": ranking_result,
                "round_confidence": round_confidence,
                "threshold_met": threshold_met
            }
            
        except Exception as e:
            logger.error(f"Failed to create round data: {e}")
            # Return basic round data without BTL scoring
            engine_responses = []
            for response in responses:
                engine_response = {
                    "engine": response.get("engine", "unknown"),
                    "text": response.get("text", ""),
                    "cost": response.get("cost", 0.0),
                    "tokens": response.get("tokens", 0),
                    "response_time": response.get("response_time", 0.0),
                    "score": None,
                    "ranking": None
                }
                engine_responses.append(engine_response)
            
            return {
                "round_number": round_number,
                "engines_used": [r["engine"] for r in responses],
                "responses": engine_responses,
                "consensus_data": None,
                "round_confidence": None,
                "threshold_met": False
            }
