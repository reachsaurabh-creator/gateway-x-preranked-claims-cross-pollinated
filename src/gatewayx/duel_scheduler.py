"""
Duel Scheduler - Manages dueling between AI models
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
import time
import random

logger = logging.getLogger(__name__)


class DuelScheduler:
    """Schedules and manages duels between AI models"""
    
    def __init__(self):
        self.active_duels = {}
        self.duel_history = []
        self.duel_count = 0
    
    async def schedule_duel(self, query: str, engines: List[str], 
                          budget: int = 20) -> str:
        """Schedule a new duel"""
        duel_id = f"duel_{self.duel_count}_{int(time.time())}"
        self.duel_count += 1
        
        duel = {
            "id": duel_id,
            "query": query,
            "engines": engines,
            "budget": budget,
            "status": "scheduled",
            "created_at": time.time(),
            "results": []
        }
        
        self.active_duels[duel_id] = duel
        return duel_id
    
    async def execute_duel(self, duel_id: str, engine_pool) -> Dict[str, Any]:
        """Execute a scheduled duel"""
        if duel_id not in self.active_duels:
            raise ValueError(f"Duel {duel_id} not found")
        
        duel = self.active_duels[duel_id]
        duel["status"] = "running"
        
        try:
            # Generate responses from all engines
            responses = await engine_pool.generate_multiple_responses(
                duel["query"], duel["engines"]
            )
            
            # Evaluate responses against each other
            evaluations = await self._evaluate_responses(
                duel["query"], responses
            )
            
            # Determine winner
            winner = self._determine_winner(evaluations)
            
            result = {
                "duel_id": duel_id,
                "query": duel["query"],
                "responses": responses,
                "evaluations": evaluations,
                "winner": winner,
                "completed_at": time.time()
            }
            
            duel["results"] = result
            duel["status"] = "completed"
            
            # Move to history
            self.duel_history.append(duel)
            del self.active_duels[duel_id]
            
            return result
            
        except Exception as e:
            logger.error(f"Duel {duel_id} failed: {e}")
            duel["status"] = "failed"
            duel["error"] = str(e)
            raise
    
    async def _evaluate_responses(self, query: str, 
                                responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate responses against each other"""
        evaluations = []
        
        for i, response1 in enumerate(responses):
            for j, response2 in enumerate(responses):
                if i != j:
                    evaluation = await self._compare_responses(
                        query, response1, response2
                    )
                    evaluations.append(evaluation)
        
        return evaluations
    
    async def _compare_responses(self, query: str, response1: Dict[str, Any], 
                               response2: Dict[str, Any]) -> Dict[str, Any]:
        """Compare two responses"""
        # Simple comparison logic
        # In a real implementation, this would use an LLM judge
        
        score1 = self._score_response(query, response1)
        score2 = self._score_response(query, response2)
        
        if score1 > score2:
            winner = response1["engine"]
            margin = score1 - score2
        elif score2 > score1:
            winner = response2["engine"]
            margin = score2 - score1
        else:
            winner = "tie"
            margin = 0
        
        return {
            "response1": response1["engine"],
            "response2": response2["engine"],
            "winner": winner,
            "margin": margin,
            "score1": score1,
            "score2": score2
        }
    
    def _score_response(self, query: str, response: Dict[str, Any]) -> float:
        """Score a response based on various factors"""
        text = response["text"]
        
        # Length score
        length_score = min(len(text.split()) / 50, 1.0)
        
        # Relevance score (keyword overlap)
        query_words = set(query.lower().split())
        response_words = set(text.lower().split())
        overlap = len(query_words.intersection(response_words))
        relevance_score = overlap / max(len(query_words), 1)
        
        # Structure score
        sentences = text.count('.') + text.count('!') + text.count('?')
        structure_score = min(sentences / 3, 1.0)
        
        # Combined score
        total_score = (
            0.4 * length_score +
            0.4 * relevance_score +
            0.2 * structure_score
        )
        
        return total_score
    
    def _determine_winner(self, evaluations: List[Dict[str, Any]]) -> str:
        """Determine overall winner from evaluations"""
        engine_scores = {}
        
        for eval in evaluations:
            if eval["winner"] != "tie":
                engine = eval["winner"]
                if engine not in engine_scores:
                    engine_scores[engine] = 0
                engine_scores[engine] += eval["margin"]
        
        if not engine_scores:
            return "tie"
        
        return max(engine_scores.items(), key=lambda x: x[1])[0]
    
    def get_duel_stats(self) -> Dict[str, Any]:
        """Get duel statistics"""
        completed_duels = [d for d in self.duel_history if d["status"] == "completed"]
        
        return {
            "total_duels": len(self.duel_history),
            "active_duels": len(self.active_duels),
            "completed_duels": len(completed_duels),
            "failed_duels": len([d for d in self.duel_history if d["status"] == "failed"])
        }
