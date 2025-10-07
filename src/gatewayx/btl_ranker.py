"""
Bradley-Terry-Luce (BTL) ranking system for response evaluation
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Any, Optional
import random

logger = logging.getLogger(__name__)


class BTLRanker:
    """Bradley-Terry-Luce ranking system for consensus building"""
    
    def __init__(self):
        self.comparison_cache = {}
        self.ranking_history = []
    
    async def rank_responses(self, query: str, responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Rank responses using BTL model"""
        if len(responses) < 2:
            return {
                "best_response": responses[0] if responses else None,
                "confidence": 1.0,
                "rankings": [responses[0]] if responses else [],
                "btl_scores": [1.0] if responses else []
            }
        
        # Store query and responses for confidence calculation
        self._current_query = query
        self._current_responses = responses
        
        # Generate pairwise comparisons
        comparisons = await self._generate_comparisons(query, responses)
        
        # Calculate BTL scores
        btl_scores = self._calculate_btl_scores(comparisons, len(responses))
        
        # Rank responses by BTL scores
        ranked_responses = sorted(
            zip(responses, btl_scores), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        best_response, best_score = ranked_responses[0]
        confidence = self._calculate_confidence(btl_scores)
        
        result = {
            "best_response": best_response,
            "confidence": confidence,
            "rankings": [r[0] for r in ranked_responses],
            "btl_scores": btl_scores,
            "comparisons": comparisons,
            "separation": max(btl_scores) - min(btl_scores) if len(btl_scores) > 1 else 0
        }
        
        # Cache result
        self.ranking_history.append(result)
        if len(self.ranking_history) > 100:  # Keep only recent history
            self.ranking_history = self.ranking_history[-100:]
        
        return result
    
    async def _generate_comparisons(self, query: str, responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate pairwise comparisons between responses"""
        comparisons = []
        n = len(responses)
        
        # Generate all possible pairs
        for i in range(n):
            for j in range(i + 1, n):
                comparison = await self._compare_responses(
                    query, responses[i], responses[j]
                )
                comparisons.append(comparison)
        
        return comparisons
    
    async def _compare_responses(self, query: str, response1: Dict[str, Any], 
                               response2: Dict[str, Any]) -> Dict[str, Any]:
        """Compare two responses and determine winner"""
        # Simple heuristic-based comparison
        # In a real implementation, this would use an LLM judge
        
        # Factors to consider:
        # 1. Length (longer responses often more detailed)
        # 2. Specificity (more specific terms)
        # 3. Structure (well-formatted responses)
        # 4. Relevance (keyword matching with query)
        
        score1 = self._score_response(query, response1)
        score2 = self._score_response(query, response2)
        
        if score1 > score2:
            winner = 1
            confidence = (score1 - score2) / max(score1, score2)
        elif score2 > score1:
            winner = 2
            confidence = (score2 - score1) / max(score1, score2)
        else:
            winner = random.choice([1, 2])
            confidence = 0.1
        
        return {
            "response1": response1,
            "response2": response2,
            "winner": winner,
            "confidence": confidence,
            "score1": score1,
            "score2": score2
        }
    
    def _score_response(self, query: str, response: Dict[str, Any]) -> float:
        """Score a response based on various factors"""
        text = response["text"]
        
        # Check if this is a clarifying question (penalize heavily)
        if self._is_clarifying_question(text):
            logger.warning(f"Clarifying question detected for engine {response['engine']}: {text[:100]}...")
            return 0.1  # Very low score for clarifying questions
        
        # Length score (normalized)
        length_score = min(len(text.split()) / 100, 1.0)
        
        # Specificity score (unique words)
        words = text.lower().split()
        unique_words = len(set(words))
        specificity_score = unique_words / max(len(words), 1)
        
        # Structure score (sentences, paragraphs)
        sentences = text.count('.') + text.count('!') + text.count('?')
        structure_score = min(sentences / 5, 1.0)
        
        # Relevance score (keyword overlap with query)
        query_words = set(query.lower().split())
        response_words = set(words)
        overlap = len(query_words.intersection(response_words))
        relevance_score = overlap / max(len(query_words), 1)
        
        # Weighted combination
        total_score = (
            0.3 * length_score +
            0.3 * specificity_score +
            0.2 * structure_score +
            0.2 * relevance_score
        )
        
        logger.debug(f"Engine {response['engine']} score: {total_score:.3f} (length: {length_score:.3f}, specificity: {specificity_score:.3f}, structure: {structure_score:.3f}, relevance: {relevance_score:.3f})")
        return total_score
    
    def _is_clarifying_question(self, text: str) -> bool:
        """Check if response is a clarifying question rather than an answer"""
        text_lower = text.lower()
        
        # Skip if this looks like a re-query response (contains the re-query prompt)
        if "please provide a direct answer to this question" in text_lower:
            return False
        
        # Common clarifying question patterns
        clarifying_patterns = [
            "could you clarify",
            "could you provide more context",
            "i'd need more context",
            "could you be more specific",
            "what do you mean by",
            "are you asking about",
            "do you want to know",
            "are you referring to",
            "which aspect",
            "what specific",
            "could you clarify what",
            "i'd need more information",
            "can you provide more details",
            "what exactly",
            "which part",
            "what kind of",
            "what type of",
            "regarding your question",
            "about your question",
            "your question about",
            "i'd need more context to provide",
            "could you provide more details about",
            "what specific aspect",
            "which specific",
            "what do you mean",
            "are you asking",
            "do you want",
            "are you referring",
            "what exactly are you asking",
            "which part of",
            "what kind of information",
            "what type of information",
            "grok here! regarding your question",
            "i'd need more context to provide a comprehensive answer",
            "could you clarify what specific aspect",
            "what specific aspect are you most interested in"
        ]
        
        # Check if text contains clarifying patterns
        for pattern in clarifying_patterns:
            if pattern in text_lower:
                return True
        
        # Check if response is very short and contains question marks
        if len(text.split()) < 50 and text.count('?') > 0:
            return True
        
        # Check if response starts with common clarifying phrases
        first_sentence = text.split('.')[0].lower()
        if any(phrase in first_sentence for phrase in [
            "could you", "can you", "would you", "do you", "are you",
            "what", "which", "how", "when", "where", "why", "grok here"
        ]):
            return True
        
        # Check if the response is asking for clarification rather than providing an answer
        if any(phrase in text_lower for phrase in [
            "need more context", "need more information", "need more details",
            "could you clarify", "could you specify", "could you elaborate",
            "what do you mean", "what exactly", "which specific",
            "comprehensive answer", "specific aspect", "most interested in"
        ]):
            return True
        
        # Check for Grok-specific patterns
        if "grok here!" in text_lower and ("context" in text_lower or "clarify" in text_lower):
            return True
        
        # Check for any response that mentions the re-query prompt
        if "answer this question directly" in text_lower and ("context" in text_lower or "clarify" in text_lower):
            return True
        
        # Check for Grok responses that are still asking for clarification
        if "grok here!" in text_lower and ("context" in text_lower or "clarify" in text_lower or "comprehensive answer" in text_lower):
            return True
        
        return False
    
    def _calculate_btl_scores(self, comparisons: List[Dict[str, Any]], 
                            num_responses: int) -> List[float]:
        """Calculate BTL scores from comparisons"""
        if not comparisons:
            return [1.0] * num_responses
        
        # Initialize scores
        scores = [1.0] * num_responses
        
        # Iterative BTL algorithm
        for iteration in range(10):  # Max 10 iterations
            old_scores = scores.copy()
            
            # Update scores based on comparisons
            for comparison in comparisons:
                winner = comparison["winner"]
                confidence = comparison["confidence"]
                
                if winner == 1:
                    idx1, idx2 = 0, 1  # Simplified indexing
                else:
                    idx1, idx2 = 1, 0
                
                # BTL update rule
                prob_win = scores[idx1] / (scores[idx1] + scores[idx2])
                error = confidence - prob_win
                
                scores[idx1] += 0.1 * error * scores[idx1]
                scores[idx2] -= 0.1 * error * scores[idx2]
                
                # Ensure non-negative scores
                scores[idx1] = max(scores[idx1], 0.01)
                scores[idx2] = max(scores[idx2], 0.01)
            
            # Check convergence
            if max(abs(scores[i] - old_scores[i]) for i in range(num_responses)) < 0.01:
                break
        
        # Normalize scores
        total = sum(scores)
        if total > 0:
            scores = [s / total for s in scores]
        
        return scores
    
    def _calculate_confidence(self, btl_scores: List[float]) -> float:
        """Calculate confidence based on BTL scores and response agreement"""
        if len(btl_scores) < 2:
            return 1.0
        
        # Get the responses to analyze agreement
        responses = getattr(self, '_current_responses', [])
        query = getattr(self, '_current_query', "")
        
        # Check if this is a subjective question
        is_subjective = self._is_subjective_question(query, responses)
        
        if is_subjective:
            # For subjective questions, use much lower confidence
            # Calculate BTL-based confidence (separation between top scores)
            sorted_scores = sorted(btl_scores, reverse=True)
            separation = sorted_scores[0] - sorted_scores[1]
            btl_confidence = min(separation * 1.5, 0.6)  # Cap at 60% for subjective
            
            # Calculate agreement-based confidence (also lower for subjective)
            agreement_confidence = self._calculate_agreement_confidence(responses)
            agreement_confidence = min(agreement_confidence * 0.7, 0.5)  # Cap at 50% for subjective
            
            # Use the higher of the two, but cap overall confidence for subjective questions
            confidence = max(agreement_confidence, btl_confidence)
            return max(confidence, 0.1)  # Minimum confidence
        else:
            # For factual questions, use original logic
            agreement_confidence = self._calculate_agreement_confidence(responses)
            
            # Calculate BTL-based confidence (separation between top scores)
            sorted_scores = sorted(btl_scores, reverse=True)
            separation = sorted_scores[0] - sorted_scores[1]
            btl_confidence = min(separation * 2, 1.0)
            
            # Use the higher of the two confidence measures
            confidence = max(agreement_confidence, btl_confidence)
            
            return max(confidence, 0.1)  # Minimum confidence
    
    def _calculate_agreement_confidence(self, responses: List[Dict[str, Any]]) -> float:
        """Calculate confidence based on response agreement"""
        if len(responses) < 2:
            return 1.0
        
        # Extract response texts
        texts = [r.get("text", "").strip().lower() for r in responses]
        
        # Check for mathematical questions using the original query
        query = getattr(self, '_current_query', texts[0] if texts else "")
        if self._is_math_question(query):
            return self._calculate_math_confidence(texts)
        
        # Check for factual questions (short, direct answers)
        if self._is_factual_question(texts):
            return self._calculate_factual_confidence(texts)
        
        # For other questions, use similarity-based confidence
        return self._calculate_similarity_confidence(texts)
    
    def _is_subjective_question(self, query: str, responses: List[Dict[str, Any]]) -> bool:
        """Check if this is a subjective question"""
        query_lower = query.lower()
        
        # Subjective question indicators
        subjective_indicators = [
            'best', 'worst', 'favorite', 'prefer', 'opinion', 'think', 'believe',
            'should', 'would', 'could', 'might', 'may', 'recommend', 'suggest',
            'better', 'worse', 'good', 'bad', 'great', 'terrible', 'amazing',
            'beautiful', 'ugly', 'interesting', 'boring', 'fun', 'exciting',
            'emperor', 'king', 'queen', 'leader', 'president', 'ruler',
            'who was the', 'which is the', 'what is the best', 'what is the worst'
        ]
        
        # Check if query contains subjective indicators
        has_subjective_indicators = any(indicator in query_lower for indicator in subjective_indicators)
        
        # Check response diversity - subjective questions often have diverse responses
        if len(responses) >= 2:
            texts = [r.get("text", "").strip().lower() for r in responses]
            # Calculate average similarity between responses
            from difflib import SequenceMatcher
            similarities = []
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    similarity = SequenceMatcher(None, texts[i], texts[j]).ratio()
                    similarities.append(similarity)
            
            if similarities:
                avg_similarity = sum(similarities) / len(similarities)
                # If responses are very different (low similarity), likely subjective
                has_diverse_responses = avg_similarity < 0.3
            else:
                has_diverse_responses = False
        else:
            has_diverse_responses = False
        
        return has_subjective_indicators or has_diverse_responses

    def _is_math_question(self, query: str) -> bool:
        """Check if this is a mathematical question"""
        math_indicators = ['*', '+', '-', '/', '=', 'plus', 'minus', 'times', 'divided', 'equals', 'sum', 'product']
        return any(indicator in query for indicator in math_indicators)
    
    def _calculate_math_confidence(self, texts: List[str]) -> float:
        """Calculate confidence for mathematical questions"""
        import re
        
        # Filter out mock responses that don't contain actual answers
        valid_responses = []
        for text in texts:
            # Skip responses that look like mock/placeholder responses
            if not any(phrase in text.lower() for phrase in [
                'mock response', 'simulated', 'in production', 'this would call',
                'placeholder', 'example response', 'grok response'
            ]):
                valid_responses.append(text)
        
        # If we have valid responses, use them; otherwise use all responses
        texts_to_analyze = valid_responses if valid_responses else texts
        
        # Extract numbers from responses
        numbers = []
        for text in texts_to_analyze:
            # Find all numbers in the response
            nums = re.findall(r'\d+', text)
            numbers.append(set(nums))
        
        # Check if all responses contain the same answer
        if len(numbers) >= 2:
            # Count how many responses have the same numbers as the first response
            matching_responses = sum(1 for nums in numbers if nums == numbers[0])
            
            # If all valid responses agree, perfect confidence
            if matching_responses == len(numbers):
                return 1.0  # 100% confidence for perfect math agreement
            # If most responses agree (excluding mock responses), very high confidence
            elif matching_responses >= len(numbers) * 0.7:
                return 0.95
            # If at least 2 responses agree, high confidence
            elif matching_responses >= 2:
                return 0.85
        
        # For math questions, even with disagreement, give reasonable confidence
        # if the responses contain numbers (suggesting they're attempting to answer)
        if any(len(nums) > 0 for nums in numbers):
            return 0.6  # Medium confidence for math questions with some numbers
        
        return 0.4  # Lower confidence if no numbers found
    
    def _is_factual_question(self, texts: List[str]) -> bool:
        """Check if this is a factual question (short, direct answers)"""
        # Factual questions typically have short, similar responses
        avg_length = sum(len(text) for text in texts) / len(texts)
        return avg_length < 100  # Short responses suggest factual questions
    
    def _calculate_factual_confidence(self, texts: List[str]) -> float:
        """Calculate confidence for factual questions"""
        # Check similarity of responses
        from difflib import SequenceMatcher
        
        similarities = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                similarity = SequenceMatcher(None, texts[i], texts[j]).ratio()
                similarities.append(similarity)
        
        if similarities:
            avg_similarity = sum(similarities) / len(similarities)
            # High similarity = high confidence for factual questions
            return min(avg_similarity * 1.2, 0.95)
        
        return 0.7
    
    def _calculate_similarity_confidence(self, texts: List[str]) -> float:
        """Calculate confidence based on text similarity"""
        from difflib import SequenceMatcher
        
        similarities = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                similarity = SequenceMatcher(None, texts[i], texts[j]).ratio()
                similarities.append(similarity)
        
        if similarities:
            avg_similarity = sum(similarities) / len(similarities)
            return min(avg_similarity * 1.1, 0.9)
        
        return 0.5
    
    def get_ranking_stats(self) -> Dict[str, Any]:
        """Get ranking statistics"""
        if not self.ranking_history:
            return {"total_rankings": 0}
        
        confidences = [r["confidence"] for r in self.ranking_history]
        separations = [r["separation"] for r in self.ranking_history]
        
        return {
            "total_rankings": len(self.ranking_history),
            "average_confidence": np.mean(confidences),
            "average_separation": np.mean(separations),
            "max_confidence": max(confidences),
            "min_confidence": min(confidences)
        }
