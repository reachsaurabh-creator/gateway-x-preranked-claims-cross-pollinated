"""Prompt management and validation for Gateway X."""

import json
import random
from typing import Dict, Any


class PromptVault:
    """Centralized prompt management with strict JSON validation."""
    
    @staticmethod
    def duel_prompt(query: str, a: str, b: str) -> str:
        """Generate a critical duel prompt for comparing two answers."""
        return (
            "You are an expert referee in a multi-AI consensus process. Your role is to CRITICALLY ANALYZE and compare two answers.\n\n"
            "EVALUATION CRITERIA:\n"
            "1. FACTUAL ACCURACY: Are the claims correct and well-supported?\n"
            "2. COMPREHENSIVENESS: Does the answer cover all important aspects?\n"
            "3. COHERENCE: Is the logic clear and well-structured?\n"
            "4. INSIGHTFULNESS: Does it provide valuable insights or just surface-level information?\n"
            "5. RELEVANCE: How well does it address the specific question?\n\n"
            "CRITICAL ANALYSIS TASK:\n"
            "- Identify specific strengths and weaknesses in each answer\n"
            "- Consider which answer better addresses the question\n"
            "- Look for factual errors, logical gaps, or missing elements\n"
            "- Determine which answer would be more useful to someone seeking the truth\n\n"
            'Return STRICT JSON only: {"winner":"A"|"B","factuality":0..1,"coherence":0..1,"note":"<=30 tokens"}\n\n'
            f"QUERY: {query}\n\n"
            f"ANSWER A: {a}\n\n"
            f"ANSWER B: {b}\n\n"
            "CRITICAL EVALUATION:"
        )

    @staticmethod
    def validate_response(response: str) -> Dict[str, Any]:
        """Validate and parse LLM response, with fallback for invalid JSON."""
        try:
            obj = json.loads(response)
            if obj.get("winner") not in ("A", "B"):
                raise ValueError("invalid winner")
            return {
                "winner": obj["winner"],
                "factuality": float(obj.get("factuality", 0.5)),
                "coherence": float(obj.get("coherence", 0.5)),
                "note": str(obj.get("note", ""))[:60],
            }
        except Exception:
            # Fallback to random decision if JSON parsing fails
            return {
                "winner": random.choice(["A", "B"]), 
                "factuality": 0.5, 
                "coherence": 0.5, 
                "note": "fallback"
            }
