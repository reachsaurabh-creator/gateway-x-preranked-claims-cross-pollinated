"""Claims extraction system for preranked claims cross-pollination algorithm."""

import json
import re
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger("gatewayx")


@dataclass
class ExtractedClaim:
    """Represents an extracted claim with metadata."""
    text: str
    confidence: float
    rank: int
    engine: str
    round_idx: int


class ClaimsExtractor:
    """Extracts and structures claims from AI engine responses."""
    
    def __init__(self):
        self.claim_patterns = [
            # Pattern 1: JSON format
            r'```json\s*(\{.*?\})\s*```',
            r'\{[^}]*"claims"[^}]*\}',
            
            # Pattern 2: Numbered list
            r'(\d+\.\s*[^\n]+(?:\n(?!\d+\.)[^\n]+)*)',
            
            # Pattern 3: Bullet points
            r'[-•]\s*([^\n]+(?:\n(?![-•])[^\n]+)*)',
            
            # Pattern 4: Paragraph-based (fallback)
            r'([A-Z][^.!?]*[.!?])',
        ]
    
    async def extract_claims_from_response(
        self, 
        response: str, 
        engine: str, 
        round_idx: int,
        ai_engine=None
    ) -> List[ExtractedClaim]:
        """Extract structured claims from an AI response."""
        try:
            # First, try to get structured claims using AI
            structured_claims = await self._extract_with_ai(
                response, engine, round_idx, ai_engine
            )
            if structured_claims:
                return structured_claims
            
            # Fallback to pattern-based extraction
            return self._extract_with_patterns(response, engine, round_idx)
            
        except Exception as e:
            logger.error(f"Error extracting claims from {engine}: {e}")
            # Ultimate fallback: treat entire response as single claim
            return [ExtractedClaim(
                text=response[:500] + "..." if len(response) > 500 else response,
                confidence=0.5,
                rank=1,
                engine=engine,
                round_idx=round_idx
            )]
    
    async def _extract_with_ai(
        self, 
        response: str, 
        engine: str, 
        round_idx: int,
        ai_engine
    ) -> Optional[List[ExtractedClaim]]:
        """Use AI to extract structured claims from response."""
        if not ai_engine:
            return None
            
        try:
            extraction_prompt = f"""Analyze the following response and extract the key claims/arguments in JSON format.

RESPONSE TO ANALYZE:
{response}

Extract 3-7 key claims from this response. Each claim should be:
1. A specific, factual statement or argument
2. Self-contained and meaningful
3. Ranked by importance/strength (1 = most important)
4. Given a confidence score (0.0 to 1.0)

Return ONLY valid JSON in this exact format:
{{
  "claims": [
    {{
      "text": "The specific claim text here",
      "confidence": 0.9,
      "rank": 1
    }},
    {{
      "text": "Another claim text here", 
      "confidence": 0.8,
      "rank": 2
    }}
  ]
}}"""
            
            extraction_response = await ai_engine.generate_response(extraction_prompt)
            
            # Try to parse JSON
            json_match = re.search(r'\{.*\}', extraction_response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                claims = []
                for i, claim_data in enumerate(data.get('claims', [])):
                    claims.append(ExtractedClaim(
                        text=claim_data.get('text', ''),
                        confidence=float(claim_data.get('confidence', 0.5)),
                        rank=int(claim_data.get('rank', i + 1)),
                        engine=engine,
                        round_idx=round_idx
                    ))
                return claims
                
        except Exception as e:
            logger.warning(f"AI-based extraction failed for {engine}: {e}")
        
        return None
    
    def _extract_with_patterns(
        self, 
        response: str, 
        engine: str, 
        round_idx: int
    ) -> List[ExtractedClaim]:
        """Extract claims using regex patterns."""
        claims = []
        
        # Try each pattern
        for pattern in self.claim_patterns:
            matches = re.findall(pattern, response, re.MULTILINE | re.DOTALL)
            if matches:
                for i, match in enumerate(matches[:7]):  # Limit to 7 claims
                    # Clean up the text
                    clean_text = self._clean_claim_text(match)
                    if len(clean_text) > 20:  # Minimum length
                        claims.append(ExtractedClaim(
                            text=clean_text,
                            confidence=0.7 - (i * 0.1),  # Decreasing confidence
                            rank=i + 1,
                            engine=engine,
                            round_idx=round_idx
                        ))
                break
        
        # If no patterns worked, split by sentences
        if not claims:
            sentences = re.split(r'[.!?]+', response)
            for i, sentence in enumerate(sentences[:5]):
                clean_text = sentence.strip()
                if len(clean_text) > 30:
                    claims.append(ExtractedClaim(
                        text=clean_text,
                        confidence=0.6 - (i * 0.1),
                        rank=i + 1,
                        engine=engine,
                        round_idx=round_idx
                    ))
        
        return claims[:7]  # Limit to 7 claims
    
    def _clean_claim_text(self, text: str) -> str:
        """Clean and normalize claim text."""
        # Remove markdown formatting
        text = re.sub(r'[*_`]', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove numbering
        text = re.sub(r'^\d+\.\s*', '', text)
        # Remove bullets
        text = re.sub(r'^[-•]\s*', '', text)
        return text
    
    def deduplicate_claims(self, claims: List[ExtractedClaim]) -> List[ExtractedClaim]:
        """Remove duplicate or very similar claims."""
        if not claims:
            return []
        
        # Sort by confidence (highest first)
        sorted_claims = sorted(claims, key=lambda x: x.confidence, reverse=True)
        
        unique_claims = []
        seen_texts = set()
        
        for claim in sorted_claims:
            # Create a normalized version for comparison
            normalized = self._normalize_for_comparison(claim.text)
            
            # Check if we've seen something very similar
            is_duplicate = False
            for seen in seen_texts:
                if self._similarity(normalized, seen) > 0.8:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_claims.append(claim)
                seen_texts.add(normalized)
        
        # Re-rank after deduplication
        for i, claim in enumerate(unique_claims):
            claim.rank = i + 1
        
        return unique_claims
    
    def _normalize_for_comparison(self, text: str) -> str:
        """Normalize text for similarity comparison."""
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _similarity(self, text1: str, text2: str) -> float:
        """Calculate simple word-based similarity."""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def format_claims_for_cross_pollination(self, claims: List[ExtractedClaim]) -> str:
        """Format claims for cross-pollination prompts."""
        if not claims:
            return "No claims available for cross-pollination."
        
        formatted = "EXTRACTED CLAIMS FROM ALL ENGINES:\n\n"
        
        # Group by engine
        by_engine = {}
        for claim in claims:
            if claim.engine not in by_engine:
                by_engine[claim.engine] = []
            by_engine[claim.engine].append(claim)
        
        for engine, engine_claims in by_engine.items():
            formatted += f"**{engine.upper()} ENGINE CLAIMS:**\n"
            for claim in sorted(engine_claims, key=lambda x: x.rank):
                formatted += f"{claim.rank}. {claim.text} (Confidence: {claim.confidence:.2f})\n"
            formatted += "\n"
        
        return formatted
