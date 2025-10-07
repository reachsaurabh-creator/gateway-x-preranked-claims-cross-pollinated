"""Pydantic models for Gateway X API."""

from typing import List, Optional
from pydantic import BaseModel, Field, conint, confloat

from .config import CONFIG


class ClaimScore(BaseModel):
    """Score and confidence interval for a claim."""
    cid: str
    score: float
    ci_low: float = 0.0
    ci_high: float = 1.0


class TimelineItem(BaseModel):
    """Timeline item representing one round of consensus."""
    run_id: str
    round_index: int
    convergence_score: float
    best_claim_cid: Optional[str] = None
    best_claim_text: Optional[str] = None
    summary: Optional[str] = None
    top_claims: List[ClaimScore] = []


class QueryIn(BaseModel):
    """Input schema for consensus query."""
    query: str = Field(..., min_length=3)
    budget: conint(ge=1, le=CONFIG.MAX_BUDGET) = CONFIG.DEFAULT_BUDGET
    confidence_threshold: confloat(ge=0.5, le=0.999) = CONFIG.CONFIDENCE_THRESHOLD


class QueryOut(BaseModel):
    """Output schema for consensus query result."""
    run_id: str
    query: str
    best_claim: str
    confidence: float
    rounds: int
    total_duels: int
    stop_reason: str
