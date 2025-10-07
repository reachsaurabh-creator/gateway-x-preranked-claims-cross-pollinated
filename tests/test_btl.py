"""
Tests for BTL ranking system
"""

import pytest
import asyncio
from src.gatewayx.btl_ranker import BTLRanker


@pytest.fixture
def btl_ranker():
    return BTLRanker()


@pytest.fixture
def sample_responses():
    return [
        {
            "text": "This is a detailed response with comprehensive information.",
            "engine": "engine1",
            "tokens": 10,
            "cost": 0.01
        },
        {
            "text": "Short response.",
            "engine": "engine2", 
            "tokens": 2,
            "cost": 0.002
        },
        {
            "text": "Another detailed response with good structure and examples.",
            "engine": "engine3",
            "tokens": 12,
            "cost": 0.012
        }
    ]


@pytest.mark.asyncio
async def test_rank_responses(btl_ranker, sample_responses):
    """Test BTL ranking of responses"""
    query = "What is artificial intelligence?"
    
    result = await btl_ranker.rank_responses(query, sample_responses)
    
    assert "best_response" in result
    assert "confidence" in result
    assert "rankings" in result
    assert "btl_scores" in result
    
    assert len(result["rankings"]) == len(sample_responses)
    assert len(result["btl_scores"]) == len(sample_responses)
    assert 0 <= result["confidence"] <= 1


@pytest.mark.asyncio
async def test_single_response(btl_ranker):
    """Test BTL ranking with single response"""
    query = "Test query"
    responses = [{"text": "Single response", "engine": "test", "tokens": 2, "cost": 0.001}]
    
    result = await btl_ranker.rank_responses(query, responses)
    
    assert result["best_response"]["text"] == "Single response"
    assert result["confidence"] == 1.0
    assert len(result["rankings"]) == 1


@pytest.mark.asyncio
async def test_empty_responses(btl_ranker):
    """Test BTL ranking with empty responses"""
    query = "Test query"
    responses = []
    
    result = await btl_ranker.rank_responses(query, responses)
    
    assert result["best_response"] is None
    assert result["confidence"] == 1.0
    assert len(result["rankings"]) == 0


def test_get_ranking_stats(btl_ranker):
    """Test ranking statistics"""
    stats = btl_ranker.get_ranking_stats()
    
    assert "total_rankings" in stats
    assert stats["total_rankings"] == 0  # Initially empty
