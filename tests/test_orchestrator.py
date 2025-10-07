"""
Tests for orchestrator
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from src.gatewayx.orchestrator import Orchestrator
from src.gatewayx.engine_pool import EnginePool


@pytest.fixture
def mock_engine_pool():
    """Mock engine pool for testing"""
    pool = Mock(spec=EnginePool)
    pool.get_available_engines.return_value = ["mock1", "mock2"]
    pool.generate_response = AsyncMock(return_value={
        "text": "Test response",
        "engine": "mock1",
        "tokens": 10,
        "cost": 0.01
    })
    pool.generate_multiple_responses = AsyncMock(return_value=[
        {
            "text": "Response 1",
            "engine": "mock1",
            "tokens": 10,
            "cost": 0.01
        },
        {
            "text": "Response 2", 
            "engine": "mock2",
            "tokens": 8,
            "cost": 0.008
        }
    ])
    pool.get_pool_stats.return_value = {
        "total_engines": 2,
        "available_engines": 2,
        "total_requests": 0,
        "total_tokens": 0,
        "total_cost": 0.0
    }
    return pool


@pytest.fixture
def orchestrator(mock_engine_pool):
    return Orchestrator(mock_engine_pool)


@pytest.mark.asyncio
async def test_process_query_single_engine(orchestrator):
    """Test processing query with single engine"""
    query = "What is AI?"
    
    result = await orchestrator.process_query(query, budget=20)
    
    assert "best_claim" in result
    assert "confidence" in result
    assert "rounds_used" in result
    assert "engines_used" in result
    assert "total_cost" in result
    
    assert result["best_claim"] == "Test response"
    assert result["confidence"] == 1.0
    assert result["rounds_used"] == 1
    assert "mock1" in result["engines_used"]


@pytest.mark.asyncio
async def test_process_query_with_engines(orchestrator):
    """Test processing query with specific engines"""
    query = "What is machine learning?"
    engines = ["mock1"]
    
    result = await orchestrator.process_query(query, budget=20, engines=engines)
    
    assert result["best_claim"] == "Test response"
    assert "mock1" in result["engines_used"]


def test_get_stats(orchestrator):
    """Test orchestrator statistics"""
    stats = orchestrator.get_stats()
    
    assert "query_count" in stats
    assert "total_processing_time" in stats
    assert "average_processing_time" in stats
    assert "engine_pool_stats" in stats
    
    assert stats["query_count"] == 0  # Initially 0
    assert stats["total_processing_time"] == 0.0
