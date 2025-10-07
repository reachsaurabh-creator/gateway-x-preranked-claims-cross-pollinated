"""
Tests for LLM engines
"""

import pytest
import asyncio
from src.gatewayx.llm_engines import MockEngine, AnthropicEngine, OpenAIEngine


@pytest.fixture
def mock_engine():
    return MockEngine("test_mock")


@pytest.mark.asyncio
async def test_mock_engine_generate(mock_engine):
    """Test mock engine response generation"""
    prompt = "What is AI?"
    
    response = await mock_engine.generate_response(prompt)
    
    assert "text" in response
    assert "tokens" in response
    assert "cost" in response
    assert "engine" in response
    
    assert response["engine"] == "test_mock"
    assert isinstance(response["text"], str)
    assert response["tokens"] > 0
    assert response["cost"] > 0


@pytest.mark.asyncio
async def test_mock_engine_health_check(mock_engine):
    """Test mock engine health check"""
    is_healthy = await mock_engine.health_check()
    assert is_healthy is True


def test_mock_engine_stats(mock_engine):
    """Test mock engine statistics"""
    stats = mock_engine.get_stats()
    
    assert "name" in stats
    assert "is_available" in stats
    assert "total_requests" in stats
    assert "total_tokens" in stats
    assert "total_cost" in stats
    
    assert stats["name"] == "test_mock"
    assert stats["is_available"] is True
    assert stats["total_requests"] == 0


@pytest.mark.asyncio
async def test_anthropic_engine_without_key():
    """Test Anthropic engine without API key"""
    engine = AnthropicEngine("invalid_key")
    
    # Should raise an error or return False for health check
    is_healthy = await engine.health_check()
    assert is_healthy is False


@pytest.mark.asyncio
async def test_openai_engine_without_key():
    """Test OpenAI engine without API key"""
    engine = OpenAIEngine("invalid_key")
    
    # Should raise an error or return False for health check
    is_healthy = await engine.health_check()
    assert is_healthy is False
