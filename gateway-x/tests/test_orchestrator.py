"""Tests for orchestrator functionality."""

import pytest
import asyncio
from app.orchestrator import Orchestrator
from app.config import Config


@pytest.fixture
def config():
    """Test configuration with shorter parameters for faster tests."""
    config = Config()
    # Override some values for faster tests
    config.CI_MIN_ROUNDS = 2
    config.CI_BOOTSTRAP_SAMPLES = 10
    config.DUELS_PER_ROUND = 2
    return config


@pytest.fixture
def orchestrator(config):
    """Orchestrator instance for testing."""
    return Orchestrator(config)


@pytest.mark.asyncio
async def test_orchestrator_initialization(orchestrator):
    """Test orchestrator initialization."""
    assert len(orchestrator.runs) == 0
    assert len(orchestrator.current_engines) > 0
    assert orchestrator.scheduler is not None
    assert orchestrator.btl is not None
    assert orchestrator.selector is not None
    assert orchestrator.stopper is not None


@pytest.mark.asyncio
async def test_run_basic_consensus(orchestrator):
    """Test basic consensus run."""
    query = "What is 2+2?"
    budget = 3
    confidence_threshold = 0.8
    
    result = await orchestrator.run(query, budget, confidence_threshold)
    
    # Check result structure
    assert "run_id" in result
    assert "query" in result
    assert "best_claim" in result
    assert "confidence" in result
    assert "rounds" in result
    assert "total_duels" in result
    assert "stop_reason" in result
    
    # Check values
    assert result["query"] == query
    assert result["rounds"] >= 1
    assert result["total_duels"] >= 0
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["stop_reason"] in ["budget_exhausted", "confidence_threshold", "ci_separation", "max_rounds"]


@pytest.mark.asyncio
async def test_run_creates_timeline(orchestrator):
    """Test that consensus run creates timeline."""
    query = "Test query"
    budget = 2
    
    result = await orchestrator.run(query, budget, 0.9)
    run_id = result["run_id"]
    
    # Check timeline was created
    timeline = orchestrator.get_timeline(run_id)
    assert len(timeline) >= 1
    assert len(timeline) <= budget
    
    # Check timeline items
    for item in timeline:
        assert item.run_id == run_id
        assert item.round_index >= 1
        assert 0.0 <= item.convergence_score <= 1.0
        assert item.best_claim_cid is not None
        assert item.summary is not None


@pytest.mark.asyncio
async def test_run_with_different_budgets(orchestrator):
    """Test consensus with different budget sizes."""
    query = "Test query"
    
    for budget in [1, 3, 5]:
        result = await orchestrator.run(query, budget, 0.95)
        
        assert result["rounds"] >= 1
        assert result["rounds"] <= budget
        
        timeline = orchestrator.get_timeline(result["run_id"])
        assert len(timeline) == result["rounds"]


@pytest.mark.asyncio
async def test_run_confidence_threshold_stopping(orchestrator):
    """Test that high confidence can trigger early stopping."""
    query = "Simple question"
    budget = 10
    confidence_threshold = 0.1  # Very low threshold, should stop early
    
    result = await orchestrator.run(query, budget, confidence_threshold)
    
    # Should stop before budget exhausted due to confidence
    if result["stop_reason"] == "confidence_threshold":
        assert result["rounds"] < budget
        assert result["confidence"] >= confidence_threshold


@pytest.mark.asyncio
async def test_filter_invalid_claims(orchestrator):
    """Test claim filtering functionality."""
    # Test with valid claims
    valid_claims = ["This is a good claim", "Another valid claim", "Third claim here"]
    filtered = orchestrator._filter_invalid_claims(valid_claims)
    assert len(filtered) == 3
    
    # Test with short claims (should be filtered out)
    short_claims = ["a", "bb", "This is long enough"]
    filtered = orchestrator._filter_invalid_claims(short_claims)
    assert len(filtered) == 1
    assert "This is long enough" in filtered
    
    # Test with duplicate claims
    duplicate_claims = ["Same claim", "Same claim", "Different claim"]
    filtered = orchestrator._filter_invalid_claims(duplicate_claims)
    assert len(filtered) == 2


def test_build_truth_summary(orchestrator):
    """Test truth summary generation."""
    from app.schemas import ClaimScore
    
    # Test with no best claim
    summary = orchestrator._build_truth_summary(None, [])
    assert "No consensus yet" in summary
    
    # Test with best claim and contenders
    top_claims = [
        ClaimScore(cid="claim1", score=0.8, ci_low=0.7, ci_high=0.9),
        ClaimScore(cid="claim2", score=0.6, ci_low=0.5, ci_high=0.7),
        ClaimScore(cid="claim3", score=0.4, ci_low=0.3, ci_high=0.5),
    ]
    
    summary = orchestrator._build_truth_summary("claim1", top_claims)
    
    assert "claim1" in summary
    assert "current best" in summary
    assert "claim2" in summary
    assert "claim3" in summary


@pytest.mark.asyncio
async def test_multiple_runs_isolated(orchestrator):
    """Test that multiple runs are properly isolated."""
    query1 = "First query"
    query2 = "Second query"
    
    result1 = await orchestrator.run(query1, 2, 0.9)
    result2 = await orchestrator.run(query2, 2, 0.9)
    
    # Should have different run IDs
    assert result1["run_id"] != result2["run_id"]
    
    # Should have separate timelines
    timeline1 = orchestrator.get_timeline(result1["run_id"])
    timeline2 = orchestrator.get_timeline(result2["run_id"])
    
    assert len(timeline1) >= 1
    assert len(timeline2) >= 1
    assert timeline1[0].run_id != timeline2[0].run_id


def test_get_timeline_nonexistent(orchestrator):
    """Test getting timeline for non-existent run."""
    timeline = orchestrator.get_timeline("nonexistent_run_id")
    assert timeline == []


@pytest.mark.asyncio
async def test_run_handles_empty_pairs(orchestrator):
    """Test that orchestrator handles cases with no informative pairs."""
    query = "Test"
    budget = 1  # Very small budget
    
    # Should not crash even with minimal setup
    result = await orchestrator.run(query, budget, 0.9)
    
    assert "run_id" in result
    assert result["rounds"] >= 0


@pytest.mark.asyncio
async def test_concurrent_runs(orchestrator):
    """Test that orchestrator can handle concurrent runs."""
    queries = ["Query 1", "Query 2", "Query 3"]
    
    # Start multiple runs concurrently
    tasks = [
        orchestrator.run(query, 2, 0.9) 
        for query in queries
    ]
    
    results = await asyncio.gather(*tasks)
    
    # All should complete successfully
    assert len(results) == 3
    
    # All should have unique run IDs
    run_ids = [r["run_id"] for r in results]
    assert len(set(run_ids)) == 3
    
    # All should have their queries preserved
    result_queries = [r["query"] for r in results]
    assert set(result_queries) == set(queries)
