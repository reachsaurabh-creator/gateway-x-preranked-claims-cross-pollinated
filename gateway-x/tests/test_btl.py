"""Tests for BTL ranking functionality."""

import pytest
from app.btl_ranker import BTLRanker
from app.config import Config


@pytest.fixture
def config():
    """Test configuration."""
    return Config()


@pytest.fixture
def btl_ranker(config):
    """BTL ranker instance for testing."""
    return BTLRanker(config)


def test_btl_initialization(btl_ranker):
    """Test BTL ranker initialization."""
    assert btl_ranker.rounds() == 0
    assert len(btl_ranker.theta) == 0
    assert len(btl_ranker.wins) == 0


def test_add_claims(btl_ranker):
    """Test adding claims to BTL ranker."""
    claims = ["claim1", "claim2", "claim3"]
    btl_ranker.add_claims(claims)
    
    assert len(btl_ranker.theta) == 3
    assert all(c in btl_ranker.theta for c in claims)
    # Scores should be normalized
    assert abs(sum(btl_ranker.theta.values()) - 1.0) < 1e-10


def test_update_with_duel(btl_ranker):
    """Test updating BTL scores with duel results."""
    claims = ["claim1", "claim2"]
    btl_ranker.add_claims(claims)
    
    # Simulate claim1 winning against claim2
    duel = {
        "a": "claim1",
        "b": "claim2", 
        "result": {"winner": "A"}
    }
    
    btl_ranker.update(duel)
    
    assert btl_ranker.rounds() == 1
    assert btl_ranker.theta["claim1"] > btl_ranker.theta["claim2"]


def test_select_informative_pairs(btl_ranker):
    """Test selection of informative pairs."""
    claims = ["claim1", "claim2", "claim3", "claim4"]
    btl_ranker.add_claims(claims)
    
    pairs = btl_ranker.select_k_informative_pairs(claims, k=2)
    
    assert len(pairs) <= 2
    assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in pairs)
    
    # Check no overlapping claims in selected pairs
    used_claims = set()
    for a, b in pairs:
        assert a not in used_claims
        assert b not in used_claims
        used_claims.add(a)
        used_claims.add(b)


def test_confidence_proxy(btl_ranker):
    """Test confidence proxy calculation."""
    claims = ["claim1", "claim2"]
    btl_ranker.add_claims(claims)
    
    initial_conf = btl_ranker.get_confidence_proxy()
    
    # Add several duels where claim1 consistently wins
    for _ in range(5):
        duel = {
            "a": "claim1",
            "b": "claim2",
            "result": {"winner": "A"}
        }
        btl_ranker.update(duel)
    
    final_conf = btl_ranker.get_confidence_proxy()
    assert final_conf > initial_conf


def test_best_claim(btl_ranker):
    """Test best claim selection."""
    claims = ["claim1", "claim2", "claim3"]
    btl_ranker.add_claims(claims)
    
    # Make claim2 win against others
    duels = [
        {"a": "claim2", "b": "claim1", "result": {"winner": "A"}},
        {"a": "claim2", "b": "claim3", "result": {"winner": "A"}},
    ]
    
    for duel in duels:
        btl_ranker.update(duel)
    
    assert btl_ranker.best_claim() == "claim2"


def test_compute_cis(btl_ranker):
    """Test confidence interval computation."""
    claims = ["claim1", "claim2"]
    btl_ranker.add_claims(claims)
    
    # Add some duels
    for _ in range(10):
        duel = {
            "a": "claim1",
            "b": "claim2",
            "result": {"winner": "A"}
        }
        btl_ranker.update(duel)
    
    btl_ranker.compute_cis(n_bootstrap=50)
    
    assert len(btl_ranker.cis) == 2
    for claim in claims:
        ci_low, ci_high = btl_ranker.cis[claim]
        assert 0.0 <= ci_low <= ci_high <= 1.0


def test_ci_gap(btl_ranker):
    """Test CI gap calculation."""
    claims = ["claim1", "claim2"]
    btl_ranker.add_claims(claims)
    
    # Add duels to create separation
    for _ in range(10):
        duel = {
            "a": "claim1",
            "b": "claim2",
            "result": {"winner": "A"}
        }
        btl_ranker.update(duel)
    
    btl_ranker.compute_cis(n_bootstrap=50)
    gap = btl_ranker.ci_gap()
    
    assert gap >= 0.0
