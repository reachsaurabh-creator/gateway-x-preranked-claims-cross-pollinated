"""Tests for playbook selector functionality."""

import pytest
from app.playbook_selector import PlaybookSelector
from app.config import Config


@pytest.fixture
def config():
    """Test configuration."""
    return Config()


@pytest.fixture
def selector(config):
    """Playbook selector instance for testing."""
    return PlaybookSelector(config)


def test_selector_initialization(selector):
    """Test selector initialization."""
    assert len(selector.arms) == 4
    assert "SelfConsistency" in selector.arms
    assert "Debate" in selector.arms
    assert "EvidenceFirst" in selector.arms
    assert "FocusOnDisputes" in selector.arms
    
    # All arms should start with 0 selections and 0 reward
    for arm in selector.arms:
        assert selector.n[arm] == 0
        assert selector.mu[arm] == 0.0
    
    assert selector.t == 0


def test_choose_playbook_exploration(selector):
    """Test that selector explores all arms initially."""
    state = {"claim1": 0.5, "claim2": 0.5}
    chosen_arms = set()
    
    # Should choose each arm at least once during exploration
    for round_num in range(1, 10):
        arm = selector.choose_playbook(state, round_num, total_budget=10)
        chosen_arms.add(arm)
        selector.update_performance(arm, 0.1)  # Small positive reward
    
    # Should have explored multiple arms
    assert len(chosen_arms) >= 3


def test_choose_playbook_phase_adaptation(selector):
    """Test that selector adapts to different phases."""
    state = {"claim1": 0.5, "claim2": 0.5}
    
    # Early phase (round 1 of 10)
    early_arm = selector.choose_playbook(state, round_num=1, total_budget=10)
    selector.update_performance(early_arm, 0.5)
    
    # Late phase (round 9 of 10) 
    late_arm = selector.choose_playbook(state, round_num=9, total_budget=10)
    
    # Both should be valid arms
    assert early_arm in selector.arms
    assert late_arm in selector.arms


def test_update_performance(selector):
    """Test performance update mechanism."""
    arm = "SelfConsistency"
    
    # Initial state
    assert selector.n[arm] == 0
    assert selector.mu[arm] == 0.0
    
    # Update with reward
    selector.update_performance(arm, 0.8)
    
    assert selector.n[arm] == 1
    assert selector.mu[arm] == 0.8
    
    # Update again with different reward
    selector.update_performance(arm, 0.6)
    
    assert selector.n[arm] == 2
    assert selector.mu[arm] == 0.7  # Running average: (0.8 + 0.6) / 2


def test_update_performance_running_average(selector):
    """Test running average calculation in performance updates."""
    arm = "Debate"
    rewards = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    for reward in rewards:
        selector.update_performance(arm, reward)
    
    expected_avg = sum(rewards) / len(rewards)
    assert abs(selector.mu[arm] - expected_avg) < 1e-10
    assert selector.n[arm] == len(rewards)


def test_ucb_exploitation(selector):
    """Test that selector exploits high-performing arms."""
    state = {"claim1": 0.5, "claim2": 0.5}
    
    # Give one arm consistently high rewards
    high_performing_arm = "EvidenceFirst"
    for _ in range(5):
        selector.update_performance(high_performing_arm, 0.9)
    
    # Give other arms lower rewards
    for arm in selector.arms:
        if arm != high_performing_arm:
            for _ in range(2):
                selector.update_performance(arm, 0.1)
    
    # After exploration, should tend to choose high-performing arm
    choices = []
    for round_num in range(20, 30):  # Later rounds
        arm = selector.choose_playbook(state, round_num, total_budget=30)
        choices.append(arm)
        # Continue updating to maintain performance difference
        if arm == high_performing_arm:
            selector.update_performance(arm, 0.9)
        else:
            selector.update_performance(arm, 0.1)
    
    # High-performing arm should be chosen more frequently
    high_performing_count = choices.count(high_performing_arm)
    assert high_performing_count > len(choices) // 3  # At least 1/3 of the time


def test_all_arms_valid(selector):
    """Test that all returned arms are valid."""
    state = {"claim1": 0.3, "claim2": 0.7}
    
    for round_num in range(1, 20):
        arm = selector.choose_playbook(state, round_num, total_budget=20)
        assert arm in selector.arms
        selector.update_performance(arm, 0.5)  # Neutral reward


@pytest.mark.parametrize("budget", [5, 10, 20, 50])
def test_different_budgets(selector, budget):
    """Test selector behavior with different budget sizes."""
    state = {"claim1": 0.4, "claim2": 0.6}
    
    for round_num in range(1, min(budget + 1, 10)):
        arm = selector.choose_playbook(state, round_num, total_budget=budget)
        assert arm in selector.arms
        selector.update_performance(arm, 0.5)


def test_confidence_bound_calculation(selector):
    """Test that confidence bounds affect selection."""
    state = {"claim1": 0.5, "claim2": 0.5}
    
    # Create different selection counts for arms
    selector.update_performance("SelfConsistency", 0.5)  # n=1
    for _ in range(10):
        selector.update_performance("Debate", 0.5)  # n=10
    
    # Arm with fewer selections should have higher confidence bound
    # and might be selected despite same average reward
    selections = []
    for round_num in range(1, 20):
        arm = selector.choose_playbook(state, round_num, total_budget=20)
        selections.append(arm)
        selector.update_performance(arm, 0.5)
    
    # Should have some exploration of less-selected arms
    assert "SelfConsistency" in selections
