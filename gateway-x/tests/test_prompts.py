"""Tests for prompt management and validation."""

import json
import pytest
from app.prompts import PromptVault


def test_duel_prompt_generation():
    """Test duel prompt generation."""
    query = "What is the capital of France?"
    a = "Paris is the capital of France."
    b = "The capital of France is Lyon."
    
    prompt = PromptVault.duel_prompt(query, a, b)
    
    assert query in prompt
    assert a in prompt
    assert b in prompt
    assert "JSON" in prompt
    assert "winner" in prompt


def test_validate_response_valid_json():
    """Test validation of valid JSON response."""
    valid_response = json.dumps({
        "winner": "A",
        "factuality": 0.8,
        "coherence": 0.9,
        "note": "Clear and accurate"
    })
    
    result = PromptVault.validate_response(valid_response)
    
    assert result["winner"] == "A"
    assert result["factuality"] == 0.8
    assert result["coherence"] == 0.9
    assert result["note"] == "Clear and accurate"


def test_validate_response_minimal_json():
    """Test validation with minimal required fields."""
    minimal_response = json.dumps({"winner": "B"})
    
    result = PromptVault.validate_response(minimal_response)
    
    assert result["winner"] == "B"
    assert result["factuality"] == 0.5  # Default value
    assert result["coherence"] == 0.5   # Default value
    assert result["note"] == ""         # Default value


def test_validate_response_invalid_winner():
    """Test validation with invalid winner."""
    invalid_response = json.dumps({
        "winner": "C",  # Invalid winner
        "factuality": 0.7
    })
    
    result = PromptVault.validate_response(invalid_response)
    
    # Should fallback to random choice
    assert result["winner"] in ["A", "B"]
    assert result["factuality"] == 0.5
    assert result["coherence"] == 0.5
    assert result["note"] == "fallback"


def test_validate_response_malformed_json():
    """Test validation with malformed JSON."""
    malformed_response = "This is not JSON at all"
    
    result = PromptVault.validate_response(malformed_response)
    
    # Should fallback gracefully
    assert result["winner"] in ["A", "B"]
    assert result["factuality"] == 0.5
    assert result["coherence"] == 0.5
    assert result["note"] == "fallback"


def test_validate_response_empty_string():
    """Test validation with empty string."""
    result = PromptVault.validate_response("")
    
    # Should fallback gracefully
    assert result["winner"] in ["A", "B"]
    assert result["factuality"] == 0.5
    assert result["coherence"] == 0.5
    assert result["note"] == "fallback"


def test_validate_response_note_truncation():
    """Test that long notes are truncated."""
    long_note = "x" * 100  # 100 character note
    response = json.dumps({
        "winner": "A",
        "note": long_note
    })
    
    result = PromptVault.validate_response(response)
    
    assert len(result["note"]) <= 60
    assert result["note"] == long_note[:60]


@pytest.mark.parametrize("winner", ["A", "B"])
def test_validate_response_both_winners(winner):
    """Test validation works for both valid winners."""
    response = json.dumps({"winner": winner})
    result = PromptVault.validate_response(response)
    assert result["winner"] == winner


@pytest.mark.parametrize("factuality,coherence", [
    (0.0, 0.0),
    (0.5, 0.5), 
    (1.0, 1.0),
    (0.123, 0.987)
])
def test_validate_response_score_ranges(factuality, coherence):
    """Test validation with various score ranges."""
    response = json.dumps({
        "winner": "A",
        "factuality": factuality,
        "coherence": coherence
    })
    
    result = PromptVault.validate_response(response)
    
    assert result["factuality"] == factuality
    assert result["coherence"] == coherence
