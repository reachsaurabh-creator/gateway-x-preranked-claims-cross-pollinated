"""Configuration management for Gateway X."""

import os
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Config:
    """Environment-driven configuration with safe defaults for local development."""
    
    # Budgets & rounds
    DEFAULT_BUDGET: int = int(os.getenv("GX_DEFAULT_BUDGET", "6"))
    MAX_BUDGET: int = int(os.getenv("GX_MAX_BUDGET", "200"))
    MAX_ROUNDS: int = int(os.getenv("GX_MAX_ROUNDS", "200"))
    DUELS_PER_ROUND: int = int(os.getenv("GX_DUELS_PER_ROUND", "3"))

    # BTL & CI
    CI_MIN_ROUNDS: int = int(os.getenv("GX_CI_MIN_ROUNDS", "6"))
    CI_BOOTSTRAP_SAMPLES: int = int(os.getenv("GX_CI_BOOTSTRAP", "200"))
    CI_SEPARATION_MIN: float = float(os.getenv("GX_CI_SEP_MIN", "0.05"))
    PAIR_SAMPLE_LIMIT: int = int(os.getenv("GX_PAIR_SAMPLE_LIMIT", "120"))
    BTL_FLOOR: float = float(os.getenv("GX_BTL_FLOOR", "1e-6"))
    BTL_ITERS: int = int(os.getenv("GX_BTL_ITERS", "3"))

    # UCB
    UCB_WEIGHT: float = float(os.getenv("GX_UCB_WEIGHT", "0.5"))

    # Dawid–Skene (stub weights)
    DS_INIT_ACC: float = float(os.getenv("GX_DS_INIT_ACC", "0.6"))
    DS_MIN_WEIGHT: float = float(os.getenv("GX_DS_MIN_WEIGHT", "0.1"))
    DS_EMA_ALPHA: float = float(os.getenv("GX_DS_EMA_ALPHA", "0.1"))

    # Claims
    MIN_CLAIM_CHARS: int = int(os.getenv("GX_MIN_CLAIM_CHARS", "10"))

    # Confidence default
    CONFIDENCE_THRESHOLD: float = float(os.getenv("GX_CONFIDENCE_THR", "0.95"))

    # LLM configuration
    USE_REAL_LLM: bool = os.getenv("GX_USE_REAL_LLM", "true").lower() == "true"
    
    # API Keys
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    XAI_API_KEY: str = os.getenv("XAI_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    
    # Model configurations
    ANTHROPIC_MODEL: str = os.getenv("GX_ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
    OPENAI_MODEL: str = os.getenv("GX_OPENAI_MODEL", "gpt-4o")
    XAI_MODEL: str = os.getenv("GX_XAI_MODEL", "grok-3")
    GEMINI_MODEL: str = os.getenv("GX_GEMINI_MODEL", "gemini-2.5-flash")
    
    # Generation parameters
    MAX_TOKENS: int = int(os.getenv("GX_MAX_TOKENS", "4000"))
    TEMPERATURE: float = float(os.getenv("GX_TEMPERATURE", "0.7"))
    
    # Engine selection
    ENABLED_ENGINES: List[str] = None
    REFEREE_ENGINE: str = os.getenv("GX_REFEREE_ENGINE", "anthropic")
    
    def __post_init__(self):
        if self.ENABLED_ENGINES is None:
            object.__setattr__(self, 'ENABLED_ENGINES', os.getenv("GX_ENABLED_ENGINES", "anthropic,openai,xai,gemini").split(","))


# Global config instance
CONFIG = Config()
