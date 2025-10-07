"""
Configuration settings for Gateway X
"""

from pydantic_settings import BaseSettings
from typing import List, Optional
import os


class Settings(BaseSettings):
    """Gateway X configuration settings"""
    
    # Server Configuration
    server_port: int = 3001
    server_host: str = "0.0.0.0"
    log_level: str = "INFO"
    
    # Multi-Engine Mode
    multi_engine_mode: bool = False
    load_balancing_strategy: str = "weighted"  # weighted, round_robin, least_loaded
    enable_consensus_judging: bool = False
    consensus_judge_count: int = 3
    
    # Engine API Keys
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    cohere_api_key: Optional[str] = None
    
    # Use Mock Mode for Testing
    use_real_llm: bool = False
    
    # Orchestration Settings
    default_budget: int = 20
    max_budget: int = 200
    batch_size: int = 3
    min_rounds: int = 3
    confidence_threshold: float = 0.95
    
    # Statistical Settings
    use_bootstrap_ci: bool = True
    ci_min_rounds: int = 6
    ci_bootstrap_samples: int = 200
    ci_separation_min: float = 0.05
    
    # Response Logging
    persist_responses: bool = True
    response_log_file: str = "data/logs/responses.jsonl"
    max_responses_in_memory: int = 10000
    
    class Config:
        env_prefix = "GATEWAYX_"
        case_sensitive = False
        env_file = "config/secrets/.env.local"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()
