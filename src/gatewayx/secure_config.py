"""
Secure Configuration Manager for Gateway X
Handles API key loading with proper security practices
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class SecureConfig:
    """Secure configuration manager that prioritizes local secrets"""
    
    def __init__(self):
        self.config = {}
        self._load_config()
    
    def _load_config(self):
        """Load configuration with security priority order"""
        
        # Priority order for config loading (highest to lowest priority)
        config_paths = [
            # 1. Local secrets (highest priority - never committed)
            "config/secrets/.env.local",
            # 2. User home directory (for personal configs)
            os.path.expanduser("~/.gatewayx/.env"),
            # 3. Project root (for development)
            ".env",
            # 4. Template (lowest priority - safe to commit)
            "config/templates/.env.template"
        ]
        
        loaded_configs = []
        
        for config_path in config_paths:
            if os.path.exists(config_path):
                try:
                    load_dotenv(config_path, override=False)  # Don't override existing values
                    loaded_configs.append(config_path)
                    logger.info(f"✅ Loaded config from: {config_path}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load {config_path}: {e}")
        
        if not loaded_configs:
            logger.warning("⚠️ No configuration files found! Using defaults.")
        
        # Load environment variables into config dict
        self._load_env_vars()
        
        # Validate API key security
        self._validate_api_key_security()
    
    def _load_env_vars(self):
        """Load environment variables into config dictionary"""
        
        # Server Configuration
        self.config.update({
            'server_port': int(os.getenv('GATEWAYX_SERVER_PORT', '3001')),
            'server_host': os.getenv('GATEWAYX_SERVER_HOST', '0.0.0.0'),
            'log_level': os.getenv('GATEWAYX_LOG_LEVEL', 'INFO'),
        })
        
        # Multi-Engine Mode
        self.config.update({
            'multi_engine_mode': os.getenv('GATEWAYX_MULTI_ENGINE_MODE', 'false').lower() == 'true',
            'load_balancing_strategy': os.getenv('GATEWAYX_LOAD_BALANCING_STRATEGY', 'weighted'),
            'enable_consensus_judging': os.getenv('GATEWAYX_ENABLE_CONSENSUS_JUDGING', 'false').lower() == 'true',
            'consensus_judge_count': int(os.getenv('GATEWAYX_CONSENSUS_JUDGE_COUNT', '3')),
        })
        
        # API Keys (sensitive - handled specially)
        self.config.update({
            'anthropic_api_key': os.getenv('GATEWAYX_ANTHROPIC_API_KEY'),
            'openai_api_key': os.getenv('GATEWAYX_OPENAI_API_KEY'),
            'google_api_key': os.getenv('GATEWAYX_GOOGLE_API_KEY'),
            'grok_api_key': os.getenv('GATEWAYX_GROK_API_KEY'),
        })
        
        # Engine Settings
        self.config.update({
            'use_real_llm': os.getenv('GATEWAYX_USE_REAL_LLM', 'false').lower() == 'true',
        })
        
        # Orchestration Settings
        self.config.update({
            'default_budget': float(os.getenv('GATEWAYX_DEFAULT_BUDGET', '20')),
            'max_budget': float(os.getenv('GATEWAYX_MAX_BUDGET', '200')),
            'batch_size': int(os.getenv('GATEWAYX_BATCH_SIZE', '3')),
            'min_rounds': int(os.getenv('GATEWAYX_MIN_ROUNDS', '3')),
            'confidence_threshold': float(os.getenv('GATEWAYX_CONFIDENCE_THRESHOLD', '0.95')),
        })
        
        # Statistical Settings
        self.config.update({
            'use_bootstrap_ci': os.getenv('GATEWAYX_USE_BOOTSTRAP_CI', 'true').lower() == 'true',
            'ci_min_rounds': int(os.getenv('GATEWAYX_CI_MIN_ROUNDS', '6')),
            'ci_bootstrap_samples': int(os.getenv('GATEWAYX_CI_BOOTSTRAP_SAMPLES', '200')),
            'ci_separation_min': float(os.getenv('GATEWAYX_CI_SEPARATION_MIN', '0.05')),
        })
        
        # Response Logging
        self.config.update({
            'persist_responses': os.getenv('GATEWAYX_PERSIST_RESPONSES', 'true').lower() == 'true',
            'response_log_file': os.getenv('GATEWAYX_RESPONSE_LOG_FILE', 'data/logs/responses.jsonl'),
            'max_responses_in_memory': int(os.getenv('GATEWAYX_MAX_RESPONSES_IN_MEMORY', '10000')),
        })
    
    def _validate_api_key_security(self):
        """Validate API key security and provide warnings"""
        
        api_keys = {
            'Anthropic': self.config.get('anthropic_api_key'),
            'OpenAI': self.config.get('openai_api_key'),
            'Google': self.config.get('google_api_key'),
            'Cohere': self.config.get('cohere_api_key'),
        }
        
        # Check for placeholder values
        placeholder_values = [
            'your_anthropic_key_here',
            'your_openai_key_here', 
            'your_google_key_here',
            'your_cohere_key_here',
            'sk-...',
            '...'
        ]
        
        for provider, key in api_keys.items():
            if key and key in placeholder_values:
                logger.warning(f"⚠️ {provider} API key appears to be a placeholder. Please set your real API key.")
            elif key and len(key) < 10:
                logger.warning(f"⚠️ {provider} API key seems too short. Please verify it's correct.")
            elif key:
                # Mask the key for logging
                masked_key = key[:8] + "..." + key[-4:] if len(key) > 12 else "***"
                logger.info(f"✅ {provider} API key loaded: {masked_key}")
        
        # Check if we're in mock mode when real keys are available
        if self.config.get('use_real_llm') == False and any(key and key not in placeholder_values for key in api_keys.values()):
            logger.info("ℹ️ Real API keys detected but USE_REAL_LLM=false. Set GATEWAYX_USE_REAL_LLM=true to use real LLMs.")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(key, default)
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for specific provider with security logging"""
        # Dynamic key mapping - look for any GATEWAYX_{PROVIDER}_API_KEY pattern
        key_name = f"{provider.lower()}_api_key"
        api_key = self.config.get(key_name)
        
        if api_key:
            # Log access for security monitoring
            logger.debug(f"🔑 API key accessed for {provider}")
            return api_key
        else:
            logger.warning(f"⚠️ No API key found for {provider}")
            return None
    
    def discover_available_providers(self) -> List[str]:
        """Discover all available providers from config"""
        providers = []
        
        # Look for all API key patterns in config
        for key, value in self.config.items():
            if key.endswith('_api_key') and value and value not in [
                'your_anthropic_key_here',
                'your_openai_key_here', 
                'your_google_key_here',
                'your_grok_key_here',
                'sk-...',
                '...'
            ]:
                # Extract provider name
                provider = key.replace('_api_key', '')
                providers.append(provider)
        
        logger.info(f"🔍 Discovered providers: {providers}")
        return providers
    
    def is_secure_mode(self) -> bool:
        """Check if running in secure mode (local secrets loaded)"""
        return os.path.exists("config/secrets/.env.local")
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary (safe for logging)"""
        summary = {
            'server': {
                'host': self.config.get('server_host'),
                'port': self.config.get('server_port'),
                'log_level': self.config.get('log_level'),
            },
            'engines': {
                'multi_engine_mode': self.config.get('multi_engine_mode'),
                'use_real_llm': self.config.get('use_real_llm'),
                'load_balancing_strategy': self.config.get('load_balancing_strategy'),
            },
            'api_keys': {
                'anthropic': '***' if self.config.get('anthropic_api_key') else 'Not set',
                'openai': '***' if self.config.get('openai_api_key') else 'Not set',
                'google': '***' if self.config.get('google_api_key') else 'Not set',
                'cohere': '***' if self.config.get('cohere_api_key') else 'Not set',
            },
            'security': {
                'secure_mode': self.is_secure_mode(),
                'local_secrets_loaded': os.path.exists("config/secrets/.env.local"),
            }
        }
        return summary

# Global config instance
config = SecureConfig()
