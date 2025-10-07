"""
Dynamic Model Registry - Auto-discovers and manages LLM models
"""

import logging
import importlib
import inspect
from typing import Dict, List, Type, Optional, Any
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# Import the existing LLMEngine from llm_engines
from .llm_engines import LLMEngine

class ModelRegistry:
    """Dynamic registry for LLM models"""
    
    def __init__(self):
        self.engines: Dict[str, Type[LLMEngine]] = {}
        self.instances: Dict[str, LLMEngine] = {}
        self._register_builtin_engines()
    
    def _register_builtin_engines(self):
        """Register built-in engine classes"""
        # Import built-in engines
        from .llm_engines import (
            MockEngine, AnthropicEngine, OpenAIEngine, 
            GoogleEngine, CohereEngine
        )
        
        # Register them
        self.register_engine('mock', MockEngine)
        self.register_engine('anthropic', AnthropicEngine)
        self.register_engine('openai', OpenAIEngine)
        self.register_engine('google', GoogleEngine)
        self.register_engine('cohere', CohereEngine)
        
        # Auto-discover engines from engines package
        self._discover_dynamic_engines()
        
        logger.info(f"Registered {len(self.engines)} engines: {list(self.engines.keys())}")
    
    def _discover_dynamic_engines(self):
        """Auto-discover engines from the engines package"""
        try:
            from . import engines
            
            # Look for engine classes in the engines module
            for attr_name in dir(engines):
                attr = getattr(engines, attr_name)
                if (isinstance(attr, type) and 
                    hasattr(attr, '__name__') and 
                    attr.__name__.endswith('Engine') and
                    attr.__name__ != 'LLMEngine'):
                    
                    engine_name = attr.__name__.replace('Engine', '').lower()
                    self.register_engine(engine_name, attr)
                    logger.info(f"🔍 Auto-discovered engine: {engine_name}")
                    
        except Exception as e:
            logger.warning(f"⚠️ Failed to discover dynamic engines: {e}")
    
    def register_engine(self, name: str, engine_class: Type[LLMEngine]):
        """Register a new engine class"""
        # Check if it has the required methods instead of strict inheritance
        required_methods = ['health_check', 'generate', 'get_cost_per_token']
        if not all(hasattr(engine_class, method) for method in required_methods):
            logger.warning(f"⚠️ Engine class {engine_class} missing required methods, but registering anyway")
        
        self.engines[name.lower()] = engine_class
        logger.info(f"✅ Registered engine: {name}")
    
    def discover_engines_from_config(self, config_dict: Dict[str, Any]) -> List[str]:
        """Discover available engines from configuration"""
        discovered = []
        
        # Look for API key patterns
        for key, value in config_dict.items():
            if key.endswith('_API_KEY') and value and value not in [
                'your_anthropic_key_here',
                'your_openai_key_here', 
                'your_google_key_here',
                'your_cohere_key_here',
                'sk-...',
                '...'
            ]:
                # Extract provider name
                provider = key.replace('GATEWAYX_', '').replace('_API_KEY', '').lower()
                discovered.append(provider)
        
        logger.info(f"🔍 Discovered engines from config: {discovered}")
        return discovered
    
    async def create_engine_instance(self, name: str, api_key: str, **kwargs) -> Optional[LLMEngine]:
        """Create an instance of an engine"""
        engine_class = self.engines.get(name.lower())
        if not engine_class:
            logger.warning(f"⚠️ Unknown engine: {name}")
            return None
        
        try:
            instance = engine_class(api_key, **kwargs)
            # Skip health check during creation - let the engine pool handle it
            # await instance.health_check()
            self.instances[name] = instance
            logger.info(f"✅ Created engine instance: {name}")
            return instance
        except Exception as e:
            logger.error(f"❌ Failed to create engine {name}: {e}")
            return None
    
    def get_available_engines(self) -> List[str]:
        """Get list of available engine names"""
        return list(self.engines.keys())
    
    def get_engine_instance(self, name: str) -> Optional[LLMEngine]:
        """Get an existing engine instance"""
        return self.instances.get(name.lower())
    
    def get_all_instances(self) -> Dict[str, LLMEngine]:
        """Get all engine instances"""
        return self.instances.copy()
    
    async def cleanup(self):
        """Cleanup all engine instances"""
        for name, instance in self.instances.items():
            try:
                if hasattr(instance, 'cleanup'):
                    await instance.cleanup()
                logger.info(f"🧹 Cleaned up engine: {name}")
            except Exception as e:
                logger.warning(f"⚠️ Error cleaning up {name}: {e}")
        
        self.instances.clear()

# Global registry instance
registry = ModelRegistry()
