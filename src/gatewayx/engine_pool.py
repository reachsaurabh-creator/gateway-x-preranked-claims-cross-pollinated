"""
Engine Pool - Manages multiple LLM engines
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
import random
import time

from .llm_engines import (
    MockEngine, AnthropicEngine, OpenAIEngine, 
    GoogleEngine, CohereEngine, LLMEngine
)
from .settings import settings
from .secure_config import config
from .model_registry import registry

logger = logging.getLogger(__name__)


class EnginePool:
    """Manages a pool of LLM engines"""
    
    def __init__(self):
        self.engines: Dict[str, LLMEngine] = {}
        self.weights: Dict[str, float] = {}
        self.round_robin_index = 0
        self.initialized = False
    
    async def initialize(self):
        """Initialize the engine pool"""
        logger.info("Initializing engine pool...")
        
        if config.get("use_real_llm", False):
            # Discover available providers dynamically
            available_providers = config.discover_available_providers()
            logger.info(f"🔍 Discovered {len(available_providers)} providers: {available_providers}")
            
            # Initialize engines for each discovered provider
            for provider in available_providers:
                api_key = config.get_api_key(provider)
                if api_key:
                    try:
                        # Use the registry to create engine instances
                        engine = await registry.create_engine_instance(provider, api_key)
                        if engine:
                            # Run health check to determine availability
                            try:
                                is_healthy = await engine.health_check()
                                if is_healthy:
                                    self.engines[provider] = engine
                                    self.weights[provider] = 1.0
                                    logger.info(f"✅ {provider.title()} engine initialized")
                                else:
                                    logger.warning(f"⚠️ {provider.title()} engine failed health check")
                            except Exception as e:
                                logger.warning(f"⚠️ {provider.title()} engine health check failed: {e}")
                                # Still add the engine but mark as unavailable
                                self.engines[provider] = engine
                                self.weights[provider] = 1.0
                        else:
                            logger.warning(f"⚠️ Failed to create {provider} engine instance")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to initialize {provider} engine: {e}")
                else:
                    logger.warning(f"⚠️ No API key found for {provider}")
        else:
            # Use mock engines for testing
            for i in range(3):
                engine_name = f"mock_{i+1}"
                self.engines[engine_name] = MockEngine(engine_name)
                self.weights[engine_name] = 1.0
            logger.info("Mock engines initialized")
        
        if not self.engines:
            raise RuntimeError("No engines available")
        
        self.initialized = True
        logger.info(f"Engine pool initialized with {len(self.engines)} engines")
    
    async def cleanup(self):
        """Cleanup engine pool"""
        logger.info("Cleaning up engine pool...")
        self.engines.clear()
        self.weights.clear()
        self.initialized = False
    
    def get_available_engines(self) -> List[str]:
        """Get list of available engine names"""
        available = []
        for name, engine in self.engines.items():
            if engine.is_available:
                available.append(name)
            elif name == "anthropic":
                # Special case: include Anthropic even if marked as unavailable
                # since the health check might fail but the API call might work
                available.append(name)
        return available
    
    def get_engine(self, name: str) -> Optional[LLMEngine]:
        """Get engine by name"""
        return self.engines.get(name)
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all engines"""
        return {
            name: engine.get_stats() 
            for name, engine in self.engines.items()
        }
    
    async def select_engine(self, strategy: str = "weighted", 
                          exclude: Optional[List[str]] = None) -> Optional[LLMEngine]:
        """Select an engine based on strategy"""
        available_engines = self.get_available_engines()
        
        if exclude:
            available_engines = [e for e in available_engines if e not in exclude]
        
        if not available_engines:
            return None
        
        if strategy == "weighted":
            return self._select_weighted(available_engines)
        elif strategy == "round_robin":
            return self._select_round_robin(available_engines)
        elif strategy == "least_loaded":
            return self._select_least_loaded(available_engines)
        else:
            return self._select_weighted(available_engines)
    
    def _select_weighted(self, available_engines: List[str]) -> Optional[LLMEngine]:
        """Select engine using weighted random selection"""
        if not available_engines:
            return None
        
        # Get weights for available engines
        weights = [self.weights.get(name, 1.0) for name in available_engines]
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            weights = [1.0] * len(available_engines)
            total_weight = len(available_engines)
        
        normalized_weights = [w / total_weight for w in weights]
        
        # Weighted random selection
        selected = random.choices(available_engines, weights=normalized_weights)[0]
        return self.engines[selected]
    
    def _select_round_robin(self, available_engines: List[str]) -> Optional[LLMEngine]:
        """Select engine using round-robin"""
        if not available_engines:
            return None
        
        # Find the next available engine in round-robin order
        for i in range(len(available_engines)):
            index = (self.round_robin_index + i) % len(available_engines)
            engine_name = available_engines[index]
            if engine_name in self.engines and self.engines[engine_name].is_available:
                self.round_robin_index = (index + 1) % len(available_engines)
                return self.engines[engine_name]
        
        return None
    
    def _select_least_loaded(self, available_engines: List[str]) -> Optional[LLMEngine]:
        """Select engine with least load"""
        if not available_engines:
            return None
        
        # Find engine with minimum total requests
        min_requests = float('inf')
        selected_engine = None
        
        for name in available_engines:
            engine = self.engines[name]
            if engine.total_requests < min_requests:
                min_requests = engine.total_requests
                selected_engine = engine
        
        return selected_engine
    
    async def generate_response(self, prompt: str, engine_name: Optional[str] = None,
                              **kwargs) -> Dict[str, Any]:
        """Generate response using specified or selected engine"""
        if engine_name:
            engine = self.get_engine(engine_name)
            if not engine or not engine.is_available:
                raise ValueError(f"Engine {engine_name} not available")
        else:
            strategy = settings.load_balancing_strategy
            engine = await self.select_engine(strategy)
            if not engine:
                raise RuntimeError("No available engines")
        
        return await engine.generate_response(prompt, **kwargs)
    
    async def generate_multiple_responses(self, prompt: str, 
                                        engine_names: Optional[List[str]] = None,
                                        **kwargs) -> List[Dict[str, Any]]:
        """Generate responses from multiple engines"""
        if engine_names is None:
            available_engines = self.get_available_engines()
            # Limit to reasonable number for consensus
            engine_names = available_engines[:settings.consensus_judge_count]
        
        tasks = []
        for engine_name in engine_names:
            if engine_name in self.engines and self.engines[engine_name].is_available:
                task = self.generate_response(prompt, engine_name, **kwargs)
                tasks.append(task)
        
        if not tasks:
            raise RuntimeError("No available engines for multiple responses")
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and return successful results
        successful_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Engine failed: {result}")
            else:
                successful_results.append(result)
        
        return successful_results
    
    def update_engine_weight(self, engine_name: str, weight: float):
        """Update engine weight for load balancing"""
        if engine_name in self.engines:
            self.weights[engine_name] = max(0.0, weight)
            logger.info(f"Updated weight for {engine_name}: {weight}")
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get overall pool statistics"""
        total_requests = sum(engine.total_requests for engine in self.engines.values())
        total_tokens = sum(engine.total_tokens for engine in self.engines.values())
        total_cost = sum(engine.total_cost for engine in self.engines.values())
        available_count = len(self.get_available_engines())
        
        return {
            "total_engines": len(self.engines),
            "available_engines": available_count,
            "total_requests": total_requests,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "engines": self.get_status()
        }
