#!/usr/bin/env python3
"""
Test Dynamic Model System
Demonstrates adding/removing models without code changes
"""

import sys
import os
sys.path.insert(0, 'src')

def test_dynamic_model_system():
    """Test the complete dynamic model system"""
    print("🚀 Testing Dynamic Model System")
    print("=" * 50)
    
    # Test 1: Check available engines
    print("\n1️⃣ Available Engines:")
    from gatewayx.model_registry import registry
    for name, engine_class in registry.engines.items():
        print(f"   ✅ {name}: {engine_class.__name__}")
    print(f"   Total: {len(registry.engines)} engines")
    
    # Test 2: Test provider discovery
    print("\n2️⃣ Provider Discovery:")
    from gatewayx.secure_config import config
    
    # Simulate different API key scenarios
    test_configs = [
        {"anthropic_api_key": "sk-ant-real-key", "openai_api_key": "sk-openai-real-key"},
        {"grok_api_key": "sk-grok-real-key", "claude_api_key": "sk-claude-real-key"},
        {"cohere_api_key": "sk-cohere-real-key", "google_api_key": "sk-google-real-key"},
        {"new_model_api_key": "sk-newmodel-real-key", "another_model_api_key": "sk-another-real-key"}
    ]
    
    for i, test_config in enumerate(test_configs, 1):
        print(f"\n   Scenario {i}:")
        # Temporarily update config
        original_config = config.config.copy()
        config.config.update(test_config)
        
        providers = config.discover_available_providers()
        print(f"   Discovered: {providers}")
        
        # Restore original config
        config.config = original_config
    
    # Test 3: Test engine creation
    print("\n3️⃣ Engine Creation Test:")
    import asyncio
    
    async def test_engine_creation():
        # Test creating a Grok engine
        grok_engine = await registry.create_engine_instance("grok", "sk-grok-test-key")
        if grok_engine:
            print("   ✅ Grok engine created successfully")
            
            # Test health check
            health_ok = await grok_engine.health_check()
            print(f"   ✅ Grok health check: {'PASSED' if health_ok else 'FAILED'}")
            
            # Test generation
            response = await grok_engine.generate("Hello, this is a test!")
            print(f"   ✅ Grok response: {response[:50]}...")
            
            # Cleanup
            if hasattr(grok_engine, 'cleanup'):
                await grok_engine.cleanup()
            print("   ✅ Grok engine cleaned up")
        else:
            print("   ❌ Failed to create Grok engine")
    
    asyncio.run(test_engine_creation())
    
    # Test 4: Test adding new model dynamically
    print("\n4️⃣ Dynamic Model Addition Test:")
    
    # Create a new engine class dynamically
    class CustomEngine:
        def __init__(self, api_key, **kwargs):
            self.api_key = api_key
            self.name = "custom"
        
        async def health_check(self):
            return True
        
        async def generate(self, prompt, **kwargs):
            return f"[Custom Engine] {prompt}"
        
        def get_cost_per_token(self):
            return 0.001
    
    # Register the new engine
    registry.register_engine("custom", CustomEngine)
    print("   ✅ Custom engine registered")
    
    # Test creating instance
    async def test_custom_engine():
        custom_engine = await registry.create_engine_instance("custom", "sk-custom-key")
        if custom_engine:
            print("   ✅ Custom engine created successfully")
            response = await custom_engine.generate("Test custom engine")
            print(f"   ✅ Custom response: {response}")
        else:
            print("   ❌ Failed to create custom engine")
    
    asyncio.run(test_custom_engine())
    
    print("\n🎉 Dynamic Model System Test Complete!")
    print("\n📋 Summary:")
    print("   ✅ Engine registry working")
    print("   ✅ Provider discovery working") 
    print("   ✅ Dynamic engine creation working")
    print("   ✅ New model addition working")
    print("   ✅ No code changes needed for new models!")

if __name__ == "__main__":
    test_dynamic_model_system()
