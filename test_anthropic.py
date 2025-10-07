#!/usr/bin/env python3
"""Test Anthropic API directly to debug the issue."""

import asyncio
import sys
import os

# Add the gateway-x directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'gateway-x'))

from app.ai_engines import AnthropicEngine
from app.config import CONFIG

async def test_anthropic_directly():
    """Test Anthropic API directly."""
    print("🔍 Testing Anthropic API directly...")
    print(f"API Key: {CONFIG.ANTHROPIC_API_KEY[:20]}...")
    print(f"Model: {CONFIG.ANTHROPIC_MODEL}")
    print(f"Max Tokens: {CONFIG.MAX_TOKENS}")
    print(f"Temperature: {CONFIG.TEMPERATURE}")
    print()
    
    engine = AnthropicEngine(CONFIG)
    print(f"Client available: {engine.client is not None}")
    
    if engine.client:
        try:
            print("Testing simple response...")
            response = await engine.generate_response("What is 2+2?")
            print(f"✅ Response: {response}")
            return True
        except Exception as e:
            print(f"❌ Error: {e}")
            print(f"Error type: {type(e)}")
            return False
    else:
        print("❌ No client available")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_anthropic_directly())
    print(f"\nResult: {'✅ SUCCESS' if result else '❌ FAILED'}")
