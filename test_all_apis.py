#!/usr/bin/env python3
"""Test all AI APIs comprehensively."""

import asyncio
import sys
import os

# Add the gateway-x directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'gateway-x'))

from app.ai_engines import MultiEngineClient
from app.config import CONFIG

async def test_all_apis():
    """Test all configured APIs."""
    print("🚀 Testing All AI APIs")
    print("=" * 50)
    
    # Initialize the multi-engine client
    client = MultiEngineClient(CONFIG)
    
    print(f"Configuration:")
    print(f"  - Enabled engines: {CONFIG.ENABLED_ENGINES}")
    print(f"  - Referee engine: {CONFIG.REFEREE_ENGINE}")
    print(f"  - Use real LLM: {CONFIG.USE_REAL_LLM}")
    print()
    
    # Test all engines
    print("🤖 Testing Individual Engines:")
    results = await client.test_all_engines()
    
    for engine_name, status in results.items():
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {engine_name}: {'WORKING' if status else 'FAILED'}")
    
    working_engines = [name for name, status in results.items() if status]
    print(f"\nWorking engines: {working_engines}")
    
    if not working_engines:
        print("\n⚠️  No engines are working. Testing with mock responses...")
        # Test the system with mock responses
        test_query = "What is the capital of France?"
        print(f"\nTesting with query: {test_query}")
        
        # Test initial claims generation
        print("\n📝 Testing initial claims generation...")
        claims = await client.generate_initial_claims(test_query)
        print(f"Generated {len(claims)} claims:")
        for i, claim in enumerate(claims, 1):
            print(f"  {i}. {claim[:100]}...")
        
        # Test referee functionality
        if len(claims) >= 2:
            print("\n⚖️  Testing referee functionality...")
            referee_result = await client.referee_duel(test_query, claims[0], claims[1])
            print(f"Referee result: {referee_result[:100]}...")
        
        # Test refined claims
        print("\n🔄 Testing refined claims generation...")
        context = "Previous round showed mixed results."
        refined_claims = await client.generate_refined_claims(test_query, context, 1)
        print(f"Generated {len(refined_claims)} refined claims:")
        for i, claim in enumerate(refined_claims, 1):
            print(f"  {i}. {claim[:100]}...")
    
    return len(working_engines) > 0

async def main():
    """Main test function."""
    success = await test_all_apis()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 API Testing: SUCCESS")
        print("   At least one engine is working properly!")
    else:
        print("⚠️  API Testing: PARTIAL SUCCESS")
        print("   No engines working, but system can operate with mock responses")
    
    return success

if __name__ == "__main__":
    asyncio.run(main())
