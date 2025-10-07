#!/usr/bin/env python3
"""Test script to verify Gateway X integration."""

import asyncio
import json
import requests
import sys
import os

# Add the gateway-x directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'gateway-x'))

from app.ai_engines import MultiEngineClient
from app.config import CONFIG

async def test_ai_engines():
    """Test AI engines individually."""
    print("🤖 Testing AI Engines...")
    
    client = MultiEngineClient(CONFIG)
    results = await client.test_all_engines()
    
    print(f"Engine Status: {results}")
    working_engines = [name for name, status in results.items() if status]
    print(f"Working engines: {working_engines}")
    
    return len(working_engines) > 0

def test_api_endpoints():
    """Test API endpoints."""
    print("\n🌐 Testing API Endpoints...")
    
    base_url = "http://localhost:8000"
    
    # Test health
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"Health check: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"Health check failed: {e}")
        return False
    
    # Test query
    try:
        query_data = {
            "query": "What is 2+2?",
            "budget": 2,
            "confidence_threshold": 0.8
        }
        response = requests.post(f"{base_url}/query", json=query_data, timeout=30)
        result = response.json()
        print(f"Query test: {response.status_code}")
        print(f"  - Best claim: {result['best_claim'][:50]}...")
        print(f"  - Confidence: {result['confidence']:.2f}")
        print(f"  - Rounds: {result['rounds']}")
        print(f"  - Stop reason: {result['stop_reason']}")
        return True
    except Exception as e:
        print(f"Query test failed: {e}")
        return False

def test_frontend():
    """Test frontend accessibility."""
    print("\n🎨 Testing Frontend...")
    
    try:
        response = requests.get("http://localhost:8000/", timeout=5)
        if response.status_code == 200 and "Gateway X" in response.text:
            print("Frontend: ✅ Accessible")
            return True
        else:
            print(f"Frontend: ❌ Status {response.status_code}")
            return False
    except Exception as e:
        print(f"Frontend: ❌ Error - {e}")
        return False

async def main():
    """Run all tests."""
    print("🚀 Gateway X Integration Test")
    print("=" * 40)
    
    # Test AI engines
    engines_ok = await test_ai_engines()
    
    # Test API
    api_ok = test_api_endpoints()
    
    # Test frontend
    frontend_ok = test_frontend()
    
    print("\n📊 Test Results:")
    print(f"  AI Engines: {'✅' if engines_ok else '❌'}")
    print(f"  API: {'✅' if api_ok else '❌'}")
    print(f"  Frontend: {'✅' if frontend_ok else '❌'}")
    
    if all([engines_ok, api_ok, frontend_ok]):
        print("\n🎉 All systems operational! Ready for testing.")
        print("\n🌐 Access your frontend at: http://localhost:8000")
    else:
        print("\n⚠️  Some systems need attention.")
    
    return all([engines_ok, api_ok, frontend_ok])

if __name__ == "__main__":
    asyncio.run(main())
