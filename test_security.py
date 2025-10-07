#!/usr/bin/env python3
"""
Gateway X Security Test Script
Tests the secure configuration system
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

def test_secure_config():
    """Test the secure configuration system"""
    print("🔒 Testing Gateway X Security Configuration")
    print("=" * 50)
    
    # Test 1: Import secure config
    try:
        from gatewayx.secure_config import config
        print("✅ Secure config module imported successfully")
    except Exception as e:
        print(f"❌ Failed to import secure config: {e}")
        return False
    
    # Test 2: Check configuration loading
    try:
        config_summary = config.get_config_summary()
        print("✅ Configuration loaded successfully")
        print(f"   - Server: {config_summary['server']}")
        print(f"   - Engines: {config_summary['engines']}")
        print(f"   - Security: {config_summary['security']}")
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return False
    
    # Test 3: Test API key retrieval
    try:
        providers = ['anthropic', 'openai', 'google', 'cohere']
        for provider in providers:
            key = config.get_api_key(provider)
            status = "✅ Configured" if key else "⚠️ Not configured"
            print(f"   - {provider.title()}: {status}")
    except Exception as e:
        print(f"❌ Failed to test API key retrieval: {e}")
        return False
    
    # Test 4: Test secure mode detection
    try:
        secure_mode = config.is_secure_mode()
        print(f"   - Secure mode: {'✅ Enabled' if secure_mode else '⚠️ Disabled'}")
    except Exception as e:
        print(f"❌ Failed to test secure mode: {e}")
        return False
    
    # Test 5: Test file permissions
    try:
        local_secrets_path = "config/secrets/.env.local"
        if os.path.exists(local_secrets_path):
            stat = os.stat(local_secrets_path)
            permissions = oct(stat.st_mode)[-3:]
            print(f"   - Local secrets permissions: {permissions}")
            if permissions == "600":
                print("   - ✅ Permissions are secure (600)")
            else:
                print("   - ⚠️ Permissions should be 600 for security")
        else:
            print("   - ⚠️ Local secrets file not found")
    except Exception as e:
        print(f"❌ Failed to test file permissions: {e}")
        return False
    
    print("\n✅ Security tests completed successfully!")
    return True

def test_gitignore():
    """Test that secrets are properly ignored by git"""
    print("\n🔍 Testing Git Ignore Configuration")
    print("=" * 40)
    
    try:
        # Check if .gitignore exists
        if not os.path.exists(".gitignore"):
            print("❌ .gitignore file not found")
            return False
        
        # Read .gitignore content
        with open(".gitignore", "r") as f:
            gitignore_content = f.read()
        
        # Check for security patterns
        security_patterns = [
            "config/secrets/",
            "*.key",
            "secrets/",
            ".secrets/"
        ]
        
        for pattern in security_patterns:
            if pattern in gitignore_content:
                print(f"✅ Pattern found in .gitignore: {pattern}")
            else:
                print(f"⚠️ Pattern missing from .gitignore: {pattern}")
        
        print("✅ Git ignore configuration test completed")
        return True
        
    except Exception as e:
        print(f"❌ Failed to test git ignore: {e}")
        return False

def test_server_security():
    """Test server security endpoints"""
    print("\n🌐 Testing Server Security Endpoints")
    print("=" * 40)
    
    try:
        import requests
        import time
        
        # Start server in background (if not already running)
        print("   - Checking if server is running...")
        
        try:
            response = requests.get("http://127.0.0.1:3001/health", timeout=5)
            if response.status_code == 200:
                print("   - ✅ Server is running")
            else:
                print("   - ⚠️ Server responded with status:", response.status_code)
        except requests.exceptions.ConnectionError:
            print("   - ⚠️ Server not running. Start it with:")
            print("     python3 -m uvicorn src.gatewayx.server:app --reload --port 3001")
            return False
        
        # Test security endpoint
        try:
            response = requests.get("http://127.0.0.1:3001/security/status", timeout=5)
            if response.status_code == 200:
                security_data = response.json()
                print("   - ✅ Security endpoint accessible")
                print(f"   - Status: {security_data.get('status', 'unknown')}")
                print(f"   - Secure mode: {security_data.get('secure_mode', False)}")
                print(f"   - Local secrets: {security_data.get('local_secrets_loaded', False)}")
            else:
                print(f"   - ⚠️ Security endpoint returned status: {response.status_code}")
        except Exception as e:
            print(f"   - ❌ Failed to test security endpoint: {e}")
            return False
        
        print("✅ Server security tests completed")
        return True
        
    except ImportError:
        print("   - ⚠️ requests library not available. Install with: pip install requests")
        return False
    except Exception as e:
        print(f"   - ❌ Server security test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Gateway X Security Test Suite")
    print("=" * 50)
    
    tests = [
        ("Secure Configuration", test_secure_config),
        ("Git Ignore", test_gitignore),
        ("Server Security", test_server_security),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All security tests passed!")
        return 0
    else:
        print("⚠️ Some tests failed. Please review the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
