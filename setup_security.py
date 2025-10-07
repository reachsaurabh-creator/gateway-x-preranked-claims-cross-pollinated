#!/usr/bin/env python3
"""
Gateway X Security Setup Script
Helps users configure API keys securely
"""

import os
import shutil
import sys
from pathlib import Path

def main():
    """Main setup function"""
    print("🔒 Gateway X Security Setup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("src/gatewayx"):
        print("❌ Error: Please run this script from the gateway-x-consensus root directory")
        sys.exit(1)
    
    # Create necessary directories
    os.makedirs("config/secrets", exist_ok=True)
    os.makedirs("config/templates", exist_ok=True)
    
    # Check if local secrets already exist
    local_secrets_path = "config/secrets/.env.local"
    if os.path.exists(local_secrets_path):
        print(f"✅ Local secrets file already exists: {local_secrets_path}")
        response = input("Do you want to overwrite it? (y/N): ").strip().lower()
        if response != 'y':
            print("ℹ️ Keeping existing local secrets file")
            return
    
    # Copy template to local secrets
    template_path = "config/templates/.env.template"
    if os.path.exists(template_path):
        shutil.copy2(template_path, local_secrets_path)
        print(f"✅ Created local secrets file: {local_secrets_path}")
    else:
        print("❌ Template file not found. Creating basic template...")
        create_basic_template(local_secrets_path)
    
    # Set proper permissions (readable only by owner)
    os.chmod(local_secrets_path, 0o600)
    print(f"🔒 Set secure permissions (600) on {local_secrets_path}")
    
    # Instructions
    print("\n📋 Next Steps:")
    print("1. Edit the file: config/secrets/.env.local")
    print("2. Add your real API keys (replace placeholder values)")
    print("3. Set GATEWAYX_USE_REAL_LLM=true when ready")
    print("4. Run the server: python3 -m uvicorn src.gatewayx.server:app --reload --port 3001")
    
    print("\n🔐 Security Notes:")
    print("- The config/secrets/ directory is in .gitignore")
    print("- Your API keys will NEVER be committed to git")
    print("- Only you can read the local secrets file")
    print("- Use different keys for development and production")
    
    print("\n✅ Security setup complete!")

def create_basic_template(path):
    """Create a basic template if the main one doesn't exist"""
    template_content = """# LOCAL API KEYS - NEVER COMMIT TO GIT
# Add your real API keys here

# Server Configuration
GATEWAYX_SERVER_PORT=3001
GATEWAYX_SERVER_HOST=0.0.0.0
GATEWAYX_LOG_LEVEL=INFO

# Multi-Engine Mode
GATEWAYX_MULTI_ENGINE_MODE=true
GATEWAYX_LOAD_BALANCING_STRATEGY=weighted
GATEWAYX_ENABLE_CONSENSUS_JUDGING=true
GATEWAYX_CONSENSUS_JUDGE_COUNT=3

# Engine API Keys - ADD YOUR REAL KEYS HERE
GATEWAYX_ANTHROPIC_API_KEY=your_anthropic_key_here
GATEWAYX_OPENAI_API_KEY=your_openai_key_here
GATEWAYX_GOOGLE_API_KEY=your_google_key_here
GATEWAYX_COHERE_API_KEY=your_cohere_key_here

# Use Real LLM Mode (set to true when you have real keys)
GATEWAYX_USE_REAL_LLM=false

# Orchestration Settings
GATEWAYX_DEFAULT_BUDGET=20
GATEWAYX_MAX_BUDGET=200
GATEWAYX_BATCH_SIZE=3
GATEWAYX_MIN_ROUNDS=3
GATEWAYX_CONFIDENCE_THRESHOLD=0.95

# Statistical Settings
GATEWAYX_USE_BOOTSTRAP_CI=true
GATEWAYX_CI_MIN_ROUNDS=6
GATEWAYX_CI_BOOTSTRAP_SAMPLES=200
GATEWAYX_CI_SEPARATION_MIN=0.05

# Response Logging
GATEWAYX_PERSIST_RESPONSES=true
GATEWAYX_RESPONSE_LOG_FILE=data/logs/responses.jsonl
GATEWAYX_MAX_RESPONSES_IN_MEMORY=10000
"""
    
    with open(path, 'w') as f:
        f.write(template_content)

if __name__ == "__main__":
    main()
