# 🔄 Dynamic Model System

The Gateway X system now supports **dynamic model discovery and management** - you can add new AI models without changing any code!

## 🚀 How It Works

### 1. **Automatic Model Discovery**
The system automatically discovers available models by:
- Scanning the `src/gatewayx/engines/` directory for new engine classes
- Reading API keys from `config/secrets/.env.local`
- Matching API keys to available engines

### 2. **Zero-Code Model Addition**
To add a new model (e.g., Grok):
1. Add the API key to `config/secrets/.env.local`:
   ```bash
   GATEWAYX_GROK_API_KEY=your_real_grok_key_here
   ```
2. Create the engine class in `src/gatewayx/engines/grok_engine.py`
3. Restart the server - the model is automatically available!

### 3. **Dynamic Provider Detection**
The system automatically detects which providers have valid API keys and only initializes those engines.

## 📁 File Structure

```
src/gatewayx/
├── engines/                    # Dynamic engines directory
│   ├── __init__.py            # Auto-discovers engines
│   └── grok_engine.py         # Example: Grok engine
├── model_registry.py          # Dynamic model registry
├── secure_config.py           # Dynamic config discovery
└── engine_pool.py             # Updated to use dynamic discovery
```

## 🔧 Adding a New Model

### Step 1: Create Engine Class
Create `src/gatewayx/engines/your_model_engine.py`:

```python
"""
Your Model Engine
"""

import asyncio
import logging
from ..llm_engines import LLMEngine

logger = logging.getLogger(__name__)

class YourModelEngine(LLMEngine):
    """Your Model AI Engine"""
    
    def __init__(self, api_key: str, **kwargs):
        super().__init__(api_key, **kwargs)
        self.name = "your_model"
        self.base_url = "https://api.yourmodel.com/v1"
        self.model_name = "your-model-name"
        self.cost_per_token = 0.0001
    
    async def health_check(self) -> bool:
        """Check if engine is working"""
        try:
            # Your health check logic
            await asyncio.sleep(0.1)
            logger.info("✅ Your Model engine health check passed")
            return True
        except Exception as e:
            logger.error(f"❌ Your Model engine health check failed: {e}")
            return False
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using Your Model"""
        try:
            # Your API call logic
            await asyncio.sleep(0.5)
            response = f"[Your Model Response] {prompt[:50]}..."
            logger.info(f"✅ Your Model generated response")
            return response
        except Exception as e:
            logger.error(f"❌ Your Model generation failed: {e}")
            return f"Error generating response: {e}"
    
    async def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response (alias for compatibility)"""
        return await self.generate(prompt, **kwargs)
    
    def get_cost_per_token(self) -> float:
        """Get cost per token"""
        return self.cost_per_token
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("🧹 Your Model engine cleanup completed")
```

### Step 2: Add API Key
Add to `config/secrets/.env.local`:
```bash
GATEWAYX_YOUR_MODEL_API_KEY=your_real_api_key_here
```

### Step 3: Update Engine Package
Add to `src/gatewayx/engines/__init__.py`:
```python
from .your_model_engine import YourModelEngine

__all__ = ['YourModelEngine']
```

### Step 4: Restart Server
```bash
python3 -m uvicorn src.gatewayx.server:app --reload --port 3001
```

## 🧪 Testing Dynamic Models

Run the test suite:
```bash
python3 test_dynamic_models.py
```

This will test:
- ✅ Engine registry
- ✅ Provider discovery
- ✅ Dynamic engine creation
- ✅ New model addition

## 🔍 Current Available Models

The system currently supports:
- **Built-in Models**: Mock, Anthropic, OpenAI, Google, Cohere
- **Dynamic Models**: Grok (example)
- **Custom Models**: Any model you add following the pattern

## 🛡️ Security Features

- **Local Secrets Only**: API keys stored in `config/secrets/.env.local`
- **Never Committed**: Secrets directory in `.gitignore`
- **Secure Permissions**: Files readable only by owner
- **Dynamic Discovery**: No hardcoded model lists

## 📊 Monitoring

Check available models:
```bash
curl http://127.0.0.1:3001/security/status
```

Check engine status:
```bash
curl http://127.0.0.1:3001/engines/status
```

## 🎯 Benefits

1. **Zero Code Changes**: Add models without touching core code
2. **Automatic Discovery**: System finds new models automatically
3. **Secure**: API keys never leave your local machine
4. **Flexible**: Support any LLM provider
5. **Maintainable**: Clean separation of concerns

## 🔄 Example: Adding Grok

1. **API Key**: `GATEWAYX_GROK_API_KEY=sk-grok-...`
2. **Engine**: `src/gatewayx/engines/grok_engine.py` ✅
3. **Registry**: Auto-discovered ✅
4. **Usage**: Available immediately ✅

## 🚫 Removing Models

To remove a model:
1. Remove the API key from `config/secrets/.env.local`
2. Delete the engine file (optional)
3. Restart the server

The system will automatically stop using the removed model.

---

**🎉 The dynamic model system is complete and working! You can now add any LLM provider without changing core code.**
