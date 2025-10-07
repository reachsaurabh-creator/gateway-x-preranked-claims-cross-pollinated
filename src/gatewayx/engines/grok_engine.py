"""
Grok Engine - Example of dynamically added model
"""

import asyncio
import logging
from typing import Optional
from ..llm_engines import LLMEngine

logger = logging.getLogger(__name__)

class GrokEngine(LLMEngine):
    """Grok AI Engine - Example of dynamically added model"""
    
    def __init__(self, api_key: str, **kwargs):
        super().__init__(api_key, **kwargs)
        self.name = "grok"
        self.base_url = "https://api.x.ai/v1"  # Example URL
        self.model_name = "grok-beta"
        self.cost_per_token = 0.0001  # Example cost
    
    async def health_check(self) -> bool:
        """Check if Grok engine is working"""
        try:
            # Simulate health check
            await asyncio.sleep(0.1)
            logger.info("✅ Grok engine health check passed")
            return True
        except Exception as e:
            logger.error(f"❌ Grok engine health check failed: {e}")
            return False
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using Grok"""
        try:
            # Simulate API call delay
            await asyncio.sleep(0.5)
            
            # Generate a more realistic response based on the prompt
            # This simulates what Grok would actually respond
            if "president" in prompt.lower() and "us" in prompt.lower():
                response = "Based on historical analysis, many consider Abraham Lincoln, Franklin D. Roosevelt, and George Washington as among the best US presidents. Lincoln preserved the Union and ended slavery, FDR led through the Great Depression and WWII, and Washington established many presidential precedents. The 'best' president often depends on the criteria used - leadership during crisis, policy impact, or constitutional legacy."
            elif "prime minister" in prompt.lower() and "india" in prompt.lower():
                response = "India has had many notable prime ministers. Jawaharlal Nehru, the first PM, established democratic foundations. Indira Gandhi was a strong leader who made India self-reliant. Atal Bihari Vajpayee was known for economic reforms and nuclear tests. Manmohan Singh brought economic liberalization. The 'best' depends on criteria like economic growth, foreign policy, or social reforms."
            elif "2+2" in prompt or "2 + 2" in prompt:
                response = "2 + 2 = 4"
            elif "math" in prompt.lower() or any(op in prompt for op in ['+', '-', '*', '/', '=']):
                # Try to solve simple math
                try:
                    # Extract and solve simple math expressions
                    import re
                    # Find simple arithmetic expressions
                    math_pattern = r'(\d+)\s*([+\-*/])\s*(\d+)'
                    match = re.search(math_pattern, prompt)
                    if match:
                        a, op, b = int(match.group(1)), match.group(2), int(match.group(3))
                        if op == '+':
                            result = a + b
                        elif op == '-':
                            result = a - b
                        elif op == '*':
                            result = a * b
                        elif op == '/':
                            result = a / b
                        response = f"{a} {op} {b} = {result}"
                    else:
                        response = "I can help with math problems. Please provide a clear arithmetic expression."
                except:
                    response = "I can help with mathematical calculations. Please provide a clear math problem."
            else:
                # General response
                response = f"Grok here! Regarding your question about '{prompt[:30]}...', I'd need more context to provide a comprehensive answer. Could you clarify what specific aspect you're most interested in?"
            
            logger.info(f"✅ Grok generated response for prompt: {prompt[:30]}...")
            return response
            
        except Exception as e:
            logger.error(f"❌ Grok generation failed: {e}")
            return f"Error generating response: {e}"
    
    async def generate_response(self, prompt: str, **kwargs) -> dict:
        """Generate response using Grok (alias for compatibility)"""
        start_time = asyncio.get_event_loop().time()
        
        # Generate the response text
        response_text = await self.generate(prompt, **kwargs)
        
        # Calculate tokens (rough estimation)
        tokens = len(response_text.split())
        cost = tokens * self.cost_per_token
        
        # Calculate response time
        response_time = asyncio.get_event_loop().time() - start_time
        
        # Update stats
        self.total_requests += 1
        self.total_tokens += tokens
        self.total_cost += cost
        self.last_request_time = start_time
        
        return {
            "text": response_text,
            "tokens": tokens,
            "cost": cost,
            "engine": self.name,
            "response_time": response_time
        }
    
    def get_cost_per_token(self) -> float:
        """Get cost per token for Grok"""
        return self.cost_per_token
    
    async def cleanup(self):
        """Cleanup Grok engine resources"""
        logger.info("🧹 Grok engine cleanup completed")
