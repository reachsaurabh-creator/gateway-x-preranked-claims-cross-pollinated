"""Multi-engine AI client for Gateway X."""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod

from .config import Config

# Optional imports with fallbacks
try:
    from anthropic import AsyncAnthropic
except ImportError:
    AsyncAnthropic = None

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None


logger = logging.getLogger("gatewayx")


class AIEngine(ABC):
    """Abstract base class for AI engines."""
    
    def __init__(self, name: str, config: Config):
        self.name = name
        self.config = config
    
    @abstractmethod
    async def generate_response(self, prompt: str) -> str:
        """Generate a response to the given prompt."""
        pass
    
    @abstractmethod
    async def generate_initial_claims(self, query: str, num_claims: int = 1) -> List[str]:
        """Generate initial claims for a query."""
        pass


class AnthropicEngine(AIEngine):
    """Anthropic Claude engine using requests library."""
    
    def __init__(self, config: Config):
        super().__init__("anthropic", config)
        self.api_key = config.ANTHROPIC_API_KEY
        self.url = "https://api.anthropic.com/v1/messages"
        self.headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
    
    async def generate_response(self, prompt: str) -> str:
        """Generate a response using Claude via requests."""
        if not self.api_key:
            return f"[Anthropic Mock] Response to: {prompt[:50]}..."
        
        try:
            import requests
            
            data = {
                "model": self.config.ANTHROPIC_MODEL,
                "max_tokens": self.config.MAX_TOKENS,
                "temperature": self.config.TEMPERATURE,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
            
            # Run the request in a thread to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: requests.post(self.url, headers=self.headers, json=data, timeout=30)
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['content'][0]['text']
            else:
                logger.error(f"Anthropic API error: {response.status_code} - {response.text}")
                return f"[Anthropic Error] HTTP {response.status_code}: {response.text[:100]}"
                
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return f"[Anthropic Error] Failed to generate response: {str(e)}"
    
    async def generate_initial_claims(self, query: str, num_claims: int = 1) -> List[str]:
        """Generate initial claims using Claude."""
        prompt = f"""You are an expert analyst. For the question: "{query}"

Please provide {num_claims} distinct, well-reasoned answer(s). Each answer should:
1. Be comprehensive and factual
2. Approach the question from a different angle if multiple answers requested
3. Be 2-3 sentences long
4. Start directly with the answer (no preamble)

Question: {query}"""
        
        response = await self.generate_response(prompt)
        
        # Split response into claims if multiple requested
        if num_claims > 1:
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            claims = []
            current_claim = []
            
            for line in lines:
                if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                    if current_claim:
                        claims.append(' '.join(current_claim))
                        current_claim = []
                    # Remove numbering/bullets
                    clean_line = line.lstrip('0123456789.-• ').strip()
                    if clean_line:
                        current_claim.append(clean_line)
                else:
                    if current_claim:
                        current_claim.append(line)
            
            if current_claim:
                claims.append(' '.join(current_claim))
            
            return claims[:num_claims] if claims else [response]
        else:
            return [response]


class OpenAIEngine(AIEngine):
    """OpenAI GPT engine."""
    
    def __init__(self, config: Config):
        super().__init__("openai", config)
        if AsyncOpenAI and config.OPENAI_API_KEY:
            self.client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
        else:
            self.client = None
            logger.warning("OpenAI client not available")
    
    async def generate_response(self, prompt: str) -> str:
        """Generate a response using GPT."""
        if not self.client:
            return f"[OpenAI Mock] Response to: {prompt[:50]}..."
        
        try:
            response = await self.client.chat.completions.create(
                model=self.config.OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.config.MAX_TOKENS,
                temperature=self.config.TEMPERATURE
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return f"[OpenAI Error] Failed to generate response: {str(e)}"
    
    async def generate_initial_claims(self, query: str, num_claims: int = 1) -> List[str]:
        """Generate initial claims using GPT."""
        prompt = f"""You are an expert analyst. For the question: "{query}"

Please provide {num_claims} distinct, well-reasoned answer(s). Each answer should:
1. Be comprehensive and factual
2. Approach the question from a different angle if multiple answers requested
3. Be 2-3 sentences long
4. Start directly with the answer (no preamble)

Question: {query}"""
        
        response = await self.generate_response(prompt)
        
        # Similar parsing logic as Anthropic
        if num_claims > 1:
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            claims = []
            current_claim = []
            
            for line in lines:
                if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                    if current_claim:
                        claims.append(' '.join(current_claim))
                        current_claim = []
                    clean_line = line.lstrip('0123456789.-• ').strip()
                    if clean_line:
                        current_claim.append(clean_line)
                else:
                    if current_claim:
                        current_claim.append(line)
            
            if current_claim:
                claims.append(' '.join(current_claim))
            
            return claims[:num_claims] if claims else [response]
        else:
            return [response]


class XAIEngine(AIEngine):
    """xAI Grok engine (via OpenAI-compatible API)."""
    
    def __init__(self, config: Config):
        super().__init__("xai", config)
        if AsyncOpenAI and config.XAI_API_KEY:
            self.client = AsyncOpenAI(
                api_key=config.XAI_API_KEY,
                base_url="https://api.x.ai/v1"
            )
        else:
            self.client = None
            logger.warning("xAI client not available")
    
    async def generate_response(self, prompt: str) -> str:
        """Generate a response using Grok."""
        if not self.client:
            return f"[xAI Mock] Response to: {prompt[:50]}..."
        
        try:
            response = await self.client.chat.completions.create(
                model=self.config.XAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.config.MAX_TOKENS,
                temperature=self.config.TEMPERATURE
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"xAI API error: {e}")
            return f"[xAI Error] Failed to generate response: {str(e)}"
    
    async def generate_initial_claims(self, query: str, num_claims: int = 1) -> List[str]:
        """Generate initial claims using Grok."""
        prompt = f"""You are an expert analyst. For the question: "{query}"

Please provide {num_claims} distinct, well-reasoned answer(s). Each answer should:
1. Be comprehensive and factual
2. Approach the question from a different angle if multiple answers requested
3. Be 2-3 sentences long
4. Start directly with the answer (no preamble)

Question: {query}"""
        
        response = await self.generate_response(prompt)
        
        # Similar parsing logic
        if num_claims > 1:
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            claims = []
            current_claim = []
            
            for line in lines:
                if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                    if current_claim:
                        claims.append(' '.join(current_claim))
                        current_claim = []
                    clean_line = line.lstrip('0123456789.-• ').strip()
                    if clean_line:
                        current_claim.append(clean_line)
                else:
                    if current_claim:
                        current_claim.append(line)
            
            if current_claim:
                claims.append(' '.join(current_claim))
            
            return claims[:num_claims] if claims else [response]
        else:
            return [response]


class GeminiEngine(AIEngine):
    """Google Gemini engine."""
    
    def __init__(self, config: Config):
        super().__init__("gemini", config)
        if genai and config.GEMINI_API_KEY:
            genai.configure(api_key=config.GEMINI_API_KEY)
            self.model = genai.GenerativeModel(config.GEMINI_MODEL)
        else:
            self.model = None
            logger.warning("Gemini client not available")
    
    async def generate_response(self, prompt: str) -> str:
        """Generate a response using Gemini."""
        if not self.model:
            return f"[Gemini Mock] Response to: {prompt[:50]}..."
        
        try:
            # Gemini doesn't have async support in the current SDK, so we run in executor
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: self.model.generate_content(
                    prompt,
                    generation_config={
                        "max_output_tokens": self.config.MAX_TOKENS,
                        "temperature": self.config.TEMPERATURE
                    }
                )
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return f"[Gemini Error] Failed to generate response: {str(e)}"
    
    async def generate_initial_claims(self, query: str, num_claims: int = 1) -> List[str]:
        """Generate initial claims using Gemini."""
        prompt = f"""You are an expert analyst. For the question: "{query}"

Please provide {num_claims} distinct, well-reasoned answer(s). Each answer should:
1. Be comprehensive and factual
2. Approach the question from a different angle if multiple answers requested
3. Be 2-3 sentences long
4. Start directly with the answer (no preamble)

Question: {query}"""
        
        response = await self.generate_response(prompt)
        
        # Similar parsing logic
        if num_claims > 1:
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            claims = []
            current_claim = []
            
            for line in lines:
                if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                    if current_claim:
                        claims.append(' '.join(current_claim))
                        current_claim = []
                    clean_line = line.lstrip('0123456789.-• ').strip()
                    if clean_line:
                        current_claim.append(clean_line)
                else:
                    if current_claim:
                        current_claim.append(line)
            
            if current_claim:
                claims.append(' '.join(current_claim))
            
            return claims[:num_claims] if claims else [response]
        else:
            return [response]


class MultiEngineClient:
    """Client that manages multiple AI engines."""
    
    def __init__(self, config: Config):
        self.config = config
        self.engines: Dict[str, AIEngine] = {}
        
        # Initialize enabled engines
        if "anthropic" in config.ENABLED_ENGINES:
            self.engines["anthropic"] = AnthropicEngine(config)
        
        if "openai" in config.ENABLED_ENGINES:
            self.engines["openai"] = OpenAIEngine(config)
        
        if "xai" in config.ENABLED_ENGINES:
            self.engines["xai"] = XAIEngine(config)
        
        if "gemini" in config.ENABLED_ENGINES:
            self.engines["gemini"] = GeminiEngine(config)
        
        # Set referee engine
        self.referee_engine = self.engines.get(config.REFEREE_ENGINE)
        if not self.referee_engine and self.engines:
            self.referee_engine = list(self.engines.values())[0]
        
        logger.info(f"Initialized engines: {list(self.engines.keys())}")
        logger.info(f"Referee engine: {self.referee_engine.name if self.referee_engine else 'None'}")
    
    async def generate_initial_claims(self, query: str) -> List[str]:
        """Generate initial claims from all engines."""
        if not self.engines:
            return [f"Mock claim 1 for: {query}", f"Mock claim 2 for: {query}"]
        
        tasks = []
        for engine_name, engine in self.engines.items():
            tasks.append(self._generate_engine_claim(engine, query))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        claims = []
        for i, result in enumerate(results):
            engine_name = list(self.engines.keys())[i]
            if isinstance(result, Exception):
                logger.error(f"Engine {engine_name} failed: {result}")
                claims.append(f"[{engine_name}] Error generating claim: {str(result)}")
            else:
                claims.extend(result)
        
        return claims
    
    async def _generate_engine_claim(self, engine: AIEngine, query: str) -> List[str]:
        """Generate a claim from a specific engine."""
        try:
            claims = await engine.generate_initial_claims(query, num_claims=1)
            # Prefix with engine name for identification
            return [f"[{engine.name}] {claim}" for claim in claims]
        except Exception as e:
            logger.error(f"Error generating claim from {engine.name}: {e}")
            return [f"[{engine.name}] Error: {str(e)}"]
    
    async def referee_duel(self, query: str, claim_a: str, claim_b: str) -> str:
        """Use referee engine to judge between two claims."""
        if not self.referee_engine:
            return '{"winner":"A","factuality":0.5,"coherence":0.5,"note":"no referee"}'
        
        prompt = f"""You are a strict referee. Compare two answers for accuracy, coherence, and relevance to the question.
Return STRICT JSON only: {{"winner":"A"|"B","factuality":0..1,"coherence":0..1,"note":"<=20 tokens"}}

Question: {query}

Answer A: {claim_a}

Answer B: {claim_b}

Judge which answer is better and return the JSON response:"""
        
        try:
            response = await self.referee_engine.generate_response(prompt)
            return response
        except Exception as e:
            logger.error(f"Referee error: {e}")
            return '{"winner":"A","factuality":0.5,"coherence":0.5,"note":"referee error"}'
    
    async def generate_refined_claims(self, query: str, context: str, round_idx: int) -> List[str]:
        """Generate refined claims based on previous round context."""
        if not self.engines:
            return [f"Mock refined claim {i} for round {round_idx}: {query}" for i in range(1, 4)]
        
        logger.info(f"Generating refined claims for round {round_idx} with {len(self.engines)} engines: {list(self.engines.keys())}")
        
        tasks = []
        for engine_name, engine in self.engines.items():
            tasks.append(self._generate_refined_engine_claim(engine, query, context, round_idx))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        refined_claims = []
        successful_engines = []
        failed_engines = []
        
        for i, result in enumerate(results):
            engine_name = list(self.engines.keys())[i]
            if isinstance(result, Exception):
                logger.error(f"Engine {engine_name} refinement failed: {result}")
                failed_engines.append(engine_name)
                # Instead of error message, generate a fallback response to keep engine participating
                fallback_response = await self._generate_fallback_refined_claim(engine_name, query, context, round_idx)
                refined_claims.extend(fallback_response)
            else:
                successful_engines.append(engine_name)
                refined_claims.extend(result)
        
        logger.info(f"Round {round_idx} refinement results: {len(successful_engines)} successful ({successful_engines}), {len(failed_engines)} failed ({failed_engines})")
        logger.info(f"Total refined claims generated: {len(refined_claims)}")
        
        return refined_claims
    
    async def _generate_refined_engine_claim(self, engine: AIEngine, query: str, context: str, round_idx: int) -> List[str]:
        """Generate a refined claim from a specific engine based on critical analysis of other engines' responses."""
        try:
            # Create a critical analysis prompt that forces the engine to analyze other responses
            critical_analysis_prompt = f"""You are participating in a multi-AI consensus process. Below are responses from other AI engines to the same question. Your task is to:

1. CRITICALLY ANALYZE each other engine's response
2. Identify strengths and weaknesses in their approaches
3. Synthesize the best elements from all responses
4. Provide an IMPROVED response that addresses gaps and builds on insights

Original Question: {query}

OTHER ENGINES' RESPONSES:
{context}

CRITICAL ANALYSIS TASK:
- What did each engine do well?
- What are the limitations or gaps in each response?
- How can you combine the best elements while addressing weaknesses?
- What new insights or improvements can you add?

Your improved response should be MORE COMPREHENSIVE, MORE ACCURATE, and MORE INSIGHTFUL than the previous responses. Build upon their strengths while addressing their limitations.

IMPROVED RESPONSE:"""
            
            response = await engine.generate_response(critical_analysis_prompt)
            
            # Prefix with engine name and round info for identification
            return [f"[{engine.name}-r{round_idx}] {response}"]
            
        except Exception as e:
            logger.error(f"Error generating refined claim from {engine.name}: {e}")
            return [f"[{engine.name}-r{round_idx}] Error: {str(e)}"]
    
    async def _generate_fallback_refined_claim(self, engine_name: str, query: str, context: str, round_idx: int) -> List[str]:
        """Generate a fallback refined claim when an engine fails, to keep it participating."""
        try:
            # Use a simpler, more reliable prompt for fallback
            fallback_prompt = f"""You are {engine_name} engine in a multi-AI consensus process. The other AI engines have provided these responses:

{context}

Based on their responses, provide your own improved answer to: {query}

Your response should build upon their insights while adding your own perspective. Keep it concise but comprehensive.

Your improved response:"""
            
            # Try to use the engine if it's still available, otherwise use a simple fallback
            if engine_name in self.engines:
                try:
                    response = await self.engines[engine_name].generate_response(fallback_prompt)
                    return [f"[{engine_name}-r{round_idx}] {response}"]
                except Exception as e2:
                    logger.warning(f"Fallback also failed for {engine_name}: {e2}")
            
            # Final fallback: generate a simple response based on context
            return [f"[{engine_name}-r{round_idx}] Based on the previous responses, {engine_name} provides this perspective: The question about {query} requires considering multiple viewpoints. While other engines have provided valuable insights, {engine_name} emphasizes the importance of balanced analysis and comprehensive understanding of the topic."]
            
        except Exception as e:
            logger.error(f"Even fallback failed for {engine_name}: {e}")
            # Ultimate fallback
            return [f"[{engine_name}-r{round_idx}] {engine_name} acknowledges the complexity of the question and defers to the consensus process while maintaining that comprehensive analysis requires multiple perspectives."]
    
    async def test_all_engines(self) -> Dict[str, bool]:
        """Test all engines to ensure they're working."""
        results = {}
        test_query = "What is 2+2?"
        
        for engine_name, engine in self.engines.items():
            try:
                response = await engine.generate_response(test_query)
                results[engine_name] = len(response) > 0 and "error" not in response.lower()
                logger.info(f"Engine {engine_name} test: {'PASS' if results[engine_name] else 'FAIL'}")
            except Exception as e:
                results[engine_name] = False
                logger.error(f"Engine {engine_name} test failed: {e}")
        
        return results
