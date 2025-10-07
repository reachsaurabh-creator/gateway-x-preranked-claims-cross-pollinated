"""
Prompt Vault - Centralized prompt management
"""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class PromptVault:
    """Centralized prompt management system"""
    
    def __init__(self):
        self.prompts = {}
        self._load_default_prompts()
    
    def _load_default_prompts(self):
        """Load default prompts"""
        self.prompts = {
            "query_processing": {
                "system": "You are a helpful AI assistant. Provide accurate, helpful, and detailed responses to user questions.",
                "user_template": "{query}"
            },
            "response_evaluation": {
                "system": "You are an expert evaluator. Compare two AI responses to the same question and determine which is better.",
                "user_template": """Question: {query}

Response A: {response_a}

Response B: {response_b}

Which response is better? Consider accuracy, completeness, clarity, and helpfulness. Respond with:
- "A" if Response A is better
- "B" if Response B is better  
- "TIE" if they are equally good

Provide a brief explanation for your choice."""
            },
            "consensus_judge": {
                "system": "You are a consensus judge. Evaluate multiple AI responses and determine the best overall answer.",
                "user_template": """Question: {query}

Responses:
{responses}

Evaluate all responses and provide:
1. The best overall response
2. A confidence score (0-1)
3. Brief reasoning for your choice"""
            },
            "quality_assessment": {
                "system": "You are a quality assessor. Rate the quality of an AI response on multiple dimensions.",
                "user_template": """Question: {query}

Response: {response}

Rate this response on a scale of 1-10 for each dimension:
- Accuracy: How factually correct is the response?
- Completeness: How thoroughly does it answer the question?
- Clarity: How clear and well-structured is the response?
- Helpfulness: How useful is the response to the user?

Provide scores and brief explanations."""
            }
        }
    
    def get_prompt(self, prompt_name: str, **kwargs) -> Optional[Dict[str, str]]:
        """Get a prompt by name with variable substitution"""
        if prompt_name not in self.prompts:
            logger.warning(f"Prompt '{prompt_name}' not found")
            return None
        
        prompt_template = self.prompts[prompt_name]
        
        # Substitute variables in templates
        formatted_prompt = {}
        for key, template in prompt_template.items():
            try:
                formatted_prompt[key] = template.format(**kwargs)
            except KeyError as e:
                logger.error(f"Missing variable {e} for prompt '{prompt_name}'")
                return None
        
        return formatted_prompt
    
    def add_prompt(self, name: str, prompt: Dict[str, str]):
        """Add a new prompt"""
        self.prompts[name] = prompt
        logger.info(f"Added prompt '{name}'")
    
    def list_prompts(self) -> List[str]:
        """List all available prompts"""
        return list(self.prompts.keys())
    
    def load_prompts_from_file(self, file_path: str):
        """Load prompts from a JSON file"""
        try:
            with open(file_path, 'r') as f:
                prompts = json.load(f)
            
            for name, prompt in prompts.items():
                self.prompts[name] = prompt
            
            logger.info(f"Loaded prompts from {file_path}")
        except Exception as e:
            logger.error(f"Failed to load prompts from {file_path}: {e}")
    
    def save_prompts_to_file(self, file_path: str):
        """Save prompts to a JSON file"""
        try:
            with open(file_path, 'w') as f:
                json.dump(self.prompts, f, indent=2)
            
            logger.info(f"Saved prompts to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save prompts to {file_path}: {e}")
    
    def get_prompt_stats(self) -> Dict[str, Any]:
        """Get prompt statistics"""
        return {
            "total_prompts": len(self.prompts),
            "prompt_names": list(self.prompts.keys())
        }
