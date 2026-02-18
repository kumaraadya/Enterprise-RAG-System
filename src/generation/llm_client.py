"""
LLM Client for Answer Generation
Supports:
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)
- Error handling & retries
"""
import os
import json
from typing import Dict, List, Optional
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

class LLMClient:
    def __init__(self, provider: str = "openai", model: str = "gpt-4o-mini"):
        self.provider = provider
        self.model = model

        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in enviroment")
            self.client = OpenAI(api_key=api_key)

        elif provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in enviroment")
            self.client = Anthropic(api_key=api_key)

        else:
            raise ValueError(f"Unknown provider: {provider}")

    """
        Generate response from LLM.        
        Args:
            messages: List of chat messages
            temperature: Sampling temperature (lower = more deterministic)
            max_tokens: Max response length            
        Returns:
            Response dict with 'content' and 'usage'
    """    
    def generate(self, messages: List[Dict], temperature: float = 0.1,
                 max_tokens: int = 1000) -> Dict:
        if self.provider == "openai":
            return self._generate_openai(messages, temperature, max_tokens)
        elif self.provider == "anthropic":
            return self.__generate_anthropic(messages, temperature, max_tokens)