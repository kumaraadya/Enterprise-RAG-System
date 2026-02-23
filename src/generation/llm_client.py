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
        
    """OpenAI-specific generation."""
    def _generate_openai(self, messages, temperature, max_tokens) -> Dict:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content

            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                parsed = {
                    "answer": content,
                    "citations": [],
                    "confidence": "low",
                    "has_sufficient_info": True
                }
            return {
                "content": parsed,
                "usage":{
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {e}")
    
    """Anthropic-specific generation."""
    def _generate_anthropic(self, messages, temperature, max_tokens) -> Dict:
        try:
            system_msg = next((m['content'] for m in messages if m['role'] == 'system'), "")
            user_msgs = [m for m in messages if m['role'] != 'system']

            response = self.client.messages.create(
                model = self.model,
                system = system_msg,
                messages = user_msgs,
                temperature = temperature,
                max_tokens = max_tokens
            )
            content = response.content[0].text

            try:
                parsed = json.Loads(content)
            except json.JSONDecodeError:
                parsed = {
                    "answer": content,
                    "citations": [],
                    "confidence": "low",
                    "has_sufficient_info": True
                }
            return {
                "content": parsed,
                "usage": {
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                }
            }
        
        except Exception as e:
            raise RuntimeError(f"Anthropic API error: {e}")
    
    """
        Safely parse LLM JSON response.
    
        Handles:
        - Valid JSON
        - JSON wrapped in markdown code blocks
        - Plain text fallback
    """
    def parse_llm_json_response(response_text: str) -> Dict:
        text = response_text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {
                "answer": text,
                "citations": [],
                "confidence": "low",
                "has_sufficient_info": True
            }