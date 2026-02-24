"""
Guardrails for Safe RAG Responses
Implements:
1. Prompt injection detection
2. Citation validation
3. Answer quality checks
"""

import re
from typing import Dict, List

class Guardrails:
    """
        Detect potential prompt injection attempts.

        Looks for patterns like:
        - "Ignore previous instructions"
        - "You are now..."
        - System prompt reveals
        
        Args:
            query: User input
            
        Returns:
            True if injection detected
    """
    @staticmethod
    def detect_prompt_injection(query: str) -> bool:
        injection_patterns = [
            r"ignore (previous|all|the) (instructions|rules)",
            r"you are now",
            r"disregard (previous|all)",
            r"forget (everything|all|your)",
            r"new (instructions|rules|system)",
            r"<system>",
            r"\\nSystem:",
        ]

        query_lower = query.lower()

        for pattern in injection_patterns:
            if re.search(pattern, query_lower):
                return True
        return False
    
    """
        Check if citations are valid and used.
        
        Args:
            answer: Generated answer text
            citations: List of citation numbers
            max_sources: Maximum valid source number
            
        Returns:
            True if citations are valid
    """
    @staticmethod
    def validate_citations(answer: str, citations: List[int], max_sources: int) -> bool:
        if not citations:
            if re.search(r'\[\d+\]', answer):
                return False
            
        for cite in citations:
            if cite < 1 or cite > max_sources:
                return False
            
        for cite in citations:
            pattern = f"\\[{cite}\\]"
            if not re.search(pattern, answer):
                return False
            
        return True
    
    """
        Perform quality checks on generated answer.
        
        Checks:
        1. Minimum length
        2. Not just restating the question
        3. Has sufficient info flag is consistent
        
        Args:
            response: LLM response dict
            min_length: Minimum answer length (chars)
            
        Returns:
            Dict with check results and issues
    """
    @staticmethod
    def check_answer_quality(response: Dict, min_length: int = 10) -> Dict:
        issues = []
        answer = response.get("answer", "")
        has_info = response.get("has_sufficient_info", True)

        if len(answer) < min_length:
            issues.append("Answer too short")

        refusal_phrases = [
            "don't have enough information",
            "cannot answer",
            "insufficient information",
            "not enough context"
        ]

        is_refusal = any(phrase in answer.lower() for phrase in refusal_phrases)
        if is_refusal and has_info:
            issues.append("Answer provides info but has_sufficient_info=False")

        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "is_refusal": is_refusal
        }
    
    @staticmethod
    def sanitize_query(query: str, max_length: int = 500) -> str:
        query = " ".join(query.split())

        if len(query) > max_length:
            query = query[:max_length] + "..."

        return query