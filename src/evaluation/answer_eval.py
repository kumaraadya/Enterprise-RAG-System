"""
Answer Quality Evaluation

Measures:
1. Faithfulness: Is answer grounded in context?
2. Citation correctness: Are citations valid?
3. Completeness: Does answer address the question?
"""

import re
from typing import Dict, List

class AnswerEvaluator:

    """Check if answer contains citation markers like [1], [2]."""
    @staticmethod
    def check_citation_presence(answer: str) -> bool:
        return bool(re.search(r'\[\d+\]', answer))
    
    """Count number of citation markers in answer."""
    @staticmethod
    def count_citations(answer: str) -> int:
        return len(re.findall(r'\[\d+\]', answer))
    
    """
        Check what % of expected keywords appear in answer.
        
        Args:
            answer: Generated answer
            expected_keywords: Keywords that should appear
            
        Returns:
            Coverage ratio [0, 1]
    """
    @staticmethod
    def check_keyword_coverage(answer: str, expected_keywords: List[str]) -> float:
        if not expected_keywords:
            return 1.0
        answer_lower = answer.lower()
        found = sum(1 for kw in expected_keywords if 
                    kw.lower() in answer_lower)
        
        return found / len(expected_keywords)
    
    @staticmethod
    def evaluate_answer(answer_dict: Dict, expected_keywords: List[str] = None) -> Dict[str, any]:
        answer = answer_dict.get('answer', '')
        citations = answer_dict.get('citations', [])
        has_info = answer_dict.get('has_sufficient_info', True)

        results = {
            'answer_length': len(answer),
            'has_citations': AnswerEvaluator.check_citation_presence(answer),
            'num_citations': AnswerEvaluator.count_citations(answer),
            'citation_count_matches': len(citations) == AnswerEvaluator.count_citations(answer),
            'has_sufficient_info': has_info
        }

        if expected_keywords:
            results['keyword_coverage'] = AnswerEvaluator.check_keyword_coverage(answer, expected_keywords)

        return results