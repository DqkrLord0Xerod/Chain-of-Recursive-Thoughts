"""Enhanced quality evaluation for responses."""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from core.interfaces import QualityEvaluator as IQualityEvaluator, EmbeddingProvider


class EnhancedQualityEvaluator(IQualityEvaluator):
    """
    Advanced quality evaluator with multiple metrics.
    
    Evaluates:
    - Relevance (semantic similarity to prompt)
    - Completeness (coverage of prompt requirements)
    - Clarity (readability and structure)
    - Accuracy (factual correctness indicators)
    - Coherence (logical flow)
    """
    
    def __init__(
        self,
        embedding_provider: Optional[EmbeddingProvider] = None,
        weights: Optional[Dict[str, float]] = None,
        thresholds: Optional[Dict[str, float]] = None,
    ):
        self.embedding_provider = embedding_provider
        self.weights = weights or {
            "relevance": 0.35,
            "completeness": 0.25,
            "clarity": 0.20,
            "coherence": 0.15,
            "accuracy": 0.05,
        }
        self.thresholds = thresholds or {
            "overall": 0.92,
            "relevance": 0.8,
            "completeness": 0.8,
            "clarity": 0.8,
            "coherence": 0.75,
            "accuracy": 0.7,
        }
        
    def score(self, response: str, prompt: str) -> float:
        """Return overall quality score between 0 and 1."""
        detailed = self.detailed_score(response, prompt)
        return detailed["overall"]
        
    def detailed_score(self, response: str, prompt: str) -> Dict[str, float]:
        """Return detailed quality metrics."""
        metrics = {}
        
        # Relevance - how well response addresses the prompt
        metrics["relevance"] = self._score_relevance(response, prompt)
        
        # Completeness - coverage of prompt requirements
        metrics["completeness"] = self._score_completeness(response, prompt)
        
        # Clarity - readability and structure
        metrics["clarity"] = self._score_clarity(response)
        
        # Coherence - logical flow and consistency
        metrics["coherence"] = self._score_coherence(response)
        
        # Accuracy - factual correctness indicators
        metrics["accuracy"] = self._score_accuracy(response)
        
        # Calculate weighted overall score
        overall = sum(
            metrics[key] * self.weights.get(key, 0)
            for key in metrics
        )
        
        metrics["overall"] = max(0.0, min(1.0, overall))
        
        return metrics
        
    async def _score_relevance(self, response: str, prompt: str) -> float:
        """Score semantic relevance using embeddings."""
        if not self.embedding_provider:
            # Fallback to keyword overlap
            return self._keyword_relevance(response, prompt)
            
        try:
            similarity = await self.embedding_provider.similarity(response, prompt)
            
            # Also check key terms from prompt appear in response
            keyword_score = self._keyword_relevance(response, prompt)
            
            # Combine semantic and keyword relevance
            return 0.7 * similarity + 0.3 * keyword_score
            
        except Exception:
            # Fallback to keyword-based
            return self._keyword_relevance(response, prompt)
            
    def _keyword_relevance(self, response: str, prompt: str) -> float:
        """Calculate keyword-based relevance."""
        # Extract key terms from prompt
        prompt_terms = set(self._extract_key_terms(prompt.lower()))
        response_terms = set(self._extract_key_terms(response.lower()))
        
        if not prompt_terms:
            return 0.5
            
        # Check how many prompt terms appear in response
        overlap = len(prompt_terms & response_terms)
        coverage = overlap / len(prompt_terms)
        
        return min(1.0, coverage * 1.5)  # Boost slightly
        
    def _score_completeness(self, response: str, prompt: str) -> float:
        """Score how completely the response addresses the prompt."""
        # Check for question words and if they're answered
        questions = self._extract_questions(prompt)
        
        if not questions:
            # Not a question-based prompt
            return self._score_general_completeness(response, prompt)
            
        answered = 0
        for question in questions:
            if self._is_question_answered(question, response):
                answered += 1
                
        return answered / len(questions) if questions else 0.5
        
    def _score_clarity(self, response: str) -> float:
        """Score clarity and readability."""
        if not response.strip():
            return 0.0
            
        # Factors for clarity
        scores = []
        
        # Sentence structure
        sentences = self._split_sentences(response)
        if sentences:
            # Average sentence length (optimal ~15-20 words)
            avg_length = sum(len(s.split()) for s in sentences) / len(sentences)
            length_score = 1.0 - abs(avg_length - 17.5) / 50.0
            scores.append(max(0, length_score))
            
        # Paragraph structure
        paragraphs = response.strip().split('\n\n')
        if len(paragraphs) > 1:
            scores.append(0.9)  # Good structure
        elif len(response) > 500:
            scores.append(0.4)  # Long without breaks
        else:
            scores.append(0.7)  # Short enough to not need breaks
            
        # Use of formatting (lists, headers)
        if any(marker in response for marker in ['•', '-', '*', '1.', '#']):
            scores.append(0.9)
        else:
            scores.append(0.6)
            
        # Vocabulary complexity (simple is better)
        complex_words = len([
            w for w in response.split()
            if len(w) > 12
        ])
        complexity_score = max(0, 1.0 - complex_words / 100)
        scores.append(complexity_score)
        
        return sum(scores) / len(scores) if scores else 0.5
        
    def _score_coherence(self, response: str) -> float:
        """Score logical flow and consistency."""
        sentences = self._split_sentences(response)
        
        if len(sentences) < 2:
            return 0.8  # Too short to evaluate flow
            
        # Check for logical connectors
        connectors = [
            'therefore', 'however', 'moreover', 'furthermore',
            'additionally', 'consequently', 'thus', 'hence',
            'first', 'second', 'finally', 'next', 'then',
        ]
        
        connector_count = sum(
            1 for word in response.lower().split()
            if word in connectors
        )
        
        connector_score = min(1.0, connector_count / (len(sentences) / 3))
        
        # Check for consistency in terminology
        # (Simplified - would use more sophisticated NLP in production)
        consistency_score = 0.8
        
        # Check for clear conclusion
        conclusion_indicators = [
            'in conclusion', 'to summarize', 'in summary',
            'therefore', 'thus', 'overall',
        ]
        
        has_conclusion = any(
            indicator in response.lower()
            for indicator in conclusion_indicators
        )
        
        conclusion_score = 0.9 if has_conclusion else 0.7
        
        return (connector_score + consistency_score + conclusion_score) / 3
        
    def _score_accuracy(self, response: str) -> float:
        """Score accuracy indicators (not fact-checking)."""
        # Look for hedging language that indicates uncertainty
        hedging_phrases = [
            'might be', 'could be', 'possibly', 'perhaps',
            'it seems', 'apparently', 'allegedly', 'supposedly',
        ]
        
        hedging_count = sum(
            1 for phrase in hedging_phrases
            if phrase in response.lower()
        )
        
        # Some hedging is good (shows appropriate uncertainty)
        # Too much suggests lack of confidence
        if hedging_count == 0:
            hedging_score = 0.8  # Too confident
        elif hedging_count <= 2:
            hedging_score = 1.0  # Appropriate
        else:
            hedging_score = max(0.4, 1.0 - hedging_count * 0.1)
            
        # Check for citation patterns (even if not real citations)
        citation_patterns = [
            r'according to',
            r'research shows',
            r'studies indicate',
            r'experts say',
        ]
        
        has_citations = any(
            re.search(pattern, response, re.IGNORECASE)
            for pattern in citation_patterns
        )
        
        citation_score = 0.9 if has_citations else 0.7
        
        return (hedging_score + citation_score) / 2
        
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text."""
        # Remove common words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
            'to', 'for', 'of', 'with', 'by', 'from', 'is', 'are',
            'was', 'were', 'been', 'be', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'what', 'when', 'where', 'who', 'why', 'how',
        }
        
        words = re.findall(r'\b\w+\b', text.lower())
        return [w for w in words if w not in stop_words and len(w) > 2]
        
    def _extract_questions(self, prompt: str) -> List[str]:
        """Extract questions from prompt."""
        # Split by question marks and common question patterns
        questions = []
        
        # Direct questions
        for part in prompt.split('?'):
            part = part.strip()
            if part:
                questions.append(part + '?')
                
        # Indirect questions
        indirect_patterns = [
            r'explain\s+\w+',
            r'describe\s+\w+',
            r'what\s+is',
            r'how\s+does',
            r'why\s+does',
        ]
        
        for pattern in indirect_patterns:
            matches = re.findall(pattern, prompt.lower())
            questions.extend(matches)
            
        return questions
        
    def _is_question_answered(self, question: str, response: str) -> bool:
        """Check if a question is answered in the response."""
        # Extract key terms from question
        question_terms = self._extract_key_terms(question)
        
        if not question_terms:
            return True  # Can't evaluate
            
        # Check if key terms appear in response
        response_lower = response.lower()
        found = sum(
            1 for term in question_terms
            if term in response_lower
        )
        
        return found >= len(question_terms) * 0.5
        
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitter
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
        
    def _score_general_completeness(self, response: str, prompt: str) -> float:
        """Score completeness for non-question prompts."""
        # Length relative to prompt
        prompt_length = len(prompt.split())
        response_length = len(response.split())
        
        # Response should be proportional to prompt complexity
        if prompt_length < 10:
            expected_ratio = 5.0
        elif prompt_length < 50:
            expected_ratio = 3.0
        else:
            expected_ratio = 2.0
            
        actual_ratio = response_length / max(prompt_length, 1)
        
        if actual_ratio < expected_ratio * 0.5:
            return 0.3  # Too short
        elif actual_ratio > expected_ratio * 3:
            return 0.7  # Too long
        else:
            return 0.9  # Good length


class SimpleQualityEvaluator(IQualityEvaluator):
    """Simple quality evaluator for testing and fallback."""

    def __init__(self, thresholds: Optional[Dict[str, float]] = None) -> None:
        self.thresholds = thresholds or {"overall": 0.5}

    def score(self, response: str, prompt: str) -> float:
        """Basic scoring based on length and keyword overlap."""
        if not response:
            return 0.0
            
        # Length score
        length = len(response.split())
        if length < 10:
            length_score = 0.3
        elif length < 50:
            length_score = 0.7
        elif length < 200:
            length_score = 0.9
        else:
            length_score = 0.8
            
        # Keyword overlap
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())
        
        if prompt_words:
            overlap = len(prompt_words & response_words) / len(prompt_words)
            overlap_score = min(1.0, overlap * 2)
        else:
            overlap_score = 0.5
            
        return (length_score + overlap_score) / 2
        
    def detailed_score(self, response: str, prompt: str) -> Dict[str, float]:
        """Return simple detailed scores."""
        overall = self.score(response, prompt)
        
        return {
            "relevance": overall,
            "completeness": overall,
            "clarity": overall,
            "coherence": overall,
            "accuracy": overall,
            "overall": overall,
        }
