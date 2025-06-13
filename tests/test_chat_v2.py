"""Tests for the refactored chat engine."""

import json
from typing import Dict, List, Optional

import pytest
from unittest.mock import MagicMock

from core.chat_v2 import (
    RecursiveThinkingEngine,
    AdaptiveThinkingStrategy,
    ThinkingResult,
    ThinkingRound,
)
from core.providers.cache import InMemoryLRUCache
from core.context_manager import ContextManager
from core.recursion import ConvergenceStrategy


# Mock implementations for testing
class MockLLMResponse:
    def __init__(self, content: str, tokens: int = 100):
        self.content = content
        self.usage = {
            "prompt_tokens": tokens // 2,
            "completion_tokens": tokens // 2,
            "total_tokens": tokens,
        }
        self.model = "test-model"
        self.cached = False


class MockLLMProvider:
    def __init__(self, responses: Optional[List[str]] = None):
        self.responses = responses or ["Test response"]
        self.call_count = 0
        self.calls = []
        
    async def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.7,
        **kwargs
    ) -> MockLLMResponse:
        self.calls.append({
            "messages": messages,
            "temperature": temperature,
            "kwargs": kwargs,
        })
        
        response = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        
        return MockLLMResponse(response)


class MockQualityEvaluator:
    def __init__(self, scores: Optional[Dict[str, float]] = None, thresholds: Optional[Dict[str, float]] = None):
        self.scores = scores or {}
        self.default_score = 0.5
        self.thresholds = thresholds or {"overall": 0.9}
        
    def score(self, response: str, prompt: str) -> float:
        return self.scores.get(response, self.default_score)


class MockThinkingStrategy:
    def __init__(self, rounds: int = 3, should_continue_until: int = 3):
        self.rounds = rounds
        self.should_continue_until = should_continue_until
        
    async def determine_rounds(self, prompt: str, *, request_id: str) -> int:
        return self.rounds
        
    async def should_continue(
        self,
        rounds_completed: int,
        quality_scores: List[float],
        responses: List[str],
        *,
        request_id: str,
    ) -> tuple[bool, str]:
        if rounds_completed >= self.should_continue_until:
            return False, "test_complete"
        return True, "continue"


class TestRecursiveThinkingEngine:
    
    @pytest.fixture
    def mock_tokenizer(self):
        tokenizer = MagicMock()
        tokenizer.encode = lambda text: text.split()
        return tokenizer
        
    @pytest.fixture
    async def engine(self, mock_tokenizer):
        llm = MockLLMProvider([
            "Initial response",
            json.dumps({
                "alternatives": ["Alt 1", "Alt 2"],
                "selection": "1",
                "thinking": "Alt 1 is better",
            }),
            json.dumps({
                "alternatives": ["Alt 3", "Alt 4"],
                "selection": "current",
                "thinking": "Current is best",
            }),
        ])
        
        cache = InMemoryLRUCache(max_size=100)
        evaluator = MockQualityEvaluator({
            "Initial response": 0.6,
            "Alt 1": 0.8,
            "Alt 2": 0.7,
            "Alt 3": 0.85,
            "Alt 4": 0.82,
        })
        
        context_manager = ContextManager(
            max_tokens=1000,
            tokenizer=mock_tokenizer,
        )

        strategy = MockThinkingStrategy(rounds=2, should_continue_until=2)
        convergence = ConvergenceStrategy(
            lambda a, b: evaluator.score(a, b),
            evaluator.score,
            max_iterations=3,
        )

        engine = RecursiveThinkingEngine(
            llm=llm,
            cache=cache,
            evaluator=evaluator,
            context_manager=context_manager,
            thinking_strategy=strategy,
            convergence_strategy=convergence,
            model_selector=None,
        )
        
        return engine
        
    @pytest.mark.asyncio
    async def test_basic_thinking_process(self, engine):
        """Test basic recursive thinking flow."""
        result = await engine.think_and_respond(
            "Test prompt",
            alternatives_per_round=2,
        )
        
        assert isinstance(result, ThinkingResult)
        assert result.response == "Alt 1"  # Should select Alt 1 as best
        assert result.thinking_rounds == 2
        assert len(result.thinking_history) >= 3  # Initial + alternatives
        assert result.convergence_reason == "test_complete"
        assert result.total_tokens > 0
        assert result.processing_time > 0
        
    @pytest.mark.asyncio
    async def test_conversation_history(self, engine):
        """Test conversation history management."""
        # First interaction
        await engine.think_and_respond("Hello")
        
        history = engine.conversation.get()
        assert len(history) == 2  # User + assistant
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello"
        assert history[1]["role"] == "assistant"
        
        # Second interaction
        await engine.think_and_respond("How are you?")
        
        history = engine.conversation.get()
        assert len(history) == 4  # 2 interactions
        
    @pytest.mark.asyncio
    async def test_caching(self, engine):
        """Test response caching."""
        # First call
        result1 = await engine.think_and_respond("Test prompt")
        llm_calls_1 = engine.llm.call_count
        
        # Clear history to ensure same context
        engine.conversation.clear()
        
        # Second call with same prompt
        result2 = await engine.think_and_respond("Test prompt")
        llm_calls_2 = engine.llm.call_count
        
        # Should have cache hits, so fewer LLM calls
        assert llm_calls_2 < llm_calls_1 * 2
        assert result1.response == result2.response
        
    @pytest.mark.asyncio
    async def test_override_thinking_rounds(self, engine):
        """Test overriding thinking rounds."""
        result = await engine.think_and_respond(
            "Test prompt",
            thinking_rounds=1,  # Override strategy
        )
        
        assert result.thinking_rounds == 1
        
    @pytest.mark.asyncio
    async def test_thinking_history_structure(self, engine):
        """Test thinking history contains expected data."""
        result = await engine.think_and_respond("Test prompt")
        
        for round in result.thinking_history:
            assert isinstance(round, ThinkingRound)
            assert hasattr(round, "round_number")
            assert hasattr(round, "response")
            assert hasattr(round, "quality_score")
            assert hasattr(round, "duration")
            assert round.duration >= 0
            assert 0 <= round.quality_score <= 1
            
    @pytest.mark.asyncio
    async def test_metadata_passthrough(self, engine):
        """Test metadata is preserved through the process."""
        metadata = {"request_id": "test123", "user_id": "user456"}
        
        result = await engine.think_and_respond(
            "Test prompt",
            metadata=metadata,
        )
        
        assert "request_id" in result.metadata
        assert result.metadata["request_id"] == "test123"
        assert "quality_progression" in result.metadata
        assert "final_quality" in result.metadata


class TestAdaptiveThinkingStrategy:
    
    @pytest.fixture
    def strategy(self):
        llm = MockLLMProvider(["3"])  # Will return "3" for rounds determination
        evaluator = MockQualityEvaluator(thresholds={"overall": 0.95})
        return AdaptiveThinkingStrategy(
            llm=llm,
            evaluator=evaluator,
            min_rounds=1,
            max_rounds=5,
            improvement_threshold=0.01,
        )
        
    @pytest.mark.asyncio
    async def test_determine_rounds(self, strategy):
        """Test adaptive round determination."""
        rounds = await strategy.determine_rounds(
            "Complex prompt",
            request_id="test",
        )
        assert 1 <= rounds <= 5
        assert rounds == 3  # Based on mock response
        
    @pytest.mark.asyncio
    async def test_should_continue_quality_met(self, strategy):
        """Test stopping when quality threshold is met."""
        should_continue, reason = await strategy.should_continue(
            rounds_completed=2,
            quality_scores=[0.5, 0.8, 0.96],
            responses=["r1", "r2", "r3"],
        )
        
        assert not should_continue
        assert reason == "quality_threshold_met"
        
    @pytest.mark.asyncio
    async def test_should_continue_plateau(self, strategy):
        """Test stopping when quality plateaus."""
        should_continue, reason = await strategy.should_continue(
            rounds_completed=3,
            quality_scores=[0.5, 0.8, 0.805, 0.808],
            responses=["r1", "r2", "r3", "r4"],
        )
        
        assert not should_continue
        assert reason == "quality_plateau"
        
    @pytest.mark.asyncio
    async def test_should_continue_oscillation(self, strategy):
        """Test stopping when responses oscillate."""
        should_continue, reason = await strategy.should_continue(
            rounds_completed=3,
            quality_scores=[0.5, 0.6, 0.7, 0.6],
            responses=["response1", "response2", "response3", "response1"],
        )
        
        assert not should_continue
        assert reason == "oscillation_detected"
        
    @pytest.mark.asyncio
    async def test_should_continue_max_rounds(self, strategy):
        """Test stopping at max rounds."""
        should_continue, reason = await strategy.should_continue(
            rounds_completed=5,
            quality_scores=[0.5, 0.6, 0.7, 0.8, 0.85],
            responses=["r1", "r2", "r3", "r4", "r5"],
        )
        
        assert not should_continue
        assert reason == "max_rounds_reached"


class TestIntegration:
    """Integration tests with real components."""
    
    @pytest.mark.asyncio
    async def test_full_thinking_cycle(self):
        """Test complete thinking cycle with real components."""
        
        # Setup with specific responses
        llm = MockLLMProvider([
            "2",  # Rounds determination
            "Initial thoughts on the topic",
            json.dumps({
                "alternatives": [
                    "A more detailed analysis of the topic",
                    "A creative approach to the topic",
                ],
                "evaluation": {
                    "current": {"score": 6, "strengths": "Clear", "weaknesses": "Basic"},
                    "1": {"score": 8, "strengths": "Detailed", "weaknesses": "Long"},
                    "2": {"score": 7, "strengths": "Creative", "weaknesses": "Unclear"},
                },
                "selection": "1",
                "thinking": "The detailed analysis provides more value",
            }),
        ])
        
        cache = InMemoryLRUCache(max_size=10)
        
        evaluator = MockQualityEvaluator({
            "Initial thoughts on the topic": 0.6,
            "A more detailed analysis of the topic": 0.85,
            "A creative approach to the topic": 0.75,
        })
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode = lambda text: text.split()
        
        context_manager = ContextManager(
            max_tokens=1000,
            tokenizer=mock_tokenizer,
        )
        
        strategy = AdaptiveThinkingStrategy(
            llm=llm,
            evaluator=evaluator,
            improvement_threshold=0.05,
        )
        convergence = ConvergenceStrategy(
            lambda a, b: evaluator.score(a, b),
            evaluator.score,
            max_iterations=3,
        )

        engine = RecursiveThinkingEngine(
            llm=llm,
            cache=cache,
            evaluator=evaluator,
            context_manager=context_manager,
            thinking_strategy=strategy,
            convergence_strategy=convergence,
            model_selector=None,
        )
        
        # Execute thinking
        result = await engine.think_and_respond(
            "Explain quantum computing",
            alternatives_per_round=2,
            temperature=0.7,
            metadata={"test": True},
        )
        
        # Verify results
        assert result.response == "A more detailed analysis of the topic"
        assert result.thinking_rounds >= 1
        assert result.convergence_reason in ["quality_plateau", "quality_threshold_met"]
        
        # Check thinking history
        selected_rounds = [r for r in result.thinking_history if r.selected]
        assert len(selected_rounds) >= 1
        
        # Verify metadata
        assert result.metadata["test"] is True
        assert "quality_progression" in result.metadata
        assert result.metadata["final_quality"] >= 0.8
        
        # Check cache stats
        cache_stats = await cache.stats()
        assert cache_stats["hits"] > 0  # Should have some cache hits
        
    @pytest.mark.asyncio
    async def test_save_and_load_conversation(self, tmp_path):
        """Test saving and loading conversations."""
        
        llm = MockLLMProvider(["1", "Response 1"])
        cache = InMemoryLRUCache()
        evaluator = MockQualityEvaluator()
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode = lambda text: text.split()
        
        context_manager = ContextManager(100, mock_tokenizer)
        strategy = MockThinkingStrategy(rounds=1)
        convergence = ConvergenceStrategy(
            lambda a, b: evaluator.score(a, b),
            evaluator.score,
            max_iterations=2,
        )

        engine = RecursiveThinkingEngine(
            llm=llm,
            cache=cache,
            evaluator=evaluator,
            context_manager=context_manager,
            thinking_strategy=strategy,
            convergence_strategy=convergence,
            model_selector=None,
        )
        
        # Have a conversation
        await engine.think_and_respond("Hello")
        await engine.think_and_respond("How are you?")
        
        # Save conversation
        save_path = tmp_path / "conversation.json"
        await engine.conversation.save(str(save_path))
        
        # Create new engine and load
        new_engine = RecursiveThinkingEngine(
            llm=llm,
            cache=cache,
            evaluator=evaluator,
            context_manager=context_manager,
            thinking_strategy=strategy,
            convergence_strategy=convergence,
            model_selector=None,
        )
        
        await new_engine.conversation.load(str(save_path))
        
        # Verify history is loaded
        history = new_engine.conversation.get()
        assert len(history) == 4  # 2 exchanges
        assert history[0]["content"] == "Hello"
        assert history[2]["content"] == "How are you?"
