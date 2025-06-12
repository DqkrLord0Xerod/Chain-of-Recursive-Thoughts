"""Stub parallel thinking optimizers for tests."""


class ParallelThinkingOptimizer:
    def __init__(self, *args, **kwargs):
        pass

    async def think_parallel(self, prompt, initial_response):
        return initial_response, [], {"rounds": 0, "convergence_reason": "stub"}


class AdaptiveThinkingOptimizer:
    def __init__(self, parallel_optimizer):
        self.parallel_optimizer = parallel_optimizer
        self.parameter_performance = {}

    async def think_adaptive(self, prompt, initial, category):
        return initial, [], {"rounds": 0, "convergence_reason": "stub"}
