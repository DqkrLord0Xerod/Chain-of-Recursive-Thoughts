# Thinking Strategies

The project supports multiple strategies for controlling the number of recursive
thinking rounds. Strategies live in `core.strategies` and can be selected at
runtime.

Strategy creation is handled by `StrategyFactory`, which receives an LLM
provider and quality evaluator when an engine is built.

## Available Strategies

- **adaptive** – Uses the LLM to decide how many rounds are required and stops
  early when quality improves little.
- **fixed** – Runs a fixed number of rounds regardless of quality.

## Selecting a Strategy

`CoRTConfig` has a `thinking_strategy` field which defaults to the value of the
`THINKING_STRATEGY` environment variable (falling back to `"adaptive"`). When
creating an engine using `create_default_engine`, the strategy is obtained from
`StrategyFactory`.

```python
from core import CoRTConfig, create_default_engine

config = CoRTConfig(
    thinking_strategy="fixed",
    quality_thresholds={"overall": 0.8},
)
engine = create_default_engine(config)
```

Unknown strategy names fall back to the adaptive implementation.

codex/document-components-and-update-readme
## Advanced configuration

`load_strategy` can be used directly when you need to pass custom parameters or
register your own implementation.

```python
from core.strategies import load_strategy
from core.chat_v2 import RecursiveThinkingEngine

strategy = load_strategy("fixed", llm, evaluator, rounds=2)
engine = RecursiveThinkingEngine(
    llm=llm,
    cache=cache,
    evaluator=evaluator,
    context_manager=context,
    thinking_strategy=strategy,
)
```

## Runtime configuration

The active strategy can also be set via the `THINKING_STRATEGY` environment
variable or the equivalent setting in a configuration file. The factory will use
this value when no explicit strategy name is provided.

