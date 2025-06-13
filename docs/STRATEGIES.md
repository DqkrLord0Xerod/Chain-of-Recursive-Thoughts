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

config = CoRTConfig(thinking_strategy="fixed")
engine = create_default_engine(config)
```

Unknown strategy names fall back to the adaptive implementation.

## Runtime configuration

The active strategy can also be set via the `THINKING_STRATEGY` environment
variable or the equivalent setting in a configuration file. The factory will use
this value when no explicit strategy name is provided.
