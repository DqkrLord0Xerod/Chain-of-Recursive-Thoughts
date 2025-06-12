# Thinking Strategies

The project supports multiple strategies for controlling the number of recursive
thinking rounds. Strategies live in `core.strategies` and can be selected at
runtime.

## Available Strategies

- **adaptive** – Uses the LLM to decide how many rounds are required and stops
  early when quality improves little.
- **fixed** – Runs a fixed number of rounds regardless of quality.

## Selecting a Strategy

`CoRTConfig` has a `thinking_strategy` field which defaults to `"adaptive"`.
When creating an engine using `create_default_engine`, the provided value is
resolved by `core.strategies.load_strategy`.

```python
from core import CoRTConfig, create_default_engine

config = CoRTConfig(thinking_strategy="fixed")
engine = create_default_engine(config)
```

Unknown strategy names fall back to the adaptive implementation.
