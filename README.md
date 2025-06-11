# 🧠 CoRT - Chain of Recursive Thoughts

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)

**CoRT** is an advanced AI reasoning engine that dramatically improves response quality through recursive refinement. Instead of accepting the first LLM output, CoRT iteratively generates alternatives, evaluates them, and converges on optimal responses through intelligent feedback loops.

## 🎯 Why CoRT?

Traditional LLM interactions are one-shot: prompt → response. CoRT transforms this into a **thinking process**:

```
Prompt → Initial Response → Generate Alternatives → Evaluate Quality → Select Best → Converge
         ↑                                                                              ↓
         └──────────────────── Repeat if quality insufficient ─────────────────────────┘
```

This recursive approach consistently produces higher quality outputs by:
- 🔄 **Self-correction**: Catches and fixes errors through multiple passes
- 🎯 **Convergence**: Stops when quality plateaus or thresholds are met
- 📊 **Adaptive depth**: Simple queries resolve quickly, complex ones get more rounds
- 🚀 **Parallel alternatives**: Explores multiple solution paths simultaneously

## 🏗️ Architecture Overview

```mermaid
graph TB
    subgraph "API Layer"
        A[FastAPI Server] --> B[WebSocket Handler]
        A --> C[REST Endpoints]
    end
    
    subgraph "Core Engine"
        D[RecursiveThinkingEngine] --> E[ThinkingStrategy]
        D --> F[ContextManager]
        D --> G[QualityEvaluator]
    end
    
    subgraph "Provider Layer"
        H[LLMProvider] --> I[OpenRouter]
        H --> J[Fallback Providers]
        K[CacheProvider] --> L[Memory/Disk/Redis]
    end
    
    subgraph "Resilience"
        M[CircuitBreaker] --> H
        N[RetryPolicy] --> H
        O[HedgingExecutor] --> H
    end
    
    subgraph "Observability"
        P[OpenTelemetry] --> Q[Prometheus]
        P --> R[Jaeger]
        S[MetricsAnalyzer] --> T[Grafana]
    end
    
    C --> D
    B --> D
    D --> H
    D --> K
```

### Key Components

| Component | Purpose | Location |
|-----------|---------|----------|
| **RecursiveThinkingEngine** | Orchestrates the thinking loop | `core/chat_v2.py` |
| **LLMProvider** | Abstraction for LLM backends | `core/providers/llm.py` |
| **CacheProvider** | Multi-tier response caching | `core/providers/cache.py` |
| **QualityEvaluator** | Scores response quality | `core/providers/quality.py` |
| **CircuitBreaker** | Fault isolation | `core/resilience/circuit_breaker.py` |
| **MetricsAnalyzer** | Real-time performance insights | `monitoring/metrics_v2.py` |

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Redis (optional, for distributed caching)
- OpenRouter API key

### Installation

```bash
# Clone the repository
git clone https://github.com/PhialsBasement/Chain-of-Recursive-Thoughts.git
cd Chain-of-Recursive-Thoughts

# Install dependencies
pip install -r requirements.txt

# For production features
pip install -r requirements-prod.txt

# Set up environment
cp .env.example .env
# Edit .env with your OPENROUTER_API_KEY
```

### Running the Application

#### 1. CLI Mode
```bash
export OPENROUTER_API_KEY="your-key-here"
python -m cli.main
```

#### 2. Web API Mode
```bash
# Start the backend
python recthink_web.py

# In another terminal, start the frontend
cd frontend
npm install
npm start
```

The API will be available at `http://localhost:8000` and the UI at `http://localhost:3000`.

## 📡 API Usage

### REST API

#### Initialize a Session
```bash
curl -X POST http://localhost:8000/api/initialize \
  -H "Content-Type: application/json" \
  -d '{"api_key": "your-key", "model": "gpt-4"}'
```

#### Send a Message
```bash
curl -X POST http://localhost:8000/api/send_message \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "session_123",
    "message": "Explain quantum computing",
    "thinking_rounds": 3
  }'
```

### WebSocket Streaming

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/session_123');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'chunk') {
    // Intermediate thinking round
    console.log('Thinking:', data.response);
  } else if (data.type === 'final') {
    // Final refined response
    console.log('Final:', data.response);
  }
};

ws.send(JSON.stringify({
  type: 'message',
  content: 'What is consciousness?'
}));
```

### Python SDK Usage

```python
from core.chat_v2 import CoRTConfig, create_default_engine

async def main():
    # Create engine with configuration
    config = CoRTConfig(
        api_key="your-key",
        model="gpt-4",
        max_context_tokens=2000
    )
    engine = create_default_engine(config)
    
    # Execute thinking process
    result = await engine.think_and_respond(
        "Write a haiku about recursion",
        thinking_rounds=3,
        alternatives_per_round=4
    )
    
    print(f"Final response: {result.response}")
    print(f"Quality improvement: {result.metadata['improvement']:.2%}")
    print(f"Thinking rounds: {result.thinking_rounds}")

# Run it
import asyncio
asyncio.run(main())
```

## 🔧 Extending CoRT

### Custom LLM Provider

```python
from core.interfaces import LLMProvider, LLMResponse

class MyCustomProvider(LLMProvider):
    async def chat(self, messages, *, temperature=0.7, **kwargs):
        # Your implementation here
        response = await my_llm_api_call(messages)
        return LLMResponse(
            content=response.text,
            usage={"total_tokens": response.tokens},
            model="my-model"
        )

# Register in engine
engine = RecursiveThinkingEngine(
    llm=MyCustomProvider(),
    cache=InMemoryLRUCache(),
    evaluator=SimpleQualityEvaluator(),
    # ... other components
)
```

### Custom Thinking Strategy

```python
from core.interfaces import ThinkingStrategy

class ConservativeStrategy(ThinkingStrategy):
    async def determine_rounds(self, prompt: str) -> int:
        # Always use exactly 2 rounds
        return 2
    
    async def should_continue(self, rounds_completed, quality_scores, responses):
        if rounds_completed >= 2:
            return False, "max_rounds"
        return True, "continue"

# Use custom strategy
engine = RecursiveThinkingEngine(
    thinking_strategy=ConservativeStrategy(),
    # ... other components
)
```

## 📊 Monitoring & Observability

### Prometheus Metrics

CoRT exposes metrics at `http://localhost:9090/metrics`:

- `cort_thinking_rounds`: Distribution of thinking rounds per request
- `cort_quality_improvement`: Quality score improvements
- `cort_token_usage`: Token consumption patterns
- `cort_cache_hit_rate`: Cache effectiveness
- `cort_circuit_breaker_state`: Provider health status

### Grafana Dashboard

Import the pre-built dashboard:

```bash
# Start Grafana
docker run -d -p 3000:3000 grafana/grafana

# Import dashboard
curl -X POST http://admin:admin@localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @monitoring/dashboards/cort_dashboard.json
```

### Distributed Tracing

```python
# Enable Jaeger tracing
from monitoring.telemetry import initialize_telemetry

initialize_telemetry(
    service_name="cort",
    enable_prometheus=True,
    jaeger_endpoint="localhost:6831"
)
```

## ⚠️ Current State & Limitations

### Architectural Transition

⚠️ **Important**: CoRT is currently in a transition between v1 (monolithic) and v2 (clean architecture). 

- **Production API** (`recthink_web.py`): Still uses v1 architecture
- **New Engine** (`core/chat_v2.py`): Ready but not yet integrated
- **Migration Path**: See `claude/cort-migration-guide.txt` for details

### Known Limitations

1. **No Redis Provider**: Distributed caching requires implementation
2. **Limited Provider Support**: Only OpenRouter currently implemented
3. **Frontend**: Basic React UI, needs enhancement
4. **Benchmarks**: Quality improvements not yet quantified

### Immediate Roadmap

- [ ] Complete v2 API migration
- [ ] Implement Redis cache provider
- [ ] Add direct OpenAI/Anthropic providers
- [ ] Quantify quality improvements with benchmarks
- [ ] Enhanced streaming UI

## 🧪 Testing

```bash
# Run unit tests
pytest

# Run with coverage
pytest --cov=core tests/

# Run integration tests
pytest tests/test_integration.py

# Linting
flake8
```

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Follow the code style (`flake8` and `black`)
4. Add tests for new functionality
5. Update documentation as needed
6. Submit a Pull Request

See `AGENTS.md` for detailed development guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by chain-of-thought prompting research
- Built with [FastAPI](https://fastapi.tiangolo.com/) and [OpenRouter](https://openrouter.ai/)
- Monitoring powered by [OpenTelemetry](https://opentelemetry.io/)

---

<p align="center">
  <i>🔄 Recursion: See Recursion 🔄</i>
</p>
