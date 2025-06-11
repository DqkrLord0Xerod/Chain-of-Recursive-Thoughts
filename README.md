# 🧠 Multi-stage Intelligence Looping System (MILS)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)

> MILS is a modular engine for recursively evaluating, refining, and selecting reasoning outputs across iterative stages. Originally forked from CoTR, MILS refactors the system into a modular, executable reasoning system with multi-stage intelligence loops at its core.

---

## 🚀 Features

* 🔄 Recursive thinking loop with multi-pass self-correction
* 🎯 Adaptive round depth based on prompt complexity
* ⚡ Parallel alternatives to explore diverse reasoning paths
* 🔌 Modular architecture with pluggable LLM, cache, and strategy layers
* 📊 Observability-first: built-in metrics, tracing, and resilience support

---

## 📚 Table of Contents

* [Why CoRT?](#why-cort)
* [Quick Start](#quick-start)
* [🏗️ Architecture Overview](#️architecture-overview)
* [Basic API Usage](#basic-api-usage)
* [Extending CoRT](#extending-cort)
* [Monitoring & Testing](#monitoring--testing)
* [Roadmap & Limitations](#roadmap--limitations)
* [Contributing](#contributing)
* [Acknowledgments](#acknowledgments)
* [License](#license)

---

## ❓ Why CoRT?

Typical LLM workflows operate on a prompt→response basis. CoRT instead follows a recursive reasoning loop:

```mermaid
flowchart LR
  P[Prompt] --> A[Initial Response]
  A --> B[Generate Alternatives]
  B --> C[Evaluate Quality]
  C --> D[Select Best]
  D --> E{Quality OK?}
  E -- No --> B
  E -- Yes --> F[Final Response]
```

This enables:

* 🧠 Dynamic refinement through self-critique
* ✅ Error reduction and quality convergence
* 🧩 Improved performance on complex queries

---

## ⚡ Quick Start

1. Clone & install
   `git clone https://github.com/DqkrLord0Xerod/Multistage-Intelligence-Looping-System.git`
   `cd Multistage-Intelligence-Looping-System`
   `pip install -r requirements.txt`

2. Configure
   `cp .env.example .env`
   (Set `OPENROUTER_API_KEY` in `.env`)

3. Run server
   `uvicorn recthink_web:app --reload`

For CLI mode, frontend setup, and advanced options, see `docs/USAGE.md`.

---

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

| Component               | Purpose                         | Location                             |
| ----------------------- | ------------------------------- | ------------------------------------ |
| RecursiveThinkingEngine | Manages thinking loop logic     | `core/chat_v2.py`                    |
| LLMProvider             | Unified interface for LLMs      | `core/providers/llm.py`              |
| CacheProvider           | Response caching infrastructure | `core/providers/cache.py`            |
| QualityEvaluator        | Scores and compares responses   | `core/providers/quality.py`          |
| CircuitBreaker          | Limits cascading failures       | `core/resilience/circuit_breaker.py` |
| MetricsAnalyzer         | Real-time performance metrics   | `monitoring/metrics_v2.py`           |

### Architectural Transition

⚠️ **Note**: CoRT is transitioning from a v1 monolith to a clean v2 architecture.

* **Current API**: `recthink_web.py` still runs on legacy v1 logic
* **v2 Core Engine**: `core/chat_v2.py` is modular and ready
* **Migration Guide**: See `claude/cort-migration-guide.txt`

---

## 📬 Basic API Usage

Example request:

`curl -X POST http://localhost:8000/api/send_message -H "Content-Type: application/json" -d '{"session_id":"sess1","message":"Hello, CoRT!","thinking_rounds":2}'`

Full API reference available in `docs/API_REFERENCE.md`.

---

## 🔌 Extending CoRT

Want to integrate a new LLM, cache, or thinking module? Start here:

* `docs/EXTENDING.md#custom-providers`
* `docs/EXTENDING.md#custom-strategies`

---

## 📈 Monitoring & Testing

* Metrics & tracing setup → `docs/MONITORING.md`
* Run tests → `pytest` or `pytest --cov`

---

## 📅 Roadmap & Limitations

Upcoming priorities:

* Full v2 production API migration
* Redis cache backend
* Native OpenAI & Anthropic support
* Quality benchmarking

See `docs/ROADMAP.md` for details.

---

## 🤝 Contributing

We welcome contributions! You can:

* Fork the repo and open a PR
* Suggest improvements or file issues
* Review `AGENTS.md` for contributor guidelines

---

## 🙏 Acknowledgments

This project builds on PhialsBasement’s original [Chain-of-Recursive-Thoughts](https://github.com/PhialsBasement/Chain-of-Recursive-Thoughts).

Enhancements in this fork:

* ✅ Dependency injection and async execution
* ✅ Modular, layered architecture
* ✅ Persistent agent orchestration and automation
* ✅ Platform-oriented focus beyond single scripts

Thanks to:

* Original idea: *“AI that argues with itself”* by PhialsBasement
* Visuals and recursive logic inspired by early prototypes

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
