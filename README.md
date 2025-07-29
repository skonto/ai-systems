# AI‑Systems 🚀
Sample agentic AI projects & reference implementations — Retrieval-Augmented Generation (RAG), agents, evaluation pipelines, and more.

## 📑 Contents

- [Quickstart](#-quickstart)
- [Architecture](#-architecture)
- [Components](#-components)
- [Evaluation](#-evaluation)
- [Contributing](#-contributing)
- [License](#-license)

## 🔧 Quickstart

```
# Download QA data in txt format under some path eg. data/fantastic_charge
# Save in a file named <filename>.txt.

Disclaimer: The sample data in this repo are generated automatically by an LLM for a fictional product and meant to have no relation with an existing product. Any resemblance is purely coincidental.

# Install Ollama and pull the models
ollama pull llama3:8b-instruct-q4_0
olllama pull BGE-M3:latest

# start opik as shown in the next section

uv sync --dev

# Ingest txt Q&A data in Chroma DB
uv run src/data/rag_ingest.py ./data/products /tmp/ch_db

# Run the Chat Streamlit app
uv run python -m streamlit run ./src/chatbot/app.py

# or
source .venv/bin/activate
python -m streamlit run ./src/chatbot/app.py
```

### Running opik.

Follow the instructions in https://www.comet.com/docs/opik/self-host/local_deployment.
The chatbot app is configured to bypass opik url input and work with opik running locally.


### Interacting with the Q&A Assistant

Access the app at: http://localhost:8501

![ui](./ui.png)


### Building the images

Under deployment you can find several images on how to run the LLMs used in this project.
You can also use the vLLM setup for a production on premise setup.

To build the images:

```
docker build -t skonto/ollama:qa -f Dockerfile.ollama .

docker build --no-cache --progress=plain --secret "id=guard,src=$HOME/.guardrailsrc" . -t skonto/qa
```

To run the images:

```
docker run --gpus all -p8080:11434 skonto/ollama:qa

docker run -it --gpus all -e OLLAMA_HOST=localhost:8080 --net=host skonto/qa
```

## 🧠 Architecture

The system consists of:
- Document ingestion and embedding
- RAG pipeline using Ollama
- Chatbot (CLI/HTTP interface)
- Evaluation using RAGAS and custom metrics

The final goal is to have a customizable architecture on top of the following rough diagram of the RAG pipeline:

```text
RAG (Retrieval-Augmented Generation) Pipeline
==============================================

┌─────────────────┐    ┌──────────────────┐
│  User Metadata  │    │   User Query     │
│   (context,     │    │                  │
│   preferences,  │    │                  │
│   permissions)  │    │                  │
└─────────┬───────┘    └─────────┬────────┘
          │                      │
          │              ┌───────▼─────────┐
          │              │ Query Sanitizer │
          │              │ & Guardrails    │
          │              │ (input filter)  │
          │              └───────┬─────────┘
          │                      │
          │                      ▼
          │              ┌───────────────────────┐
          └──────────────┤      RETRIEVER        │
                         │                       │
                         │ ┌─────────────────────┤
                         │ │  Search Strategy:   │
                         │ │  • Hybrid Search    │
                         │ │  • BM25 (keyword)   │
                         │ │  • Semantic (vector)│
                         │ └─────────────────────┤
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │    Retrieved Docs     │
                         │   (initial results)   │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │     RE-RANKER         │
                         │  (relevance scoring   │
                         │   & result ordering)  │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │    Top-K Docs         │
                         │   (best matches)      │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │  Context Enhancement  │
                         │ (combine query +      │
                         │  retrieved docs +     │
                         │  metadata context)    │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │        LLM            │
                         │                       │
                         │ Config:               │
                         │ • Temperature: 0.7    │
                         │ • Max tokens: 2048    │
                         │ • System prompt       │
                         │ • Model: GPT-4        │
                         │                       │
                         │ Output Format:        │
                         │ ┌─────────┬─────────┐ │
                         │ │Streaming│  Chat   │ │
                         │ │ Format  │ Format  │ │
                         │ └─────────┴─────────┘ │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │  Sampling Techniques  │
                         │                       │
                         │ • Top-p (nucleus)     │
                         │ • Top-k filtering     │
                         │ • Temperature scaling │
                         │ • Repetition penalty  │
                         │ • Beam search         │
                         │ • Greedy decoding     │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │  Output Guardrails    │
                         │ (safety filter,       │
                         │  content validation,  │
                         │  bias detection)      │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │   Final Response      │
                         │   (to user)           │
                         └───────────────────────┘

Data Flow Legend:
─────────────────
│  = Processing step/component
▼  = Data flow direction
┌─ = Component boundary
┤  = Internal component section

```
and then move to an Agentic RAG pipeline as follows:

```text

Agentic RAG Pipeline with Memory, Routing, Tools & Planning
============================================================

┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  User Metadata  │    │   User Query     │    │   Conversation  │
│   (context,     │    │                  │    │     Memory      │
│   preferences,  │    │                  │    │ (short & long   │
│   permissions)  │    │                  │    │  term history)  │
└─────────┬───────┘    └─────────┬────────┘    └─────────┬───────┘
          │                      │                       │
          │              ┌───────▼────────┐              │
          │              │ Query Sanitizer│              │
          │              │ & Guardrails   │              │
          │              │ (input filter) │              │
          │              └───────┬────────┘              │
          │                      │                       │
          │                      ▼                       │
          │              ┌─────────────────────────────────┴──┐
          │              │           PLANNER                  │
          │              │                                    │
          │              │ • Task decomposition               │
          │              │ • Goal identification              │
          │              │ • Multi-step strategy              │
          │              │ • Resource allocation              │
          │              └─────────────┬──────────────────────┘
          │                            │
          │                            ▼
          │              ┌─────────────────────────────────────┐
          │              │          LLM ROUTER                 │
          │              │                                     │
          │    ┌─────────┼─── Route Decision: ─────────────────┤
          │    │         │                                     │
          │    │         │ ┌─────────┬─────────┬─────────────┐ │
          │    │         │ │Retrieval│  Tool   │   Direct    │ │
          │    │         │ │  Path   │ Calling │   Response  │ │
          │    │         │ └─────────┴─────────┴─────────────┘ │
          │    │         └─────┬───────┬───────────────┬───────┘
          │    │               │       │               │
          │    │    ┌──────────┘       │               │
          │    │    │                  │               │
          │    │    ▼                  ▼               ▼
          │    │ ┌─────────────────┐ ┌───────────────────────┐ │
          │    │ │   RETRIEVER     │ │    TOOL CALLING       │ │
          │    │ │                 │ │                       │ │
          └────┼─┤ Search Strategy:│ │ Available Tools:      │ │
               │ │ • Hybrid Search │ │ • Web search          │ │
               │ │ • BM25 (keyword)│ │ • Calculator          │ │
               │ │ • Semantic      │ │ • Code interpreter    │ │
               │ │ • Memory-guided │ │ • Database queries    │ │
               │ └─────────┬───────┘ │ • API calls           │ │
               │           │         │ • File operations     │ │
               │           ▼         └───────────┬───────────┘ │
               │ ┌─────────────────┐             │             │
               │ │ Retrieved Docs  │             │             │
               │ │ (initial results│             │             │
               │ │  + memory refs) │             │             │
               │ └─────────┬───────┘             │             │
               │           │                     │             │
               │           ▼                     │             │
               │ ┌─────────────────┐             │             │
               │ │   RE-RANKER     │             │             │
               │ │(relevance +     │             │             │
               │ │ memory context) │             │             │
               │ └─────────┬───────┘             │             │
               │           │                     │             │
               │           ▼                     │             │
               │ ┌─────────────────┐             │             │
               │ │   Top-K Docs    │             │             │
               │ │ (best matches)  │             │             │
               │ └─────────┬───────┘             │             │
               │           │                     │             │
               │           └─────────────────────┼─────────────┘
               │                                 │
               │                                 ▼
               │                   ┌─────────────────────────┐
               │                   │  Context Enhancement    │
               │                   │                         │
               │                   │ Combine:                │
               │                   │ • Query + retrieved docs│
               │                   │ • Conversation memory   │
               │                   │ • Tool results          │
               │                   │ • User metadata         │
               │                   │ • Planning context      │
               │                   └─────────────┬───────────┘
               │                                 │
               │                                 ▼
               │                   ┌─────────────────────────┐
               │                   │        LLM              │
               │                   │                         │
               │                   │ Config:                 │
               │                   │ • Temperature: 0.7      │
               │                   │ • Max tokens: 4096      │
               │                   │ • System prompt         │
               │                   │ • Model: GPT-4          │
               │                   │ • Function calling      │
               │                   │                         │
               │                   │ Output Format:          │
               │                   │ ┌─────────┬───────────┐ │
               │                   │ │Streaming│   Chat    │ │
               │                   │ │ Format  │  Format   │ │
               │                   │ └─────────┴───────────┘ │
               │                   └─────────────┬───────────┘
               │                                 │
               │                                 ▼
               │                   ┌─────────────────────────┐
               │                   │  Sampling Techniques    │
               │                   │                         │
               │                   │ • Top-p (nucleus)       │
               │                   │ • Top-k filtering       │
               │                   │ • Temperature scaling   │
               │                   │ • Repetition penalty    │
               │                   │ • Beam search           │
               │                   │ • Greedy decoding       │
               │                   └─────────────┬───────────┘
               │                                 │
               │                                 ▼
               │                   ┌─────────────────────────┐
               │                   │  Output Guardrails      │
               │                   │                         │
               │                   │ Streaming Mode:         │
               │                   │ • Token-level filter    │
               │                   │ • Sliding window        │
               │                   │ • Circuit breaker       │
               │                   │ • Real-time scoring     │
               │                   │                         │
               │                   │ Batch Mode:             │
               │                   │ • Full content scan     │
               │                   │ • Complete validation   │
               │                   │ • Comprehensive bias    │
               │                   │   detection             │
               │                   └─────────────┬───────────┘
               │                                 │
               │                                 ▼
               │              ┌─────────────────────────────────┐
               │              │      MEMORY UPDATE              │
               │              │                                 │
               │              │ • Store conversation turn       │
               │              │ • Update user preferences       │
               │              │ • Cache retrieved documents     │
               │              │ • Log tool usage patterns       │
               │              │ • Update planning context       │
               │              └─────────────┬───────────────────┘
               │                            │
               │              ┌─────────────▼───────────────────┐
               │              │      Final Response             │
               │              │                                 │
               │              │ ┌─────────┬─────────────────────┤
               │              │ │Streaming│ Batch Response      │
               │              │ │Delivery │ (with memory        │
               │              │ │         │  updates complete)  │
               │              │ └─────────┴─────────────────────┤
               │              └─────────────────────────────────┘
               │                            │
               └────────────────────────────┘
                      
Data Flow Legend:
─────────────────
│  = Processing step/component  
▼  = Data flow direction
┌─ = Component boundary
┤  = Internal component section
└─ = Feedback/memory loop
```

## 📦 Components

- `src/data/`: Document ingestion & indexing
- `src/rag/`: RAG pipelines using Ollama
- `src/chatbot/`: CLI/HTTP chatbot interfaces
- `src/test/`: Unit + integration tests and benchmarks
- `Makefile`, `uv`: commands for lint, test, format, eval

## 📈 Evaluation

Evaluate generated answers using [RAGAS](https://github.com/explodinggradients/ragas):

```bash
uv run pytest -m integration
```

Metrics include:
- `LLMContextRecall`
- `FactualCorrectness`
- `BleuScore`
- `ResponseRelevancy`

## 🤝 Contributing

Contributions welcome!

1. Fork and clone
2. Run `uv sync --dev`
3. Lint: `make lint`
4. Type-check: `make type-check`
5. Test: `pytest`

## 📄 License

This repository is licensed under the Apache 2 License — see [LICENSE](LICENSE) for full details.
