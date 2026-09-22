# DocuLens AI: Agentic Self-Corrective RAG

An enterprise-ready Agentic RAG system powered by **LangGraph**, **Google Gemini 2.5 Flash**, and **ChromaDB**. Unlike naive RAG pipelines, DocuLens AI evaluates document relevance via Pydantic schemas, dynamically rewrites ambiguous queries, verifies post-generation factual grounding, and incorporates a stateful Human-In-The-Loop (HITL) interrupt gate.

---

## 🌟 Key Architecture & Capabilities

- **Stateful Cyclic Agent:** Implemented via LangGraph `StateGraph` with graph checkpointing (`MemorySaver`).
- **Pydantic Evaluator Nodes:**
  - *Document Relevance Grader:* Prunes noisy or off-topic retrieved chunks before generation.
  - *History-Aware Query Rewriter:* Resolves pronoun ambiguities and extracts core semantic keywords using conversation history.
  - *Hallucination Guardrail:* Performs strict factual grounding checks against retrieved context prior to response delivery.
- **Human-In-The-Loop (HITL):** Uses native LangGraph `interrupt()` and `Command(resume=...)` to pause execution and solicit user steering when automated retrieval limits are reached.
- **Intent-Based Routing:** Bypasses vector retrieval for conversational metadata questions to optimize latency and prevent false-negative retrieval crashes.
- **Hybrid Document Indexing:** Seamlessly switches between pre-indexed enterprise policies and ad-hoc PDF uploads using a Windows-safe Chroma collection management workflow.

---

## 🛠️ Tech Stack

- **Orchestration:** LangGraph, LangChain Core
- **LLM & Embeddings:** Google Gemini 2.5 Flash, `text-embedding-004`
- **Vector Database:** ChromaDB (`langchain-chroma`)
- **Document Parsing:** PyPDF, LangChain Text Splitters
- **Data Validation:** Pydantic v2
- **Interface:** Streamlit

---

## 🚀 Getting Started

### 1. Clone & Set Up Environment
```bash
git clone [https://github.com/](https://github.com/)<YOUR_USERNAME>/DocuLens-AI.git
cd DocuLens-AI
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate
pip install -r requirements.txt