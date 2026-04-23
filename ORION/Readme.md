# Regulatory AI Assistant (RAG + Action Layer)

## Overview

This project implements an **AI-powered regulatory intelligence system** that enables:

* Semantic search over regulatory documents
* Natural language Q&A with citations
* Long-term insight generation (audit/compliance use cases)
*  **post-LLM actions use MCP here** (Jira, email, etc.)

The system follows a **Retrieval-Augmented Generation (RAG)** architecture with an optional **Action Layer** and future support for **MCP-based integrations**.

---

## Architecture

```
User Query
    ↓
Query Processor
    ↓
Hybrid Retrieval (Keyword + Vector)
    ↓
Reranker
    ↓
Top-K Chunks
    ↓
LLM (Answer Generation)
    ↓
Answer + Citations
    ↓
Action Layer (Optional)
    ↓
Jira / Email / Workflow / DB
```

---

## Core Concepts

### 1. LLM (Large Language Model)

* Converts regulatory text into human-readable insights
* Used for summarization, explanation, and reasoning

**Risk:** Hallucination
**Mitigation:** Always use RAG (never standalone)

---

### 2. RAG (Retrieval-Augmented Generation)

* Combines search + LLM
* Ensures answers are grounded in actual documents

**Flow:**

1. Retrieve relevant chunks
2. Pass to LLM
3. Generate answer

**Pros:**

* Accurate
* Traceable
* Enterprise-safe

**Cons:**

* Dependent on retrieval quality
* Slightly higher latency

---

### 3. Chunking

* Splits large documents into smaller sections

**Why:**

* LLM token limits
* Better search granularity

**Risk:**

* Context loss

**Fix:**

* Overlapping chunks
* Section-aware splitting

---

### 4. Embeddings

* Converts text into vectors for semantic search

**Example:**
"vendor risk" ≈ "third-party risk"

**Pros:**

* Meaning-based search

**Cons:**

* Imperfect similarity

---

### 5. Vector Database

* Stores embeddings
* Enables similarity search

**Examples:**

* FAISS
* Pinecone
* Weaviate

---

### 6. Hybrid Search

* Combines:

  * Keyword search (BM25)
  * Semantic search (vectors)

**Why:**
Regulatory data has both:

* exact terms (AML, SYSC)
* conceptual meaning

---

### 7. Reranker

* Improves relevance of retrieved results

**Why:**
Initial retrieval is noisy

---

### 8. Prompting

* Controls LLM behavior

**Best Practice:**

* Force grounding
* Avoid hallucination
* Require citations

---

### 9. Insight Engine

* Generates long-term trends

**Correct approach:**

* Deterministic analytics → then LLM summary

**Example:**

* Topic frequency over time
* Regulatory focus shifts

---

### 10. Citations (Critical)

* Attach document + section references to answers

**Why:**

* Audit requirement
* Trust

---

## Action Layer (Post-LLM Actions)

The system can trigger actions after generating an answer:

### Supported Actions

* Create Jira ticket
* Send email
* Trigger workflow
* Write to database
* Notify Slack/Teams

---

### Recommended Pattern

```
LLM → Suggest Action → Validation Layer → Execute Tool
```

---

### Example Output

```json
{
  "answer": "Outsourcing risk requires stronger exit planning.",
  "recommended_action": "create_jira",
  "action_payload": {
    "project": "AUDIT",
    "summary": "Review outsourcing controls",
    "priority": "High"
  },
  "confidence": 0.85
}
```

---

### Risks & Mitigation

| Risk              | Mitigation               |
| ----------------- | ------------------------ |
| Wrong action      | Human approval           |
| Duplicate tickets | Deduplication logic      |
| Wrong recipient   | Role-based email control |
| Tool failure      | Retry + queue            |

---

## MCP (Model Context Protocol)

### What MCP is

A standard to connect AI systems with external tools and data sources.

---

### Where MCP helps

* Jira integration
* Email integration
* Internal systems
* Workflow automation

---

### Where MCP does NOT help

* Retrieval quality
* Chunking
* Embeddings
* Insight generation

---

### Recommendation

| Stage         | Approach               |
| ------------- | ---------------------- |
| Initial build | Direct API integration |
| Scale stage   | Introduce MCP          |

---

## Failure Scenarios & Fallbacks

### 1. Hallucination

* **Cause:** LLM guessing
* **Fix:** Strict RAG

---

### 2. Retrieval Failure

* **Cause:** Poor search
* **Fix:** Hybrid search + reranking

---

### 3. Context Loss

* **Cause:** Bad chunking
* **Fix:** Overlap chunks

---

### 4. OCR Issues

* **Cause:** Scanned PDFs
* **Fix:** Validate extraction

---

### 5. Cost Explosion

* **Cause:** Too many LLM calls
* **Fix:** Caching + batching

---

### 6. Latency

* **Cause:** Multiple components
* **Fix:** Async + caching

---

## Project Structure

```
regulatory_ai_assistant/
│
├── ingestion/
├── retrieval/
├── llm/
├── insights/
├── actions/
├── api/
├── models/
└── utils/
```

---

## MVP Build Plan

### Step 1

* Load 10 PDFs
* Extract text

### Step 2

* Chunk + embeddings

### Step 3

* Vector search

### Step 4

* Add LLM (RAG)

### Step 5

* Add citations

### Step 6

* Add action layer

---

## Tech Stack

* Python
* FastAPI
* Vector DB (FAISS / Pinecone / Weaviate)
* LLM (OpenAI / enterprise model)
* Optional: LangChain / LlamaIndex

---

## Embedding Strategy

Two embedding providers are supported and switched via the `EMBEDDING_PROVIDER` environment variable.

| Provider | When to use | Config |
| -------- | ----------- | ------ |
| OpenAI `text-embedding-3-small` | Default — fast, cheap, 1536-dim | `EMBEDDING_PROVIDER=openai` |
| OpenAI `text-embedding-3-large` | Higher accuracy, 3072-dim | `OPENAI_EMBEDDING_MODEL=text-embedding-3-large` |
| Qwen local (HuggingFace) | Air-gapped / zero API cost | `EMBEDDING_PROVIDER=qwen`, leave `QWEN_API_KEY` empty |
| Qwen DashScope API | Qwen quality without a local GPU | `EMBEDDING_PROVIDER=qwen` + `QWEN_API_KEY` + `QWEN_API_BASE` |

**Recommendation:** Start with OpenAI `text-embedding-3-small`. Switch to Qwen local once you want zero embedding API cost — the system switches automatically based on `EMBEDDING_PROVIDER`.

### Why Hybrid Search Matters

Regulatory text has two distinct retrieval needs:

* **Exact terms** (AML, SYSC, Basel III, PSD2) → BM25 keyword search
* **Conceptual meaning** ("outsourcing risk" ≈ "third-party risk") → vector search

Neither alone is sufficient. The system fuses both using **Reciprocal Rank Fusion (RRF)**, then applies a **cross-encoder reranker** for final ordering.

```
Query
  ├── Embed → FAISS vector search ──┐
  └── Tokenise → BM25 keyword ──────┤ RRF fusion → top-K → CrossEncoder rerank → GPT-4o
```

---

## Final Thought

This is not just an LLM project.

It is a:

> **Search + Reasoning + Audit + Workflow system**

Build it step-by-step.
Focus on **retrieval quality first**, then add intelligence and actions.

---

