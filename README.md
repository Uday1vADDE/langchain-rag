# LangChain RAG with Memory 🧠

Most RAG chatbots have a frustrating problem — they forget everything the moment you ask a follow-up question.

Ask "Who created MCP?" and get a great answer. Then ask "What else did they build?" and suddenly the bot has no idea who "they" refers to.

This project fixes that.

---

## The problem it solves

Standard RAG pipeline:
```
User question → search PDF → answer → forget everything
```

This pipeline:
```
User question → rewrite using history → search PDF → answer → remember
```

The key addition is **query rewriting** — before searching the PDF, the question gets rewritten using conversation history to make it standalone and clear.

Example:
```
You:  "Who created MCP?"
Bot:  "Anthropic created MCP in 2024"

You:  "What else did they build?"
      ↓ query rewriter kicks in
      "What other technologies has Anthropic built besides MCP?"
      ↓ now searches PDF correctly
Bot:  "Anthropic also built Claude..."
```

Without query rewriting, searching for "what else did they build" would return random chunks — the PDF has no word "they".

---

## How it works

**Three layers working together:**

**Layer 1 — Hybrid Search**

Instead of pure vector similarity search, this combines two approaches:
- BM25 keyword search — great for exact terms like "MCP", "stdio", "JSON-RPC"
- Vector semantic search — great for meaning-based queries

Both run in parallel, results get merged and deduplicated. This catches things that pure vector search misses.

**Layer 2 — CrossEncoder Reranking**

After hybrid search returns 14-16 candidate chunks, a CrossEncoder model reads each chunk alongside the query and scores actual relevance — not just vector similarity. Top 3 most relevant chunks go to the LLM.

The difference matters:
- Vector similarity: are these mathematically close in embedding space?
- CrossEncoder relevance: does this chunk actually answer the question?

**Layer 3 — Conversation Memory + Query Rewriting**

Chat history gets stored in a simple list. Before every search, the current question gets rewritten using that history to resolve pronouns and references. The full history also gets passed to the LLM so it can give contextually aware answers.

---

## Pipeline in full

```
User types question
        ↓
Query rewriter uses chat history to make question standalone
        ↓
Hybrid search (BM25 + Vector) retrieves 14-16 candidates
        ↓
CrossEncoder reranker scores and picks top 3 chunks
        ↓
LLM receives: system prompt + chat history + top 3 chunks + question
        ↓
Answer gets saved to chat history for next question
```

---

## Tech stack

| Tool | Role |
|---|---|
| LangChain | Document loading, text splitting, vector retrieval |
| ChromaDB | Stores chunk embeddings locally |
| Google Gemini Embeddings | Converts text chunks to vectors |
| BM25Retriever | Keyword-based retrieval |
| CrossEncoder (ms-marco-MiniLM-L-6-v2) | Reranks candidates by relevance |
| Groq + LLaMA 3.3 70B | Fast LLM for answers and query rewriting |

---

## Getting started

**1. Clone the repo**
```bash
git clone https://github.com/Uday1vADDE/langchain-rag
cd langchain-rag
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add your API keys**

Create a `.env` file:
```
GROQ_API_KEY=your_groq_key_here
GEMINI_API_KEY=your_gemini_key_here
```

Free keys:
- Groq: [console.groq.com](https://console.groq.com)
- Gemini: [aistudio.google.com](https://aistudio.google.com)

**4. Add your PDF**

Place any PDF in the project folder and update the filename in `app.py`:
```python
loader = PyPDFLoader("your_document.pdf")
```

**5. Run**
```bash
python app.py
```

---

## Example conversation

```
RAG Chatbot with Memory ready!
Type 'quit' to exit

You: What is MCP?
Bot: MCP stands for Model Context Protocol. It is an open standard
     developed by Anthropic that defines how AI assistants communicate
     with external tools and data sources...

You: Who created it?
Bot: Anthropic created MCP, releasing it as an open standard in
     November 2024...

You: What transport mechanisms does it support?
Bot: MCP supports two transport mechanisms: stdio for local servers
     and HTTP with SSE for remote cloud servers...

You: Tell me more about the second one
     → Rewritten: "What are the details about HTTP with SSE transport in MCP?"
Bot: HTTP with SSE (Server-Sent Events) is used for remote servers
     hosted in the cloud. The client connects over HTTP and the server
     uses Server-Sent Events to push messages back...

You: quit
Goodbye!
```

---

## Project structure

```
langchain-rag/
├── app.py              # main RAG pipeline with memory
├── requirements.txt    # dependencies
├── .env                # API keys (not committed)
└── mcp_guide.pdf       # sample PDF to test with
```

---

## Limitations

- Runs in terminal — no web UI
- Memory resets when you restart the script
- CrossEncoder model downloads on first run (~80MB)
- Gemini free tier has embedding limits (1000/day)
- Works best with text-heavy PDFs — not great for image-heavy documents

---

## What makes this different from basic RAG

Basic RAG answers individual questions well but breaks on any follow-up. This pipeline handles natural conversation the way people actually talk — with pronouns, references, and context that builds over time.

The query rewriting step is the key. It's a small addition that makes the difference between a demo and something actually usable.
