# 🧠 Local AI Code Agent (AST-Aware RAG)

An intelligent codebase assistant that understands real-world backend systems using **AST parsing + RAG + tool-based reasoning**.

This project goes beyond basic code search by extracting **structured semantics** (functions, classes, REST endpoints) and enabling **grounded answers** using a hybrid retrieval + agent pipeline.

---

## 🚀 Features

### 🔍 AST-Aware Code Understanding

* Parses Python and Java using Tree-sitter
* Extracts:

  * Functions (Python)
  * Classes & Methods (Java)
  * REST Endpoints (`@GetMapping`, `@PostMapping`, etc.)
* Builds structured metadata for retrieval

---

### ⚡ Endpoint Intelligence (Spring Boot)

* Detects:

  * Controllers (`@RestController`)
  * Base paths (`@RequestMapping`)
  * HTTP methods (`GET`, `POST`, etc.)
* Constructs full endpoints:

  ```
  /groups/{groupId}/settlements
  ```

---

### 🧠 RAG + Agent Hybrid Architecture

* Retrieval-first (grounded answers)
* Multi-step agent with tool usage:

  * `search_documents`
  * `read_file`
  * `find_endpoint`
* Falls back to tools only when necessary

---

### 🎯 Smart Retrieval

* Query rewriting (bridges natural language ↔ code)
* Metadata-aware ranking:

  * Endpoint > Method > Class
* Keyword + semantic alignment

---

### 🛡️ Hallucination Control

* Forces answers from retrieved context
* Prevents LLM from guessing when data is missing

---

## 🏗️ Architecture

```
User Query
   ↓
Query Rewriting
   ↓
Embedding Search (Qdrant)
   ↓
AST-Enriched Context
   ↓
Agent (LLM + Tools)
   ↓
Final Answer
```

---

## 🛠️ Tech Stack

* Python 3.10+
* Tree-sitter (AST parsing)
* Qdrant (vector database)
* Local LLM (via Ollama)
* Custom RAG + Agent pipeline

---

## 📦 Setup

### 1. Clone repo

```bash
git clone <your-repo-url>
cd local-ai-agent
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start local model (Ollama)

```bash
ollama serve
```

---

## ▶️ Usage

### Index a codebase

```bash
python -m app.cli
load <path_to_codebase>
```

### Ask questions

```text
Where is authentication handled?
What are the GET endpoints for settlements?
Which controller handles group balances?
```

---

## 📌 Example Output

```
GET endpoints for settlements:

- GET /groups/{groupId}/settlements
  Method: getSettlements
  File: SettlementController.java

- GET /settlements/{groupId}
  Method: getSettlements
  File: SettlementQueryController.java
```

---

## 🔥 Key Innovations

* AST-based chunking (not naive text splitting)
* Endpoint-aware retrieval for backend systems
* Query rewriting for code-language alignment
* Hybrid RAG + tool-based agent loop

---

## 🚧 Future Improvements

* Call graph extraction (Controller → Service → Repository)
* Hybrid search (vector + keyword)
* Intent classification (auto tool routing)
* UI for interactive code exploration

---

## 👨‍💻 Author

Shubham Jain

---

## ⭐ Why this project stands out

Most RAG systems:

* Treat code as plain text ❌

This system:

* Understands code structure ✅
* Extracts semantics (endpoints, methods) ✅
* Produces grounded, explainable answers ✅
