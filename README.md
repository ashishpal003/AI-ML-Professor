# 📚 AI-ML Professor

A **production-ready Retrieval-Augmented Generation (RAG) system** that helps users learn Machine Learning and AI concepts using local PDF documents.

Built with **FastAPI, LangChain, FAISS, Ollama, Streamlit**, and enhanced with:

* Caching
* Evaluation
* Observability
* Query transformation
* Retrieval confidence scoring

---

# 🚀 Features

## 🔍 Core RAG Capabilities

* Semantic search over local PDFs
* FAISS-based vector store
* Cross-encoder reranking
* Multi-query retrieval
* Query rewriting

## ⚡ Performance Optimizations

* Semantic caching (FAISS-based)
* Retrieval caching (Redis)
* Async streaming responses
* Parallel retrieval execution

## 🧠 Intelligence Layer

* Query rewriting for better retrieval
* Multi-query expansion
* Retrieval confidence scoring
* Guardrails for low-confidence responses

## 📊 Evaluation & Observability

* DeepEval integration:

  * Faithfulness
  * Answer relevancy
* MLflow tracking:

  * Latency
  * Retrieval metrics
  * Chunk statistics

## 💬 Conversational Features

* Session-based memory
* Chat history tracking
* Context-aware responses

## 📄 Document Handling

* Upload PDFs dynamically
* Incremental vector store updates
* Automatic chunking & embedding

## 🖥️ UI

* Streamlit chat interface
* File upload support
* Streaming responses

---

# 📁 Project Structure

```
AI-ML-Professor/

data/
    machine_learning_book.pdf

rag_src/
    caching/
    config/
    evaluation/
    generation/
    ingestion/
    observability/
    query_transform/
    retrieval/
    utils/

app.py
streamlit_app.py
requirements.txt
```

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone <your-repo-url>
cd AI-ML-Professor
```

## 2️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🧠 Setup Requirements

## 🔹 Install Ollama

Download and install Ollama, then run:

```bash
ollama pull llama3
```

---

## 🔹 Start Redis (for caching)

Using Docker:

```bash
docker run -d -p 6379:6379 redis
```

---

# 📦 Run Ingestion Pipeline (IMPORTANT)

Before querying, build the vector database:

```bash
python run_ingestion.py
```

This will:

* Load PDFs from `data/`
* Split into chunks
* Generate embeddings
* Store in FAISS vector DB

---

# 🚀 Running the Application

## 1️⃣ Start FastAPI Backend

```bash
uvicorn app:app --reload
```

Backend runs at:

```
http://localhost:8000
```

---

## 2️⃣ Start Streamlit UI

```bash
streamlit run streamlit_app.py
```

UI runs at:

```
http://localhost:8501
```

---

# 🔌 API Endpoints

## 🔹 Query (Streaming)

```
POST /stream
```

Request:

```json
{
  "query": "What is machine learning?",
  "session_id": "abc123"
}
```

---

## 🔹 Upload Document

```
POST /upload
```

* Accepts PDF files
* Adds to vector store

---

## 🔹 Health Check

```
GET /health
```

---

## 🔹 Clear Cache

```
POST /clear-cache
```

---

# 🧠 How It Works

## 🔄 RAG Pipeline Flow

```
User Query
   ↓
Query Rewriting
   ↓
Multi Query Generation
   ↓
Parallel Retrieval (FAISS)
   ↓
Reranking (Cross Encoder)
   ↓
Confidence Scoring
   ↓
Prompt Construction
   ↓
LLM Response (Streaming)
   ↓
Caching + Memory + Evaluation
```

---

# 📊 Confidence Scoring

* Based on reranker scores
* Detects:

  * Weak retrieval
  * Ambiguous queries
* Enables:

  * Guardrails
  * Fallback responses

---

# 🛠️ Configuration

Edit in:

```
rag_src/config/setting.py
```

Key parameters:

```python
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
K = 8

SEMANTIC_THRESHOLD = 0.85

CHAT_MODEL = "llama3:8b"
TEMPERATURE = 0.2

VECTOR_DB_PATH = "vectorstore/faiss_store"
```

---

# 🧪 Evaluation

Uses DeepEval:

* Faithfulness
* Answer Relevancy

Runs asynchronously for performance.

---

# 📈 Observability

Tracked via MLflow:

* Retrieval latency
* Chunk stats
* Query metrics

---

# 🐳 Dockerization (Optional)

## Build Image

```bash
docker build -t ai-ml-professor .
```

## Run Container

```bash
docker run -p 8000:8000 ai-ml-professor
```

---

# ⚠️ Common Issues

## ❌ Vector DB not found

➡ Run ingestion pipeline first

---

## ❌ Low confidence scores

➡ Causes:

* Missing vector DB
* Poor chunking
* Broad queries

---

## ❌ Streamlit not showing response

➡ Ensure:

* Correct API URL (`http://localhost:8000`)
* Streaming handled properly

---

## ❌ Upload 422 Error

➡ Ensure request key is:

```
file (not files)
```

---

# 🔮 Future Improvements

* Hybrid retrieval (BM25 + FAISS)
* Adaptive retrieval (dynamic K)
* Self-reflection (LLM verification)
* Auto-ingestion pipeline
* Multi-document management UI

---

# 👨‍💻 Author

Ashish Pal

---

# ⭐ If you found this useful

Give it a ⭐ and contribute!
