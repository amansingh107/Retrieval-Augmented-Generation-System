# Agentic RAG System 🤖📚

A modular, production-ready **Retrieval-Augmented Generation (RAG)** system that goes beyond simple document search. It features an **autonomous agent** capable of routing queries between local documents, web search, and conversational memory, built with a focus on latency reduction and accuracy.

## 🚀 Key Features

* **🧠 Intelligent Routing:** An agent that decides whether to answer from memory, search local documents, or browse the web based on user intent.
* **🔍 Hybrid Search:** Combines **Dense Retrieval** (Semantic/Vector) and **Sparse Retrieval** (BM25/Keyword) for maximum accuracy.
* **⚡ Multi-Level Caching:**
    * **Level 1:** Redis-backed Semantic Cache to serve repeated queries instantly.
    * **Level 2:** Local embedding cache to avoid re-computing vectors.
* **🥈 Reranking:** Uses **Cohere Rerank** to re-score retrieval results, significantly reducing hallucinations.
* **🌐 Web Fallback:** Automatically falls back to **DuckDuckGo Search** if local documents do not contain the answer.
* **📂 Multi-Format Ingestion:** Supports ingestion of **PDFs** and **Websites** on the fly via slash commands.

---

## 🛠️ Architecture

The system is built on a modular architecture separating the "Brain" (Agent), "Hands" (Tools), and "Memory" (Redis).

```mermaid
graph TD
    User --> Agent
    Agent --> Router{Router}
    
    Router -- "Hi / Help" --> LLM
    Router -- "News / Weather" --> WebSearch[DuckDuckGo]
    Router -- "Specific Q" --> RAG[RAG Pipeline]
    
    subgraph RAG Pipeline
    RAG --> Hybrid[Hybrid Search]
    Hybrid --> VectorDB[(ChromaDB)]
    Hybrid --> BM25[BM25 Index]
    Hybrid --> Reranker[Cohere Rerank]
    end
    
    Reranker -- Low Score --> WebSearch
    Reranker -- High Score --> Context
    
    WebSearch --> Context
    Context --> LLM[Cohere Command-R]
    LLM --> User

* * *

## 📦 Tech Stack

-   **LLM:** Cohere `command-r` (or OpenAI GPT-4 optionally)
    
-   **Vector DB:** ChromaDB (Persistent)
    
-   **Embeddings:** SentenceTransformers (`all-MiniLM-L6-v2`)
    
-   **Orchestration:** Custom Python (No heavy frameworks like LangChain used for core logic)
    
-   **Memory/Cache:** Redis
    
-   **Search:** DuckDuckGo Search
    
-   **Ingestion:** `pypdf`, `beautifulsoup4`
    

* * *

## 🏃‍♂️ Getting Started

### 1\. Prerequisites

-   Python 3.10+
    
-   **Redis** installed and running (`sudo apt install redis-server` on Linux/WSL or use Docker).
    

### 2\. Installation

Clone the repository and install dependencies:

Bash

    git clone [https://github.com/yourusername/agentic-rag.git](https://github.com/yourusername/agentic-rag.git)
    cd agentic-rag
    pip install -r requirements.txt

### 3\. Configuration

Create a `.env` file in the root directory:

Code snippet

    # Required for Reranking & Generation
    COHERE_API_KEY=your_cohere_key_here
    
    # Optional: If you want to use GPT models
    OPENAI_API_KEY=your_openai_key_here

### 4\. Running the App

Start the main application loop:

Bash

    python main.py

* * *

## 🎮 Usage Guide

Once the application is running, you can interact with it naturally or use slash commands to feed it data.

### **Chatting**

> **You:** Hi, how are you? **Agent:** I'm doing well! How can I help you today?

> **You:** Who won the Super Bowl in 2024? **Agent:** _(Routes to Web Search)_ The Kansas City Chiefs won Super Bowl LVIII...

### **Adding Data (Ingestion)**

**1\. Add a Website:** Scrape a URL and add it to the vector database instantly.

Plaintext

    /add [https://en.wikipedia.org/wiki/Generative_artificial_intelligence](https://en.wikipedia.org/wiki/Generative_artificial_intelligence)

**2\. Add a PDF:** Read a local PDF file and index it.

Plaintext

    /pdf ./documents/employee_handbook.pdf

**3\. Query Your Data:** Now that the data is added, just ask:

> **You:** What does the handbook say about remote work? **Agent:** _(Routes to Vector DB)_ According to the handbook, remote work is allowed for...

* * *

## 📂 Project Structure

Plaintext

    agentic_rag/
    ├── main.py              # Entry point & CLI loop
    ├── requirements.txt     # Python dependencies
    ├── .env                 # API Keys
    └── src/
        ├── __init__.py
        ├── agent.py         # Router & Main Logic
        ├── memory.py        # Redis Caching & Chat History
        └── tools.py         # RAG, Search, Ingestion, & Reranking

## 🔮 Future Improvements

-   \[ \] **Graph RAG:** Implement a Knowledge Graph integration (Neo4j) for better relationship mapping.
    
-   \[ \] **UI:** Build a frontend using Streamlit or Chainlit.
    
-   \[ \] **Docker:** Containerize the application and Redis for one-click deployment.
    

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

[MIT](https://choosealicense.com/licenses/mit/)