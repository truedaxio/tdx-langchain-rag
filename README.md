# Stateful LangGraph RAG Agent API

This project provides a FastAPI backend for a Retrieval-Augmented Generation (RAG) agent built using LangChain and LangGraph. The agent is stateful, meaning it can maintain conversational context across multiple interactions using a persistent checkpointer. It also includes an endpoint to dynamically ingest content from websites into the underlying Pinecone vector store.

# Features
* Stateful Conversational Agent: Remembers previous interactions within a conversation thread (thread_id).
* Retrieval-Augmented Generation (RAG): Uses a Pinecone vector store to retrieve relevant context before generating responses with an OpenAI LLM.
* Multiple Interaction Modes:
    * Standard Request/Response (/invoke)
    * Real-time Streaming Response via Server-Sent Events (/stream)
* Dynamic Knowledge Base Updates: Ingest and index content from websites directly via an API endpoint (/add-website).
* Asynchronous Processing: Built with FastAPI and asynchronous libraries for efficient I/O handling.
* Persistent State: Uses AsyncSqliteSaver to store conversation checkpoints in a local SQLite database (checkpointer.sqlite).
* Clear API Documentation: Automatic interactive documentation provided by FastAPI (Swagger UI at /docs, ReDoc at /redoc).

# Prerequisites
* Python 3.10+
* Access to OpenAI API (requires an API key)
* Access to Pinecone API (requires an API key and an existing index)

# Setup

1. Create and Activate a Virtual Environment:

```bash
python -m venv venv
source venv/bin/activate
Install Dependencies:
pip install -r requirements.txt
```

# Configure Environment Variables:
Create a file named .env in the project's root directory and add your API keys and Pinecone details:

```bash
#.env
OPENAI_API_KEY="sk-..."
PINECONE_API_KEY="your-pinecone-api-key"
PINECONE_INDEX_NAME="your-pinecone-index-name"
# CHECKPOINTER_DB_PATH="checkpointer.sqlite"
```

# Running the Application
Start the FastAPI server using Uvicorn:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
# The API will be available at http://localhost:8000 (or your machine's IP address if using 0.0.0.0).
```

# Checkpointer
This application uses langgraph.checkpoint.sqlite.AsyncSqliteSaver. A file named checkpointer.sqlite (or as configured) will be automatically created in the project root directory when the application starts. This file stores the state of conversations, allowing the agent to resume interactions within a specific thread_id. Do not delete this file unless you want to clear all conversation history.
