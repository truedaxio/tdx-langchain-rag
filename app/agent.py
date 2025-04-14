# app/agent.py
import os
from dotenv import load_dotenv
from langgraph.checkpoint.memory import (
    MemorySaver,
)  # WARNING: Replace for production

from langgraph.checkpoint.sqlite import (
    SqliteSaver,
)  # Example persistent checkpointer

# from langgraph.checkpoint.redis import RedisSaver # Example persistent checkpointer
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_pinecone import (
    PineconeVectorStore,
)  # Assuming Pinecone is used as per user code
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone

# Load environment variables (API keys, etc.)
load_dotenv()

# --- Checkpointer Setup ---
# WARNING: MemorySaver is not suitable for production as state is lost on restart
# and not shared across multiple server processes (workers).
# Replace with a persistent, shared checkpointer like SqliteSaver, RedisSaver, or a custom DB implementation.
# memory = MemorySaver()
# Example using SQLite (requires `pip install aiosqlite`)
# memory = SqliteSaver.from_conn_string("checkpointer.sqlite")
# Example using Redis (requires `pip install redis aioredis`)
# import redis
# redis_client = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True)
# memory = RedisSaver(redis_client)


# --- LLM Initialization ---
# Ensure API keys are set in the environment (.env file)
llm = ChatOpenAI(
    model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY")
)

# --- Vector Store and Retriever Tool Setup ---
# Initialize embeddings and vector store based on user code
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large", api_key=os.getenv("OPENAI_API_KEY")
)

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "rag-app-index"  # Or load from config/env
index = pc.Index(index_name)
vector_store = PineconeVectorStore(embedding=embeddings, index=index)


@tool
def retrieve(query: str) -> str:
    """Retrieve information related to a query from the vector store."""
    retrieved_docs = vector_store.similarity_search(
        query, k=3
    )  # Adjust k as needed
    # Format the retrieved documents for the agent
    return "\n\n".join([doc.page_content for doc in retrieved_docs])


tools = [retrieve]


def get_vector_store() -> PineconeVectorStore:
    """
    Dependency function that returns the shared PineconeVectorStore instance.
    Ensures the instance is initialized before returning.
    """
    if vector_store is None:
        # In a real app, initialization should happen reliably at startup.
        # Logging this error helps diagnose configuration issues.
        print(
            "CRITICAL: Vector store dependency requested before initialization."
        )
        raise RuntimeError("Vector store has not been initialized.")
    return vector_store


# app/agent.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools import tool
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # For type hinting

# Load environment variables if not already done globally
load_dotenv()

# --- Tool Definition ---
# It's often better to initialize tools once if they have state or setup cost
# Consider initializing vector_store connection globally or via lifespan as well
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME", "rag-app-index")
index = pc.Index(index_name)
vector_store = PineconeVectorStore(embedding=embeddings, index=index)


@tool
def retrieve(query: str) -> str:
    """Retrieve information related to a query from the vector store."""
    retrieved_docs = vector_store.similarity_search(query, k=3)
    return "\n\n".join([doc.page_content for doc in retrieved_docs])


tools = [retrieve]

# --- LLM Initialization ---
llm = ChatOpenAI(
    model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY")
)


def initialize_agent_executor(checkpointer: AsyncSqliteSaver):
    """
    Initializes and returns the LangGraph agent executor using the provided checkpointer.
    """
    print("Initializing agent executor with checkpointer...")
    # Pass the LLM, tools, and the actual checkpointer instance
    agent_executor = create_react_agent(llm, tools, checkpointer=checkpointer)
    print("Agent executor initialized.")
    return agent_executor


# Remove previous global 'memory' and 'agent_executor' variables
# Remove get_agent_executor() and get_checkpointer() functions
