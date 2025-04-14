# app/main.py
import traceback
import uuid
import json
import os
from fastapi import FastAPI, HTTPException, Body, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import (
    CORSMiddleware,
)  # Optional: Add CORS for browser clients

from .schemas import ChatRequest, ChatResponse, StreamResponseChunk

import logging
from typing import List

# LangChain Imports
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from contextlib import asynccontextmanager

# App specific imports
from app.schemas import WebsiteIngestionRequest, WebsiteIngestionResponse
from app.agent import (
    get_vector_store,
    initialize_agent_executor,
)  # Import the dependency function

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Optional: Configure CORS if the API will be called from a browser frontend
# origins = [
#     "http://localhost",
#     "http://localhost:3000", # Example frontend port
# ]
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# --- Get Agent Instance ---
# Load the agent once at startup
lifespan_agent_executor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    global lifespan_agent_executor
    logger.info("Application startup: Initializing resources...")

    # Correctly initialize SqliteSaver using async with
    checkpointer_conn_string = "checkpointer.sqlite"  # Or load from config
    checkpointer_cm = AsyncSqliteSaver.from_conn_string(
        checkpointer_conn_string
    )

    async with checkpointer_cm as checkpointer_instance:
        logger.info(
            f"SqliteSaver checkpointer connected using: {checkpointer_conn_string}"
        )
        # Initialize the agent executor *after* the checkpointer is ready
        # Pass the actual yielded checkpointer_instance
        lifespan_agent_executor = initialize_agent_executor(
            checkpointer=checkpointer_instance
        )
        logger.info("Agent executor successfully initialized.")

        yield  # Application runs after this point

    # Shutdown logic (connection closed automatically by async with)
    logger.info("Application shutdown: Releasing resources...")
    lifespan_agent_executor = None  # Clear the global reference


# --- FastAPI App Initialization ---
app = FastAPI(
    title="LangGraph RAG Agent API",
    version="1.0",
    description="API to interact with a stateful RAG agent powered by LangGraph.",
    lifespan=lifespan,
)


# --- Root Endpoint ---
@app.get("/", summary="API Root/Health Check")
async def root():
    """Provides a simple health check endpoint."""
    return {"message": "LangGraph RAG Agent API is running"}


# Dependency to get the initialized agent executor
def get_agent_dependency():
    if lifespan_agent_executor is None:
        # This should ideally not happen if lifespan management is correct
        logger.error("Agent executor accessed before initialization.")
        raise HTTPException(status_code=503, detail="Service not ready")
    return lifespan_agent_executor


# --- Invoke Endpoint ---
@app.post(
    "/invoke",
    response_model=ChatResponse,
    summary="Invoke the agent (Request/Response)",
    description="Sends a message to the agent and waits for the final response.",
)
async def invoke_agent(
    request: ChatRequest = Body(...),
    agent_runnable=Depends(get_agent_dependency),
):
    """Handles a single turn interaction with the RAG agent."""
    thread_id = request.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    input_message = {"messages": [("user", request.message)]}

    try:
        # Use the injected agent_runnable
        result = await agent_runnable.ainvoke(input_message, config=config)

        # --- Extract final AI response (ensure this logic matches your agent's output) ---
        ai_message_content = "Agent did not produce a final response."
        if messages := result.get("messages"):
            if isinstance(messages, list) and len(messages) > 0:
                for msg in reversed(messages):
                    if msg.type == "ai" and not getattr(
                        msg, "tool_calls", None
                    ):
                        ai_message_content = msg.content
                        break
        # Fallback if structure is different
        if ai_message_content == "Agent did not produce a final response.":
            ai_message_content = str(
                result.get("answer", result.get("output", ai_message_content))
            )

        logger.info(f"Agent invocation successful for thread_id: {thread_id}")
        return ChatResponse(response=ai_message_content, thread_id=thread_id)

    except Exception as e:
        logger.error(
            f"Error invoking agent for thread {thread_id}: {e}", exc_info=True
        )  # Log traceback
        raise HTTPException(
            status_code=500, detail=f"Agent invocation failed: {str(e)}"
        )


# --- Streaming Endpoint Generator  ---
async def stream_agent_response_generator(
    message: str, thread_id: str, agent_runnable
):  # Pass agent
    config = {"configurable": {"thread_id": thread_id}}
    input_message = {"messages": [("user", message)]}
    current_full_response = ""
    final_response_started = False

    try:
        # Use the passed agent_runnable
        async for chunk in agent_runnable.astream(
            input_message, stream_mode="updates", config=config
        ):
            print(dir(chunk))
            print(chunk.keys())
            print(chunk.get("messages"))
            new_content_delta = ""
            # --- Extract delta (ensure logic matches your agent's stream output) ---
            # if messages := chunk.get("messages"):
            #     if isinstance(messages, list) and len(messages) > 0:
            #         last_msg = messages[-1]
            #         if last_msg.type == "ai" and hasattr(last_msg, "content"):
            #             if len(last_msg.content) > len(current_full_response):
            #                 new_content_delta = last_msg.content[
            #                     len(current_full_response) :
            #                 ]
            #                 current_full_response = last_msg.content
            #                 final_response_started = True
            #             elif (
            #                 last_msg.content != current_full_response
            #                 and final_response_started
            #             ):
            #                 new_content_delta = last_msg.content
            #                 current_full_response = last_msg.content

            response_chunk = StreamResponseChunk(chunk=new_content_delta)
            yield f"data: {str(chunk.get('messages'))}\n\n"
            # if new_content_delta:
            # response_chunk = StreamResponseChunk(chunk=new_content_delta)
            # yield f"data: {response_chunk.model_dump_json()}\n\n"

        end_event = StreamResponseChunk(event="end", thread_id=thread_id)
        yield f"data: {end_event.model_dump_json()}\n\n"

    except Exception as e:
        logger.error(
            f"Error streaming agent response for thread {thread_id}: {e}",
            exc_info=True,
        )
        error_event = StreamResponseChunk(error=str(e), thread_id=thread_id)
        try:
            yield f"data: {error_event.model_dump_json()}\n\n"
        except Exception as inner_e:
            logger.error(f"Failed to yield error event: {inner_e}")


@app.post("/stream", summary="Stream the agent's response")
async def stream_agent(
    request: ChatRequest = Body(...),
    agent_runnable=Depends(get_agent_dependency),  # Use dependency injection
):
    thread_id = request.thread_id or str(uuid.uuid4())
    logger.info(f"Streaming agent response for thread_id: {thread_id}")
    return StreamingResponse(
        stream_agent_response_generator(
            request.message, thread_id, agent_runnable
        ),  # Pass agent
        media_type="text/event-stream",
    )


# --- Optional: Add programmatic run for convenience ---
# import uvicorn
# if __name__ == "__main__":
#     # Note: Running programmatically like this might affect reload behavior.
#     # It's often better to use the uvicorn CLI command.
#     uvicorn.run(app, host="0.0.0.0", port=8000)


# --- Website Ingestion Endpoint ---
@app.post(
    "/add-website",
    response_model=WebsiteIngestionResponse,
    summary="Ingest Content from a Website URL",
    description="Loads content from the provided URL, splits it into chunks, "
    "and adds it to the Pinecone vector store.",
    tags=["Ingestion"],  # Tag for API documentation organization
)
async def add_website(
    request: WebsiteIngestionRequest,  # Inject and validate request body
    vector_store: PineconeVectorStore = Depends(
        get_vector_store
    ),  # Inject shared vector store
):
    """
    Endpoint to load, process, and ingest content from a website URL into the
    shared Pinecone vector store.
    """
    logger.info(f"Received request to ingest URL: {request.url}")

    # 1. Load Content using WebBaseLoader
    loaded_docs: List[Document] = []
    try:
        # Consider adding requests_kwargs for timeout, headers etc. if needed
        # loader = WebBaseLoader(str(request.url), requests_kwargs={'timeout': 15})
        loader = WebBaseLoader(str(request.url))
        logger.info(f"Loading content from {request.url}...")

        # Use alazy_load for native async iteration
        async for doc in loader.alazy_load():
            loaded_docs.append(doc)

        if not loaded_docs:
            logger.warning(f"No content loaded from URL: {request.url}")
            raise HTTPException(
                status_code=404,
                detail=f"Could not load any content from the URL: {request.url}. "
                "The page might be empty, require JavaScript, or block scraping.",
            )
        logger.info(
            f"Successfully loaded {len(loaded_docs)} document(s) from {request.url}."
        )

    except Exception as e:
        logger.error(f"Error loading URL {request.url}: {e}", exc_info=True)
        raise HTTPException(
            status_code=400,  # Bad request likely due to URL issue or loading problem
            detail=f"Failed to load content from URL '{request.url}': {str(e)}",
        )

    # 2. Split Content using RecursiveCharacterTextSplitter
    # Ensure chunk_size/overlap match the RAG agent's configuration
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,  # Consider token length function if using token-based models
        add_start_index=True,
    )

    try:
        chunks = text_splitter.split_documents(loaded_docs)
        if not chunks:
            logger.error(
                f"Failed to split loaded documents from {request.url} into chunks."
            )
            raise HTTPException(
                status_code=500,
                detail="Failed to split documents into chunks after successful loading.",
            )
        logger.info(f"Split content into {len(chunks)} chunks.")
    except Exception as e:
        logger.error(
            f"Error splitting documents from {request.url}: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"An internal error occurred while splitting documents: {str(e)}",
        )

    # 3. Add Chunks to Pinecone Vector Store
    try:
        logger.info(f"Adding {len(chunks)} chunks to the vector store...")
        # Use aadd_documents for async operation
        # Specify namespace if desired for organization within the Pinecone index
        await vector_store.aadd_documents(chunks, namespace="web-content")
        logger.info(
            f"Successfully added {len(chunks)} chunks for {request.url}."
        )

    except Exception as e:
        logger.error(
            f"Error adding documents to vector store for {request.url}: {e}",
            exc_info=True,
        )
        # Consider more specific exception handling for Pinecone client errors
        raise HTTPException(
            status_code=500,  # Or 503 if Pinecone service seems unavailable
            detail=f"Failed to add documents to vector store: {str(e)}",
        )

    # 4. Return Success Response
    num_docs_added = len(chunks)
    response_payload = WebsiteIngestionResponse(
        message="Successfully loaded and added website content to vector store.",
        url=str(request.url),
        documents_added=num_docs_added,
    )
    logger.info(
        f"Ingestion complete for {request.url}. Added {num_docs_added} documents."
    )
    return response_payload


# --- Add application startup/shutdown events if needed ---
# @app.on_event("startup")
# async def startup_event():
#     logger.info("Application startup: Initializing resources...")
#     # Initialize vector_store_instance here if using global variable approach
#     global vector_store_instance
#     vector_store_instance = initialize_vector_store() # Replace with your actual init function
#     logger.info("Vector store initialized.")
