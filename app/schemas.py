# app/schemas.py
from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, Dict, Any


class ChatRequest(BaseModel):
    """Request model for interacting with the chat agent."""

    message: str = Field(
        ...,
        description="The user's input message.",
        example="What is Task Decomposition?",
    )
    thread_id: Optional[str] = Field(
        None,
        description="The conversation thread ID. If None, a new thread may be started.",
        example="my-conversation-123",
    )
    # config: Optional[Dict[str, Any]] = Field(None, description="Optional configuration overrides.") # If needed


class ChatResponse(BaseModel):
    """Response model for the standard agent invocation."""

    response: str = Field(
        ...,
        description="The agent's response message.",
        example="Task decomposition breaks down complex tasks...",
    )
    thread_id: str = Field(
        ...,
        description="The conversation thread ID used or generated.",
        example="my-conversation-123",
    )
    # sources: Optional[List[Dict]] = Field(None, description="Optional list of source documents.") # If agent provides sources


class StreamResponseChunk(BaseModel):
    """Response model for a single chunk in a streaming response."""

    chunk: Optional[str] = Field(
        None, description="A piece of the agent's response message."
    )
    event: Optional[str] = Field(
        None, description="Indicates special events like 'start' or 'end'."
    )
    data: Optional[Dict[str, Any]] = Field(
        None, description="Additional data associated with an event."
    )
    thread_id: Optional[str] = Field(
        None,
        description="Conversation thread ID, typically sent with the first or last chunk.",
    )
    error: Optional[str] = Field(
        None, description="Indicates an error occurred during streaming."
    )


class WebsiteIngestionRequest(BaseModel):
    """
    Request model for the website ingestion endpoint.
    Requires a valid HTTP/HTTPS URL.
    """

    url: HttpUrl


class WebsiteIngestionResponse(BaseModel):
    """
    Response model for successful website ingestion.
    """

    message: str
    url: str  # Echo back the processed URL
    documents_added: int
