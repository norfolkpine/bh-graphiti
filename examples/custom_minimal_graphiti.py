"""
Custom Minimal Context Graph Implementation
-------------------------------------------
This script demonstrates the "Core Functionality" of a custom Graphiti-like
system using Pydantic, Postgres, and FalkorDB for agent session history.
"""

import asyncio
from datetime import datetime, timezone
from typing import Optional, List, Any
from uuid import uuid4
from pydantic import BaseModel, Field

# --- 1. Core Data Models (Pydantic) ---

class GraphNode(BaseModel):
    uuid: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    labels: List[str] = ["Entity"]
    summary: str = ""
    attributes: dict[str, Any] = {}

class GraphEdge(BaseModel):
    uuid: str = Field(default_factory=lambda: str(uuid4()))
    source: str  # Entity Name
    target: str  # Entity Name
    relation: str
    fact: str
    valid_at: datetime
    invalid_at: Optional[datetime] = None
    episodes: List[str] = []

class ChatEpisode(BaseModel):
    uuid: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    valid_at: datetime

# --- 2. Minimal Ingestion Pipeline ---

async def ingest_message(session_id: str, message: str, reference_time: datetime):
    """
    Simulated ingestion flow:
    1. Save to Postgres (Simulation)
    2. Extract Entities/Relationships (Simulation)
    3. Update FalkorDB with Temporal Logic (Simulation)
    """
    print(f"\n--- Ingesting Message for Session: {session_id} ---")

    # Create Episode
    episode = ChatEpisode(
        session_id=session_id,
        content=message,
        valid_at=reference_time
    )
    print(f"Step 1: Saved Episode {episode.uuid} to Postgres")

    # Extract Entities/Edges (In reality, this is an LLM call)
    # Using Pydantic models for Structured Output
    extracted_edges = [
        GraphEdge(
            source="Kendra",
            target="Adidas",
            relation="LOVES",
            fact="Kendra loves Adidas shoes.",
            valid_at=reference_time,
            episodes=[episode.uuid]
        )
    ]
    print(f"Step 2: Extracted {len(extracted_edges)} relationship(s) using LLM")

    # Update FalkorDB with Temporal Invalidation
    for edge in extracted_edges:
        # In a real system, you would execute this Cypher query:
        #
        # MATCH (s:Entity {name: $source})-[r:RELATES_TO {relation: $relation}]->(t:Entity {name: $target})
        # WHERE r.invalid_at IS NULL AND r.valid_at < $new_valid_at
        # SET r.invalid_at = $new_valid_at
        #
        print(f"Step 3: Invalidated old 'LOVES' facts between Kendra and Adidas (if any)")
        print(f"Step 4: Created new Relationship: {edge.fact} (valid_at: {edge.valid_at})")

# --- 3. Hybrid Retrieval Logic ---

async def get_agent_context(query: str, session_id: str):
    """
    Simulated Retrieval Flow:
    1. Semantic search for relevant facts
    2. Filtering for 'active' facts (invalid_at is NULL)
    3. Formatting for LLM Context
    """
    print(f"\n--- Retrieving Context for Query: '{query}' ---")

    # In reality, this would be a Cypher query with vector similarity:
    #
    # MATCH (s:Entity)-[r:RELATES_TO]->(t:Entity)
    # WHERE r.group_id = $session_id
    #   AND r.invalid_at IS NULL
    #   AND r.fact_embedding <SIMILARITY> $query_vector
    # RETURN r.fact, r.valid_at

    active_facts = [
        "Kendra loves Adidas shoes (as of March 2026)."
    ]

    context_str = "\n".join([f"- {fact}" for fact in active_facts])
    print(f"Retrieved active knowledge:\n{context_str}")
    return context_str

# --- Example Usage ---

async def main():
    session_id = "user_session_123"

    # Simulating a user message ingestion
    await ingest_message(
        session_id=session_id,
        message="I just bought a new pair of Adidas, I absolutely love them!",
        reference_time=datetime(2026, 3, 10, tzinfo=timezone.utc)
    )

    # Simulating the agent retrieving context later
    context = await get_agent_context(
        query="What brands does the user like?",
        session_id=session_id
    )

    print("\nResult: The agent can now use this context in its system prompt.")

if __name__ == "__main__":
    asyncio.run(main())
