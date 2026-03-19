# Custom Temporal Context Graph Implementation

This document outlines the architecture and core functionality required to build a custom, minimal version of Graphiti, specifically optimized for **Pydantic AI Agents** using **PostgreSQL** for session management and **FalkorDB** for graph knowledge.

## 1. Architecture Overview

A high-functionality context graph system consists of three primary layers:

1.  **Persistence Layer (Postgres):** Stores raw episodes (chat messages), session metadata, and user/agent state. This is the source of truth for "what was said."
2.  **Knowledge Layer (FalkorDB):** Stores the "distilled" graph of entities and relationships. Each fact in this layer is bi-temporal (`valid_at`, `invalid_at`) and links back to its source episode in Postgres.
3.  **Intelligence Layer (Pydantic + LLM):** Handles extraction of entities/relationships from episodes and resolution of duplicates.

## 2. Data Models (Pydantic)

The interface between your agent and the graph should be strictly typed.

```python
from datetime import datetime
from typing import Optional, Any
from pydantic import BaseModel, Field

class Entity(BaseModel):
    name: str = Field(..., description="Canonical name of the entity")
    labels: list[str] = Field(default_factory=lambda: ["Entity"])
    summary: str = Field("", description="Summarized history of this entity")
    attributes: dict[str, Any] = {}

class Relationship(BaseModel):
    source: str
    target: str
    relation_type: str
    fact: str
    valid_at: datetime
    invalid_at: Optional[datetime] = None
    episodes: list[str] = Field(default_factory=list, description="UUIDs of source episodes")

class Episode(BaseModel):
    uuid: str
    session_id: str
    content: str
    created_at: datetime
    valid_at: datetime
```

## 3. Postgres Schema (Session History)

```sql
CREATE TABLE sessions (
    session_id UUID PRIMARY KEY,
    user_id UUID,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE episodes (
    uuid UUID PRIMARY KEY,
    session_id UUID REFERENCES sessions(session_id),
    content TEXT NOT NULL,
    source_type VARCHAR(50), -- 'message', 'json', 'text'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    valid_at TIMESTAMP WITH TIME ZONE NOT NULL
);
```

## 4. FalkorDB Implementation (Graph Knowledge)

### Core Schema
FalkorDB uses Cypher. Your nodes and edges should include temporal properties.

**Nodes:** `(e:Entity {uuid, name, summary, attributes, name_embedding})`
**Edges:** `(source)-[r:RELATES_TO {uuid, fact, fact_embedding, valid_at, invalid_at, episodes}]->(target)`

### Bi-Temporal Invalidation Logic
When a new fact is ingested that contradicts an old one, the system MUST invalidate the old fact instead of deleting it.

**The Invalidation Query:**
```cypher
MATCH (s:Entity {name: $source_name})-[r:RELATES_TO {name: $relation_type}]->(t:Entity {name: $target_name})
WHERE r.invalid_at IS NULL
  AND r.valid_at < $new_fact_valid_at
SET r.invalid_at = $new_fact_valid_at
```

## 5. The Ingestion Flow (Core Functionality)

1.  **Extract:** Feed the current episode and `n` previous episodes to an LLM using `Structured Output` with your `Relationship` Pydantic model.
2.  **Deduplicate:**
    *   For each extracted Entity, query FalkorDB: `MATCH (e:Entity) WHERE e.name_embedding <SIMILARITY> $embedding RETURN e`.
    *   If a match exists, use its canonical name.
3.  **Temporal Update:**
    *   Run the Invalidation Query for the new relationship.
    *   Create the new relationship in FalkorDB with `valid_at = episode.valid_at` and `invalid_at = NULL`.

## 6. Hybrid Retrieval Flow

When the agent needs context, don't just use RAG. Use a Graph-Augmented approach:

1.  **Semantic Retrieval:** Query FalkorDB for edges where `fact_embedding` is similar to the user query AND `invalid_at` is `NULL`.
2.  **Graph Neighborhood:** For the top 3-5 entities found, fetch their "Contextual Neighborhood" (1-hop neighbors).
3.  **System Message Assembly:** Convert the retrieved facts into a human-readable summary for the LLM.

```python
# Pseudo-code for context assembly
context = "The following is your current knowledge about the session context:\n"
for edge in retrieved_edges:
    context += f"- {edge.fact} (became true on {edge.valid_at})\n"
```

## 7. Performance Considerations (Minimalist)

*   **Concurrency:** Use a semaphore limit (e.g., 10) for LLM extraction calls to avoid rate limits.
*   **Indices:** Ensure FalkorDB has a vector index on `Entity.name_embedding` and `Relationship.fact_embedding`.
*   **Batching:** If ingesting many messages at once (e.g., historical imports), batch the `CREATE` statements in a single Cypher transaction.
