# Production-Ready Custom Temporal Context Graph

This guide outlines how to build a production-grade temporal context graph system for AI agents using **PostgreSQL**, **FalkorDB**, and **Pydantic**. This architecture is inspired by Graphiti but simplified for direct integration into custom agent stacks.

## 1. Production Architecture

A production-ready system must handle high concurrency, provide observability, and be resilient to failures in external services (LLMs/Databases).

### Core Components
1.  **Postgres (Source of Truth):** Stores raw `Sessions` and `Episodes`.
2.  **FalkorDB (Knowledge Layer):** Stores the temporal graph of `Entities` and `Relationships`.
3.  **Knowledge Manager (Logic):** Coordinates extraction, deduplication, and temporal invalidation.
4.  **Embedder & LLM Clients:** Wrappers with built-in retries and tracing.

## 2. Production Data Models (Pydantic)

Use strict typing and validation.

```python
from datetime import datetime
from typing import Optional, Any, List, Dict
from uuid import uuid4
from pydantic import BaseModel, Field, ConfigDict

class Entity(BaseModel):
    uuid: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    labels: List[str] = ["Entity"]
    summary: str = ""
    attributes: Dict[str, Any] = {}
    name_embedding: Optional[List[float]] = None

    model_config = ConfigDict(extra='allow')

class Relationship(BaseModel):
    uuid: str = Field(default_factory=lambda: str(uuid4()))
    source_uuid: str
    target_uuid: str
    relation_type: str
    fact: str
    fact_embedding: Optional[List[float]] = None
    valid_at: datetime
    invalid_at: Optional[datetime] = None
    episodes: List[str] = []
    attributes: Dict[str, Any] = {}
```

## 3. Production Ingestion Pipeline

### Step 1: Extraction with Structured Output
Use LLM "Structured Output" (OpenAI/Gemini) to ensure the model adheres to your Pydantic schema.

### Step 2: Entity Resolution (Deduplication)
Always check if an entity exists before creating a new one. Use a hybrid of:
1.  **Exact Name Match:** Fastest, cheapest.
2.  **Vector Similarity:** Use FalkorDB's `vecf32` indices to find entities with similar names.
3.  **LLM Resolve:** Only for high-uncertainty matches.

### Step 3: Temporal Invalidation (The "Graphiti" Logic)
When a new fact arrives, invalidate the old one. This ensures your agent always has the "current" truth while retaining history.

**Cypher Invalidation Pattern:**
```cypher
MATCH (s:Entity {uuid: $source_uuid})-[r:RELATES_TO {relation_type: $rel_type}]->(t:Entity {uuid: $target_uuid})
WHERE r.invalid_at IS NULL AND r.valid_at < $new_valid_at
SET r.invalid_at = $new_valid_at
```

## 4. Operational Best Practices

### Concurrency & Rate Limiting
External LLM APIs have rate limits. Use an `asyncio.Semaphore` to limit concurrent extraction tasks.

```python
semaphore = asyncio.Semaphore(10) # Max 10 concurrent LLM calls

async def extract_with_limit(episode):
    async with semaphore:
        return await llm_client.extract(episode)
```

### Observability
Integrate OpenTelemetry for tracing the ingestion pipeline. This allows you to see exactly where bottlenecks or failures occur.

*   **Logs:** Use structured logging (JSON) to track entity resolutions and invalidations.
*   **Traces:** Wrap the `Extract -> Resolve -> Store` loop in a trace span.

### Resilience (Retries)
Use `tenacity` to handle transient network errors or LLM 429s.

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def safe_db_call(query, params):
    return await falkor_driver.execute_query(query, params)
```

### Security
*   **API Keys:** Use environment variables or a secret manager.
*   **Graph Injection:** Use parameterized Cypher queries to prevent graph injection attacks (similar to SQL injection).

## 5. Retrieval Strategy (RAG vs GraphRAG)

In production, don't just return raw strings. Return a structured `KnowledgeContext` object.

1.  **Semantic Search:** Retrieve edges with `invalid_at IS NULL` and high vector similarity.
2.  **Neighborhood Walk:** Fetch properties and summaries for all unique nodes involved in the retrieved edges.
3.  **Provenance:** Always include the `valid_at` and `episodes` list in the context so the agent can cite its sources.
