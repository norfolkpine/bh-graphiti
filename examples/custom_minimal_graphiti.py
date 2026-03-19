"""
Production-Ready Temporal Context Graph Reference Implementation
--------------------------------------------------------------
This script provides a robust, production-grade template for building a
custom Graphiti-style system using Pydantic, Postgres, and FalkorDB.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional, List, Any, Dict
from uuid import uuid4
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, ConfigDict
from tenacity import retry, stop_after_attempt, wait_exponential

# --- 1. Production Logging & Configuration ---

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ProductionGraphContext")

# --- 2. Production Data Models (Strictly Typed) ---

class GraphNode(BaseModel):
    uuid: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    labels: List[str] = ["Entity"]
    summary: str = ""
    attributes: Dict[str, Any] = {}
    name_embedding: Optional[List[float]] = None

    model_config = ConfigDict(extra='allow')

class GraphEdge(BaseModel):
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

class ChatEpisode(BaseModel):
    uuid: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    content: str
    source_type: str = "message"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    valid_at: datetime

# --- 3. External Service Interfaces (Abstraction for Mocks/Real) ---

class BaseGraphDriver(ABC):
    @abstractmethod
    async def execute_query(self, query: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        pass

class BaseLLMClient(ABC):
    @abstractmethod
    async def extract_relationships(self, content: str) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    async def get_embedding(self, text: str) -> List[float]:
        pass

# --- 4. Production Knowledge Manager ---

class TemporalKnowledgeManager:
    def __init__(self, driver: BaseGraphDriver, llm: BaseLLMClient):
        self.driver = driver
        self.llm = llm
        self.semaphore = asyncio.Semaphore(10) # Concurrency limit

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def ingest_episode(self, episode: ChatEpisode):
        """
        Production Ingestion Pipeline:
        1. Extract Relationships via LLM
        2. Resolve Entity Nodes (Deduplicate)
        3. Apply Temporal Invalidation Logic
        4. Upsert Graph Objects
        """
        logger.info(f"Ingesting episode: {episode.uuid} for session: {episode.session_id}")

        async with self.semaphore:
            # 1. Extraction (Using Pydantic for validation)
            raw_relations = await self.llm.extract_relationships(episode.content)

            for rel in raw_relations:
                try:
                    # 2. Entity Resolution (Deduplicate)
                    source_node = await self._resolve_entity(rel['source'])
                    target_node = await self._resolve_entity(rel['target'])

                    # 3. Temporal Invalidation (The core Graphiti logic)
                    await self._invalidate_old_facts(
                        source_node.uuid,
                        target_node.uuid,
                        rel['relation_type'],
                        episode.valid_at
                    )

                    # 4. Create New Fact
                    edge = GraphEdge(
                        source_uuid=source_node.uuid,
                        target_uuid=target_node.uuid,
                        relation_type=rel['relation_type'],
                        fact=rel['fact'],
                        valid_at=episode.valid_at,
                        episodes=[episode.uuid]
                    )
                    await self._save_edge(edge)
                    logger.info(f"New fact saved: {edge.fact}")

                except Exception as e:
                    logger.error(f"Failed to process relationship {rel.get('fact')}: {e}")

    async def _resolve_entity(self, name: str) -> GraphNode:
        """Simple Resolution: Exact Name Match (Extend with Vector Search)"""
        query = "MATCH (e:Entity {name: $name}) RETURN e"
        records = await self.driver.execute_query(query, {"name": name})

        if records:
            return GraphNode(**records[0]['e'])

        # Create new node if not found
        node = GraphNode(name=name)
        await self._save_node(node)
        return node

    async def _invalidate_old_facts(self, src: str, dst: str, rel_type: str, new_valid_at: datetime):
        """Invalidate facts that are contradicted by the new information."""
        query = """
        MATCH (s:Entity {uuid: $src})-[r:RELATES_TO {relation_type: $rel_type}]->(t:Entity {uuid: $dst})
        WHERE r.invalid_at IS NULL AND r.valid_at < $new_valid_at
        SET r.invalid_at = $new_valid_at
        """
        await self.driver.execute_query(query, {
            "src": src, "dst": dst, "rel_type": rel_type, "new_valid_at": new_valid_at.isoformat()
        })

    async def _save_node(self, node: GraphNode):
        query = "MERGE (e:Entity {uuid: $uuid}) SET e += $props"
        await self.driver.execute_query(query, {"uuid": node.uuid, "props": node.model_dump(exclude={'name_embedding'})})

    async def _save_edge(self, edge: GraphEdge):
        query = """
        MATCH (s:Entity {uuid: $src}), (t:Entity {uuid: $dst})
        CREATE (s)-[r:RELATES_TO {uuid: $uuid}]->(t)
        SET r += $props
        """
        await self.driver.execute_query(query, {
            "src": edge.source_uuid,
            "dst": edge.target_uuid,
            "uuid": edge.uuid,
            "props": edge.model_dump(exclude={'fact_embedding'})
        })

# --- 5. Mock Example ---

class MockDriver(BaseGraphDriver):
    def __init__(self):
        self.nodes = {}
        self.edges = []

    async def execute_query(self, query: str, params: Dict[str, Any]):
        # Mock logic to simulate FalkorDB behavior
        if "MATCH (e:Entity {name: $name})" in query:
            name = params['name']
            for n in self.nodes.values():
                if n.name == name: return [{'e': n.model_dump()}]
            return []

        if "MERGE (e:Entity" in query:
            self.nodes[params['uuid']] = GraphNode(uuid=params['uuid'], **params['props'])

        if "CREATE (s)-[r:RELATES_TO" in query:
            self.edges.append(params['props'])

        return []

class MockLLM(BaseLLMClient):
    async def extract_relationships(self, content: str):
        return [{
            "source": "Kendra",
            "target": "Adidas",
            "relation_type": "LIKES",
            "fact": "Kendra likes Adidas shoes."
        }]

    async def get_embedding(self, text: str):
        return [0.1, 0.2, 0.3]

# --- Main Runtime ---

async def main():
    manager = TemporalKnowledgeManager(MockDriver(), MockLLM())

    episode = ChatEpisode(
        session_id="session_1",
        content="Kendra mentioned she really likes Adidas shoes.",
        valid_at=datetime.now(timezone.utc)
    )

    await manager.ingest_episode(episode)

if __name__ == "__main__":
    asyncio.run(main())
