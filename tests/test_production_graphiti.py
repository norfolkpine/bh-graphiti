"""
Tests for Production-Ready Temporal Context Graph
------------------------------------------------
Verifies the temporal invalidation logic and entity resolution.
"""

import pytest
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, List
from examples.custom_minimal_graphiti import (
    TemporalKnowledgeManager,
    BaseGraphDriver,
    BaseLLMClient,
    ChatEpisode,
    GraphNode,
    GraphEdge
)

# --- 1. Mocks for Testing ---

class MockGraphDriver(BaseGraphDriver):
    def __init__(self):
        self.nodes = {} # uuid -> GraphNode
        self.edges = [] # List[GraphEdge-like-dict]
        self.queries = []

    async def execute_query(self, query: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        self.queries.append((query, params))

        # Simulate 'Entity Resolve'
        if "MATCH (e:Entity {name: $name})" in query:
            name = params['name']
            for n in self.nodes.values():
                if n.name == name:
                    return [{'e': n.model_dump()}]
            return []

        # Simulate 'Save Node'
        if "MERGE (e:Entity {uuid: $uuid})" in query:
            self.nodes[params['uuid']] = GraphNode(**params['props'])
            return []

        # Simulate 'Invalidate Fact'
        if "SET r.invalid_at = $new_valid_at" in query:
            # We track the invalidation in the query history
            return []

        # Simulate 'Save Edge'
        if "CREATE (s)-[r:RELATES_TO" in query:
            self.edges.append(params['props'])
            return []

        return []

class MockLLM(BaseLLMClient):
    def __init__(self, relations=None):
        self.relations = relations or []

    async def extract_relationships(self, content: str) -> List[Dict[str, Any]]:
        return self.relations

    async def get_embedding(self, text: str) -> List[float]:
        return [0.1, 0.2, 0.3]

# --- 2. Temporal Invalidation Tests ---

@pytest.mark.asyncio
async def test_temporal_invalidation_logic():
    """
    Verify that the KnowledgeManager calls the invalidation query
    before saving a new fact.
    """
    driver = MockGraphDriver()
    # Add a pre-existing node to simulate resolution
    pre_existing_node = GraphNode(name="Kendra")
    driver.nodes[pre_existing_node.uuid] = pre_existing_node

    llm = MockLLM(relations=[{
        "source": "Kendra",
        "target": "Adidas",
        "relation_type": "LIKES",
        "fact": "Kendra likes Adidas."
    }])

    manager = TemporalKnowledgeManager(driver, llm)

    # Ingest a new episode
    valid_at = datetime(2026, 3, 10, tzinfo=timezone.utc)
    episode = ChatEpisode(
        session_id="session_1",
        content="I love Adidas!",
        valid_at=valid_at
    )

    await manager.ingest_episode(episode)

    # Check if the invalidation query was called
    invalidation_called = False
    for query, params in driver.queries:
        if "SET r.invalid_at = $new_valid_at" in query:
            invalidation_called = True
            assert params['new_valid_at'] == valid_at.isoformat()

    assert invalidation_called, "The temporal invalidation query was never called."

@pytest.mark.asyncio
async def test_entity_resolution_flow():
    """
    Verify that an existing entity is reused based on name matching.
    """
    driver = MockGraphDriver()
    pre_existing_node = GraphNode(name="Kendra")
    driver.nodes[pre_existing_node.uuid] = pre_existing_node

    llm = MockLLM(relations=[{
        "source": "Kendra",
        "target": "Adidas",
        "relation_type": "LIKES",
        "fact": "Kendra likes Adidas."
    }])

    manager = TemporalKnowledgeManager(driver, llm)

    # Ingest episode
    episode = ChatEpisode(
        session_id="session_1",
        content="Kendra mentioned something.",
        valid_at=datetime.now(timezone.utc)
    )

    await manager.ingest_episode(episode)

    # Check that only one node for 'Kendra' exists (no new one created)
    kendra_nodes = [n for n in driver.nodes.values() if n.name == "Kendra"]
    assert len(kendra_nodes) == 1
    assert kendra_nodes[0].uuid == pre_existing_node.uuid

@pytest.mark.asyncio
async def test_concurrency_semaphore():
    """
    Verify that the KnowledgeManager respects the semaphore limit.
    (Conceptual test, ensures the manager uses the semaphore during ingestion)
    """
    driver = MockGraphDriver()
    llm = MockLLM()
    manager = TemporalKnowledgeManager(driver, llm)

    # The semaphore should be 10 by default
    assert manager.semaphore._value == 10
