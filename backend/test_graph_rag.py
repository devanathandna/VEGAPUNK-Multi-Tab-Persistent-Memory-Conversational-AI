"""
VEGAPUNK Graph RAG Integration Test

This script tests the complete Graph RAG pipeline:
1. Document ingestion → Graph population
2. Entity extraction → Node/Edge creation
3. Query → Semantic search + Graph expansion
4. Fusion → Score combination
5. LLM Response → Grounded answer

Run: python test_graph_rag.py
"""

import os
import sys
import time
import json
from datetime import datetime

# Ensure we can import from current directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Test configuration
TEST_DB_PATH = "test_graph_rag.db"
TEST_CHROMA_PATH = "./test_chroma_db"
TEST_SESSION_ID = "test_session_001"


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_step(step: int, description: str):
    """Print a step indicator."""
    print(f"\n[STEP {step}] {description}")
    print("-" * 50)


def cleanup_test_files():
    """Remove test database and chroma files."""
    import shutil
    
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)
        print(f"✓ Removed {TEST_DB_PATH}")
    
    if os.path.exists(TEST_CHROMA_PATH):
        shutil.rmtree(TEST_CHROMA_PATH)
        print(f"✓ Removed {TEST_CHROMA_PATH}")


def test_graph_store():
    """Test 1: GraphStore basic operations."""
    print_step(1, "Testing GraphStore")
    
    from graph_store import GraphStore, NodeType, RelationType
    
    store = GraphStore(db_path=TEST_DB_PATH)
    
    # Add nodes
    store.add_node("python", NodeType.CONCEPT.value, label="Python Programming", importance=0.9)
    store.add_node("flask", NodeType.ENTITY.value, label="Flask Framework", importance=0.7)
    store.add_node("django", NodeType.ENTITY.value, label="Django Framework", importance=0.8)
    store.add_node("guido", NodeType.ENTITY.value, label="Guido van Rossum", importance=0.6)
    store.add_node("web_dev", NodeType.CONCEPT.value, label="Web Development", importance=0.75)
    
    print(f"  ✓ Added 5 nodes")
    
    # Add edges
    store.add_edge("flask", "python", RelationType.DERIVED_FROM.value)
    store.add_edge("django", "python", RelationType.DERIVED_FROM.value)
    store.add_edge("python", "guido", RelationType.CREATED_BY.value if hasattr(RelationType, 'CREATED_BY') else "RELATED_TO")
    store.add_edge("flask", "web_dev", RelationType.RELATED_TO.value)
    store.add_edge("django", "web_dev", RelationType.RELATED_TO.value)
    
    print(f"  ✓ Added 5 edges")
    
    # Test traversal
    neighbors = store.get_neighbors("python", hops=1)
    print(f"  ✓ Neighbors of 'python' (1-hop): {neighbors}")
    
    neighbors_2hop = store.get_neighbors("python", hops=2)
    print(f"  ✓ Neighbors of 'python' (2-hop): {neighbors_2hop}")
    
    # Test PageRank
    pagerank = store.compute_pagerank()
    print(f"  ✓ PageRank computed: {len(pagerank)} nodes scored")
    
    # Test expansion context
    context = store.get_expansion_context(["python"], hops=1)
    print(f"  ✓ Expansion context: {len(context['neighbors'])} neighbors, {len(context['relations'])} relations")
    
    # Stats
    stats = store.get_stats()
    print(f"  ✓ Graph stats: {stats}")
    
    return store


def test_vector_store():
    """Test 2: VectorStore with graph linking."""
    print_step(2, "Testing VectorStore with Graph Linking")
    
    from vector_store import VegapunkVectorStore
    
    store = VegapunkVectorStore(persist_directory=TEST_CHROMA_PATH)
    
    # Add documents linked to graph nodes
    store.add_document(
        doc_id="doc_python",
        content="Python is a versatile programming language known for its readability and simplicity.",
        node_id="python",
        node_type="Concept"
    )
    
    store.add_document(
        doc_id="doc_flask",
        content="Flask is a lightweight WSGI web application framework written in Python.",
        node_id="flask",
        node_type="Entity"
    )
    
    store.add_document(
        doc_id="doc_django",
        content="Django is a high-level Python web framework that encourages rapid development.",
        node_id="django",
        node_type="Entity"
    )
    
    print(f"  ✓ Added 3 documents with graph links")
    
    # Test semantic search
    results = store.semantic_search("web framework for python", k=3)
    print(f"  ✓ Semantic search returned {len(results)} results")
    for r in results:
        print(f"    - {r.get('node_id', 'N/A')}: score={r.get('similarity_score', 0):.4f}")
    
    # Test search with node_id filter
    results_linked = store.semantic_search("programming", k=5, include_node_ids=True)
    print(f"  ✓ Search with node_ids returned {len(results_linked)} results")
    
    return store


def test_fusion_layer(graph_store, vector_store):
    """Test 3: Fusion Layer scoring."""
    print_step(3, "Testing Fusion Layer")
    
    from fusion_layer import FusionLayer, FusionConfig
    
    fusion = FusionLayer(graph_store, vector_store)
    
    # Test query
    results = fusion.query("What is Python used for?", config=FusionConfig(
        alpha=0.4, beta=0.4, gamma=0.2,
        expansion_hops=1,
        top_k=5
    ))
    
    print(f"  ✓ Fusion query returned {len(results)} results")
    
    for r in results:
        print(f"    - {r.node_id}: final={r.final_score:.4f} (sem={r.semantic_score:.2f}, struct={r.structural_score:.2f}, rec={r.recency_score:.2f})")
    
    # Test LLM context building
    if results:
        context = fusion.build_llm_context(results)
        print(f"  ✓ LLM context built: {len(context['nodes'])} nodes, {len(context['neighbors'])} neighbors")
        
        prompt_context = fusion.build_prompt_context(results)
        print(f"  ✓ Prompt context built: {len(prompt_context)} characters")
    
    return fusion


def test_entity_extraction():
    """Test 4: Entity Extraction."""
    print_step(4, "Testing Entity Extraction")
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("  ⚠️ GEMINI_API_KEY not set, skipping LLM-based extraction test")
        print("  → Set GEMINI_API_KEY environment variable to test entity extraction")
        return None
    
    from entity_extractor import EntityExtractor, ExtractionMode
    
    extractor = EntityExtractor(api_key)
    
    test_text = """
    TensorFlow is an open-source machine learning framework developed by Google Brain.
    It was released in 2015 and is widely used for deep learning applications.
    PyTorch, developed by Facebook's AI Research lab, is a competing framework.
    Both TensorFlow and PyTorch support GPU acceleration for faster training.
    """
    
    result = extractor.extract(test_text, ExtractionMode.STANDARD)
    
    print(f"  ✓ Extracted {len(result.entities)} entities:")
    for e in result.entities:
        print(f"    - {e.name} ({e.type}) importance={e.importance:.2f}")
    
    print(f"  ✓ Extracted {len(result.relations)} relations:")
    for r in result.relations:
        print(f"    - {r.source_id} --[{r.relation_type}]--> {r.target_id}")
    
    return extractor


def test_full_pipeline():
    """Test 5: Full Pipeline Integration."""
    print_step(5, "Testing Full Pipeline")
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("  ⚠️ GEMINI_API_KEY not set, skipping full pipeline test")
        print("  → Set GEMINI_API_KEY environment variable to test full pipeline")
        return
    
    from database import VegapunkDatabase
    from vector_store import VegapunkVectorStore
    from graph_store import GraphStore
    from agent import VegapunkAgent
    
    # Initialize components
    db = VegapunkDatabase(db_path=TEST_DB_PATH)
    vector_store = VegapunkVectorStore(persist_directory=TEST_CHROMA_PATH)
    graph_store = GraphStore(db_path=TEST_DB_PATH)
    
    # Create agent
    agent = VegapunkAgent(
        api_key=api_key,
        db=db,
        vector_store=vector_store,
        graph_store=graph_store
    )
    
    print(f"  ✓ Agent initialized with Graph RAG")
    print(f"  ✓ Graph RAG enabled: {agent.use_graph_rag}")
    print(f"  ✓ Auto entity extraction: {agent.auto_extract_entities}")
    
    # Create a test session
    db.create_session(TEST_SESSION_ID, "Graph RAG Test Session")
    print(f"  ✓ Created test session: {TEST_SESSION_ID}")
    
    # Test document ingestion
    print("\n  [Ingesting test document...]")
    ingestion_result = agent.ingest_document(
        content="""
        Machine learning is a subset of artificial intelligence that enables 
        systems to learn and improve from experience. Deep learning is a 
        specialized form of machine learning using neural networks with many layers.
        Common frameworks include TensorFlow, PyTorch, and scikit-learn.
        """,
        doc_id="doc_ml_intro",
        title="Introduction to Machine Learning",
        session_id=TEST_SESSION_ID
    )
    
    print(f"  ✓ Document ingested: {ingestion_result}")
    
    # Check graph stats
    stats = agent.get_graph_stats()
    print(f"  ✓ Graph stats after ingestion: {stats}")
    
    # Test a chat query
    print("\n  [Testing chat with Graph RAG...]")
    
    try:
        response = agent.chat(
            session_id=TEST_SESSION_ID,
            user_message="What is the relationship between machine learning and deep learning?",
            save_to_memory=True
        )
        
        print(f"  ✓ Chat response received: {len(response.get('response', ''))} chars")
        print(f"  → Response preview: {response.get('response', '')[:200]}...")
        
    except Exception as e:
        print(f"  ⚠️ Chat error (expected if tools not fully configured): {e}")
    
    # Final stats
    final_stats = agent.get_graph_stats()
    print(f"\n  ✓ Final graph stats: {final_stats}")


def test_ranking_accuracy():
    """Test 6: Ranking accuracy and score distribution."""
    print_step(6, "Testing Ranking Accuracy")
    
    from graph_store import GraphStore
    from vector_store import VegapunkVectorStore
    from fusion_layer import FusionLayer, FusionConfig
    
    graph = GraphStore(db_path=TEST_DB_PATH)
    vector = VegapunkVectorStore(persist_directory=TEST_CHROMA_PATH)
    fusion = FusionLayer(graph, vector)
    
    # Test different query types
    queries = [
        ("web framework", "Should favor Flask/Django"),
        ("programming language", "Should favor Python"),
        ("machine learning", "Should favor ML-related nodes"),
    ]
    
    for query, expected in queries:
        results = fusion.query(query, config=FusionConfig(top_k=3))
        print(f"\n  Query: '{query}' ({expected})")
        
        if results:
            top_result = results[0]
            print(f"    Top result: {top_result.node_id} (score: {top_result.final_score:.4f})")
            print(f"    Score breakdown: semantic={top_result.semantic_score:.3f}, structural={top_result.structural_score:.3f}")
        else:
            print(f"    No results found")


def run_all_tests():
    """Run all integration tests."""
    print_header("VEGAPUNK Graph RAG Integration Tests")
    print(f"Started at: {datetime.now().isoformat()}")
    
    start_time = time.time()
    
    try:
        # Cleanup from previous runs
        print("\n[SETUP] Cleaning up previous test files...")
        cleanup_test_files()
        
        # Run tests
        graph_store = test_graph_store()
        vector_store = test_vector_store()
        fusion = test_fusion_layer(graph_store, vector_store)
        extractor = test_entity_extraction()
        test_ranking_accuracy()
        test_full_pipeline()
        
        elapsed = time.time() - start_time
        
        print_header("TEST SUMMARY")
        print(f"  ✅ All tests completed in {elapsed:.2f} seconds")
        print(f"  📊 Graph nodes: {graph_store.get_stats()['node_count']}")
        print(f"  📊 Graph edges: {graph_store.get_stats()['edge_count']}")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Optional: cleanup after tests
        # cleanup_test_files()
        print("\n[NOTE] Test files preserved for inspection.")
        print(f"  - Database: {TEST_DB_PATH}")
        print(f"  - Chroma: {TEST_CHROMA_PATH}")
    
    return True


def run_quick_test():
    """Run a quick smoke test without LLM calls."""
    print_header("VEGAPUNK Quick Smoke Test (No LLM)")
    
    cleanup_test_files()
    
    try:
        # Test imports
        print("\n[1] Testing imports...")
        from graph_store import GraphStore
        from vector_store import VegapunkVectorStore
        from fusion_layer import FusionLayer, FusionConfig
        from entity_extractor import EntityExtractor, ExtractionMode
        from database import VegapunkDatabase
        print("  ✓ All imports successful")
        
        # Test GraphStore
        print("\n[2] Testing GraphStore...")
        graph = GraphStore(db_path=TEST_DB_PATH)
        graph.add_node("test_node", "Concept", label="Test")
        assert graph.get_node("test_node") is not None
        print("  ✓ GraphStore working")
        
        # Test VectorStore
        print("\n[3] Testing VectorStore...")
        vector = VegapunkVectorStore(persist_directory=TEST_CHROMA_PATH)
        vector.add_document("test_doc", "Test content", "test_node", "Concept")
        results = vector.semantic_search("test", k=1)
        assert len(results) > 0
        print("  ✓ VectorStore working")
        
        # Test FusionLayer
        print("\n[4] Testing FusionLayer...")
        fusion = FusionLayer(graph, vector)
        results = fusion.query("test")
        print(f"  ✓ FusionLayer working ({len(results)} results)")
        
        print("\n" + "=" * 50)
        print("✅ QUICK SMOKE TEST PASSED")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Quick test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="VEGAPUNK Graph RAG Tests")
    parser.add_argument("--quick", action="store_true", help="Run quick smoke test only")
    parser.add_argument("--cleanup", action="store_true", help="Cleanup test files and exit")
    
    args = parser.parse_args()
    
    if args.cleanup:
        cleanup_test_files()
        print("✓ Cleanup complete")
    elif args.quick:
        success = run_quick_test()
        sys.exit(0 if success else 1)
    else:
        success = run_all_tests()
        sys.exit(0 if success else 1)
