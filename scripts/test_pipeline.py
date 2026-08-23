#!/usr/bin/env python3
"""
Test script to validate the RAG pipeline components.
Run this after building the vector store to ensure everything works.
"""

import sys
from pathlib import Path
import json
import gc
import shutil
import tempfile

# Add src directory to Python path BEFORE importing rag
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag.data_processing import Document
from rag.vector_store import CyberRAGStore


def test_imports():
    """Test all required imports"""
    print("\n[TEST 1/5] Testing imports...")
    try:
        import torch
        import transformers
        import chromadb
        from sentence_transformers import SentenceTransformer
        print("  ✓ Core ML libraries imported")

        from rag.data_processing import Document, AttackProcessor
        from rag.vector_store import CyberRAGStore
        from rag.rag_pipeline import CyberRAGPipeline
        print("  ✓ Custom RAG modules imported")

        print("✓ All imports successful\n")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}\n")
        return False


def test_data_processing():
    """Test data processing functions"""
    print("[TEST 2/5] Testing data processing...")
    try:
        from rag.data_processing import Document

        # Create test document
        doc = Document(
            id="test-001",
            content="Test cybersecurity document about T1059",
            metadata={"name": "Test", "type": "technique"},
            source_type="attack"
        )

        assert doc.id == "test-001"
        assert "T1059" in doc.content
        print("  ✓ Document creation works")
        print("✓ Data processing test passed\n")
        return True
    except Exception as e:
        print(f"✗ Data processing failed: {e}\n")
        return False


def test_vector_store():
    """Test vector store operations"""
    print("[TEST 3/5] Testing vector store...")

    try:
        from rag.vector_store import CyberRAGStore
        from rag.data_processing import Document
        import tempfile
        import shutil
        import gc

        # Create temporary directory
        tmpdir = tempfile.mkdtemp()

        try:
            store = CyberRAGStore(
                persist_dir=Path(tmpdir),
                collection_name="test_collection"
            )

            # Create test documents
            documents = [
                Document(
                    id="test-1",
                    content="SQL injection is a web application security vulnerability.",
                    metadata={"severity": "high"},
                    source_type="cwe"
                ),
                Document(
                    id="test-2",
                    content="Cross-site scripting allows attackers to inject malicious scripts.",
                    metadata={"severity": "medium"},
                    source_type="cwe"
                )
            ]

            # Add documents
            store.add_documents(documents)
            print("  ✓ Documents added to vector store")

            # Test search
            results = store.search(
                "web application security vulnerability",
                n_results=2
            )

            assert len(results["documents"][0]) > 0
            print("  ✓ Search works")

            # Test statistics
            stats = store.get_stats()

            assert stats["total_documents"] == 2
            print("  ✓ Statistics retrieval works")

            # Explicitly release Chroma
            del store
            gc.collect()

        finally:
            # Clean up temporary directory
            try:
                shutil.rmtree(tmpdir)
            except PermissionError:
                print("  ⚠ Temporary Chroma directory could not be removed on Windows")

        print("✓ Vector store test passed\n")
        return True

    except Exception as e:
        print(f"✗ Vector store test failed: {e}")
        return False

def test_vector_store_persistence():
    """Test if vector store was built"""
    print("[TEST 4/5] Testing vector store persistence...")
    try:
        project_root = Path(__file__).parent.parent
        embeddings_dir = project_root / "data" / "embeddings"

        if not embeddings_dir.exists():
            print(f"  ⚠ Vector store not found at {embeddings_dir}")
            print("  → Run 'python scripts/build_rag.py' first")
            return False

        from rag.vector_store import CyberRAGStore

        # Load existing store
        store = CyberRAGStore(
            persist_dir=embeddings_dir,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Get stats
        stats = store.get_stats()
        print(f"  ✓ Loaded vector store with {stats['total_documents']} documents")
        print(f"  ✓ Source distribution: {json.dumps(stats['source_distribution'], indent=4)}")

        # Test search
        results = store.search("What is ATT&CK technique T1059?", n_results=3)
        print(f"  ✓ Search returned {len(results['ids'][0])} results")

        print("✓ Vector store persistence test passed\n")
        return True
    except Exception as e:
        print(f"✗ Vector store persistence test failed: {e}\n")
        return False


def test_rag_pipeline_mock():
    """Test RAG pipeline with mock data"""
    print("[TEST 5/5] Testing RAG pipeline (mock mode)...")

    try:
        from rag.rag_pipeline import CyberRAGPipeline
        from rag.vector_store import CyberRAGStore
        from rag.data_processing import Document
        import tempfile
        import gc
        import shutil

        # Create temporary directory
        tmpdir = tempfile.mkdtemp()

        try:
            # Create vector store
            store = CyberRAGStore(
                persist_dir=Path(tmpdir),
                collection_name="test_rag_collection"
            )

            # Create mock cybersecurity document
            documents = [
                Document(
                    id="test-rag-1",
                    content=(
                        "MITRE ATT&CK technique T1059 represents "
                        "Command and Scripting Interpreter. "
                        "Attackers may use command shells or scripting "
                        "languages to execute commands."
                    ),
                    metadata={
                        "technique": "T1059",
                        "name": "Command and Scripting Interpreter"
                    },
                    source_type="attack"
                )
            ]

            # Add document to vector store
            store.add_documents(documents)

            # Test retrieval
            results = store.search(
                "What is MITRE ATT&CK technique T1059?",
                n_results=1
            )

            assert len(results["documents"][0]) > 0
            assert "T1059" in results["documents"][0][0]

            print("  ✓ Retrieval pipeline works")

            # Release Chroma resources before cleanup
            del store
            gc.collect()

        finally:
            # Clean up temporary directory
            try:
                shutil.rmtree(tmpdir)
            except PermissionError:
                print(
                    "  ⚠ Temporary Chroma directory "
                    "could not be removed on Windows"
                )

        print("✓ RAG pipeline test passed (mock mode)\n")
        return True

    except Exception as e:
        print(f"✗ RAG pipeline test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 70)
    print("CYBERRAG PIPELINE TEST SUITE")
    print("=" * 70)

    tests = [
        ("Imports", test_imports),
        ("Data Processing", test_data_processing),
        ("Vector Store", test_vector_store),
        ("Vector Store Persistence", test_vector_store_persistence),
        ("RAG Pipeline (Mock)", test_rag_pipeline_mock),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"✗ Test '{name}' crashed: {e}\n")
            results.append((name, False))

    # Print summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")

    print(f"\nTotal: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("\n🎉 All tests passed! Pipeline is ready.")
        return 0
    else:
        print(f"\n⚠ {total_count - passed_count} test(s) failed. Check logs above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
