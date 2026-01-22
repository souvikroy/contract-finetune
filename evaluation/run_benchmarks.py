"""Run benchmark tests."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.benchmark import BenchmarkSuite
from src.evaluation.test_cases import get_test_cases
from src.llm.chain import RAGChain
from src.rag.retriever import HybridRetriever
from src.rag.vector_store import VectorStore
from src.data.contract_parser import ContractParser
from src.data.chunking import LegalDocumentChunker
from src.data.preprocessing import ContractPreprocessor


def initialize_system():
    """Initialize RAG system."""
    print("Initializing RAG system...")
    
    # Parse contract
    parser = ContractParser()
    chunks = parser.extract_chunks()
    text_chunks = parser.get_all_text_chunks(min_length=50)
    
    # Preprocess
    preprocessor = ContractPreprocessor()
    processed_chunks = preprocessor.preprocess_chunks(text_chunks[:100])  # Limit for testing
    
    # Chunk
    chunker = LegalDocumentChunker(chunk_size=1000, chunk_overlap=200)
    chunked_docs = chunker.chunk_text(processed_chunks)
    
    # Initialize vector store
    vector_store = VectorStore()
    
    # Check if vector store is empty
    info = vector_store.get_collection_info()
    if info.get('document_count', 0) == 0:
        print("Adding documents to vector store...")
        vector_store.add_documents(chunked_docs)
    else:
        print(f"Vector store already has {info['document_count']} documents")
    
    # Initialize RAG chain
    retriever = HybridRetriever(vector_store)
    rag_chain = RAGChain(retriever)
    
    return rag_chain


def main():
    """Run benchmarks."""
    print("="*60)
    print("Legal Contract Chatbot - Benchmark Suite")
    print("="*60)
    
    # Initialize system
    rag_chain = initialize_system()
    
    # Run benchmarks
    benchmark_suite = BenchmarkSuite(rag_chain)
    
    print("\nRunning accuracy benchmarks...")
    accuracy_results = benchmark_suite.run_benchmark()
    
    print("\nRunning latency benchmarks...")
    latency_results = benchmark_suite.run_latency_benchmark(num_queries=5)
    
    # Print results
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    
    print("\nAccuracy Metrics:")
    metrics = accuracy_results['metrics']
    print(f"  Accuracy: {metrics['accuracy']:.2%}")
    print(f"  Average Score: {metrics['avg_score']:.2f}")
    print(f"  Pass Rate: {metrics['pass_rate']:.2%}")
    print(f"  Tests Passed: {metrics['passed_tests']}/{metrics['total_tests']}")
    
    print("\nLatency Metrics:")
    print(f"  Average Latency: {latency_results['avg_latency_ms']:.2f}ms")
    print(f"  Min Latency: {latency_results['min_latency_ms']:.2f}ms")
    print(f"  Max Latency: {latency_results['max_latency_ms']:.2f}ms")
    print(f"  P95 Latency: {latency_results['p95_latency_ms']:.2f}ms")
    
    print("\nDetailed Results:")
    for result in accuracy_results['results']:
        status = "✅ PASS" if result.get('passed') else "❌ FAIL"
        print(f"  {result['test_id']}: {status} (Score: {result.get('overall_score', 0):.2f}, "
              f"Latency: {result.get('latency_ms', 0):.2f}ms)")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
