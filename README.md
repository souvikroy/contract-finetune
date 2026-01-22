# Legal Contract Chatbot

A production-ready Streamlit chatbot for legal contract analysis using Claude API, LangChain, and RAG (Retrieval-Augmented Generation).

## Features

- **RAG System**: Semantic search with hybrid retrieval (semantic + keyword)
- **Claude API Integration**: Powered by Anthropic's Claude via LangChain
- **Streamlit Frontend**: Interactive chat interface with document viewer
- **Caching**: Multi-level caching (memory + optional Redis) for performance
- **Monitoring**: Comprehensive logging and metrics collection
- **Evaluation**: Benchmark test suite for accuracy and latency
- **Cloud Deployment**: Docker, Kubernetes, and AWS CloudFormation configs

## Architecture

```
User → Streamlit Frontend → LangChain → RAG System → Vector DB (ChromaDB)
                                      ↓
                                  Claude API
```

## Setup

### Prerequisites

- Python 3.11+
- Anthropic API key
- (Optional) Redis for distributed caching

### Installation

1. Clone the repository:
```bash
cd contract
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment:
```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

4. Initialize vector store:
```python
from src.data.contract_parser import ContractParser
from src.data.chunking import LegalDocumentChunker
from src.data.preprocessing import ContractPreprocessor
from src.rag.vector_store import VectorStore

# Parse contract
parser = ContractParser()
chunks = parser.extract_chunks()
text_chunks = parser.get_all_text_chunks(min_length=50)

# Preprocess
preprocessor = ContractPreprocessor()
processed_chunks = preprocessor.preprocess_chunks(text_chunks)

# Chunk
chunker = LegalDocumentChunker(chunk_size=1000, chunk_overlap=200)
chunked_docs = chunker.chunk_text(processed_chunks)

# Add to vector store
vector_store = VectorStore()
vector_store.add_documents(chunked_docs)
```

5. Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

## Usage

### Chat Interface

1. Start the app: `streamlit run streamlit_app.py`
2. Access at `http://localhost:8501`
3. Ask questions about the contract document
4. View relevant document sections and sources

### Example Queries

- "What is the defect liability period?"
- "What are the payment terms?"
- "What happens if the contractor defaults?"
- "What are the performance security requirements?"

## Evaluation

Run benchmark tests:

```bash
python evaluation/run_benchmarks.py
```

This will:
- Test accuracy on legal document questions
- Measure latency and throughput
- Generate performance reports

## Deployment

### Docker

```bash
docker build -t contract-chatbot -f docker/Dockerfile .
docker run -p 8501:8501 --env-file .env contract-chatbot
```

### Docker Compose

```bash
docker-compose -f docker/docker-compose.yml up -d
```

### Kubernetes

```bash
kubectl apply -f deployment/kubernetes/deployment.yaml
```

### AWS (CloudFormation)

```bash
aws cloudformation create-stack \
  --stack-name contract-chatbot \
  --template-body file://deployment/aws/cloudformation.yaml \
  --parameters ParameterKey=AnthropicAPIKey,ParameterValue=YOUR_API_KEY
```

## Configuration

Key configuration options in `.env`:

- `ANTHROPIC_API_KEY`: Your Claude API key
- `CLAUDE_MODEL`: Model to use (default: claude-3-5-sonnet-20241022)
- `VECTOR_DB_TYPE`: Vector database type (chromadb/faiss)
- `CACHE_ENABLED`: Enable caching (true/false)
- `REDIS_ENABLED`: Use Redis for caching (true/false)
- `ALERT_THRESHOLD_LATENCY_MS`: Latency alert threshold (default: 200ms)

## Project Structure

```
contract/
├── streamlit_app.py          # Main Streamlit app
├── src/
│   ├── data/                 # Data processing
│   ├── rag/                  # RAG system
│   ├── llm/                  # LLM integration
│   ├── optimization/         # Caching, batching
│   ├── monitoring/           # Logging, metrics
│   ├── evaluation/           # Benchmark tests
│   └── ui/                   # UI components
├── docker/                   # Docker configs
├── deployment/               # Cloud deployment
└── evaluation/               # Benchmark scripts
```

## Performance Targets

- **Latency**: <200ms average response time
- **Accuracy**: >90% on legal benchmark tasks
- **Throughput**: Support 1-100 concurrent instances
- **Cost**: Optimized API usage through caching

## Monitoring

Access metrics dashboard:
- In Streamlit UI: Click "View Metrics" in sidebar
- Metrics include: latency, error rate, request count, function-level stats

## License

MIT License

## Contributing

Contributions welcome! Please open an issue or submit a PR.
