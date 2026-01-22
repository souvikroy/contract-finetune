# Legal Contract Management

A production-ready Streamlit chatbot for legal contract analysis using LM Studio (Qwen model), LangChain, and RAG (Retrieval-Augmented Generation). This system enables intelligent Q&A over legal documents with domain-specific fine-tuning capabilities.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Detailed Setup](#detailed-setup)
- [Usage Guide](#usage-guide)
- [Fine-Tuning](#fine-tuning)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [Performance Tuning](#performance-tuning)
- [API Reference](#api-reference)

## Features

### Core Capabilities

- **RAG System**: Advanced semantic search with hybrid retrieval combining:
  - Vector similarity search (semantic understanding)
  - Keyword-based search (exact term matching)
  - Context-aware document ranking
  
- **LM Studio Integration**: 
  - OpenAI-compatible API integration
  - Support for Qwen models (qwen/qwen3-vl-4b)
  - Streaming responses for real-time interaction
  - Configurable temperature and token limits

- **PEFT Fine-Tuning**: 
  - LoRA (Low-Rank Adaptation) for efficient training
  - QLoRA (4-bit quantization) for memory-constrained environments
  - Domain-specific adaptation on legal contract data
  - Adapter merging and deployment support

- **Streamlit Frontend**: 
  - Interactive chat interface with message history
  - Document viewer with source citations
  - Query suggestions and examples
  - Metrics dashboard integration
  - Export chat history functionality

- **Caching**: 
  - Multi-level caching system:
    - In-memory cache for fast repeated queries
    - Optional Redis integration for distributed caching
    - Configurable TTL and cache invalidation

- **Monitoring**: 
  - Comprehensive logging with structured JSON logs
  - Real-time metrics collection (latency, error rates, throughput)
  - Performance dashboards
  - Alert thresholds for latency and errors

- **Evaluation**: 
  - Benchmark test suite with legal domain test cases
  - Accuracy metrics (BLEU, ROUGE, exact match)
  - Latency and throughput measurements
  - Performance reports and analysis

- **Cloud Deployment**: 
  - Docker containerization with multi-stage builds
  - Kubernetes manifests with HPA (Horizontal Pod Autoscaler)
  - AWS CloudFormation templates for ECS deployment
  - Health checks and readiness probes

## Architecture

### System Overview

```
┌─────────────┐
│   User      │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│              Streamlit Frontend                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ Chat UI      │  │ Doc Viewer   │  │ Metrics      │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│              RAG Chain (LangChain)                      │
│  ┌──────────────┐         ┌──────────────┐            │
│  │ Retriever    │─────────▶│ LLM Client  │            │
│  │ (Hybrid)     │         │ (LM Studio) │            │
│  └──────────────┘         └──────────────┘            │
│         │                                                │
│         ▼                                                │
│  ┌──────────────┐                                       │
│  │ Vector Store │                                       │
│  │ (ChromaDB)   │                                       │
│  └──────────────┘                                       │
└─────────────────────────────────────────────────────────┘
       │                           │
       ▼                           ▼
┌──────────────┐         ┌──────────────────┐
│ Vector DB    │         │ LM Studio API    │
│ (Embeddings) │         │ (Qwen Model)     │
└──────────────┘         └──────────────────┘
```

### Data Flow

1. **Query Processing**: User submits question → Streamlit UI
2. **Retrieval**: RAG system searches vector database for relevant contract sections
3. **Context Assembly**: Retrieved documents formatted with metadata (page, section, clause)
4. **Generation**: LM Studio API generates answer using context + question
5. **Response**: Formatted answer with citations returned to user
6. **Caching**: Response cached for future similar queries

### Component Details

- **Vector Store**: ChromaDB for persistent embedding storage
- **Embeddings**: Sentence transformers (all-MiniLM-L6-v2)
- **Retriever**: Hybrid approach combining semantic and keyword search
- **LLM**: Qwen model via LM Studio (local or remote)
- **Cache**: Redis-optional in-memory cache layer

## Quick Start

Get up and running in 5 minutes:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start LM Studio with Qwen model
# (Ensure LM Studio is running at http://192.168.0.4:1601)

# 3. Configure environment
cp .env.example .env
# Edit .env with your LM Studio settings

# 4. Initialize vector store (one-time setup)
python -c "
from src.data.contract_parser import ContractParser
from src.data.chunking import LegalDocumentChunker
from src.data.preprocessing import ContractPreprocessor
from src.rag.vector_store import VectorStore

parser = ContractParser()
chunks = parser.extract_chunks()
text_chunks = parser.get_all_text_chunks(min_length=50)

preprocessor = ContractPreprocessor()
processed_chunks = preprocessor.preprocess_chunks(text_chunks)

chunker = LegalDocumentChunker(chunk_size=1000, chunk_overlap=200)
chunked_docs = chunker.chunk_text(processed_chunks)

vector_store = VectorStore()
vector_store.add_documents(chunked_docs)
print('Vector store initialized!')
"

# 5. Run the app
streamlit run streamlit_app.py
```

## Detailed Setup

### Prerequisites

#### Required
- **Python 3.11+**: Tested with Python 3.11 and 3.12
- **LM Studio**: Download from [lmstudio.ai](https://lmstudio.ai)
  - Install Qwen model (qwen/qwen3-vl-4b or compatible)
  - Start local server (default port: 1234, configure to 1601)
  - Enable OpenAI-compatible API

#### Optional
- **Redis**: For distributed caching (recommended for production)
  ```bash
  # Install Redis
  brew install redis  # macOS
  sudo apt-get install redis-server  # Ubuntu
  
  # Start Redis
  redis-server
  ```

- **GPU**: For fine-tuning (CUDA-compatible GPU with 8GB+ VRAM recommended)
  - NVIDIA GPU with CUDA 11.8+
  - cuDNN libraries
  - PyTorch with CUDA support

- **Docker**: For containerized deployment
  ```bash
  docker --version  # Verify installation
  ```

### Installation Steps

#### 1. Clone Repository

```bash
git clone <repository-url>
cd contract
```

#### 2. Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Using conda
conda create -n contract-chatbot python=3.11
conda activate contract-chatbot
```

#### 3. Install Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt

# Or install in stages (for troubleshooting)
pip install streamlit langchain openai
pip install chromadb sentence-transformers
pip install pydantic pydantic-settings python-dotenv
pip install torch transformers peft  # For fine-tuning
```

#### 4. Configure LM Studio

1. **Download and Install LM Studio**
   - Visit [lmstudio.ai](https://lmstudio.ai)
   - Download for your OS
   - Install and launch

2. **Load Qwen Model**
   - Open LM Studio
   - Go to "Search" tab
   - Search for "qwen" or "qwen3-vl-4b"
   - Download the model

3. **Start Local Server**
   - Go to "Local Server" tab
   - Select your Qwen model
   - Set port to 1601 (or update config)
   - Click "Start Server"
   - Verify: `curl http://192.168.0.4:1601/v1/models`

#### 5. Environment Configuration

Create `.env` file:

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```bash
# LM Studio Configuration
LM_STUDIO_API_URL=http://192.168.0.4:1601
LM_STUDIO_MODEL=qwen/qwen3-vl-4b

# Vector Database
VECTOR_DB_TYPE=chromadb
CHROMA_PERSIST_DIRECTORY=./chroma_db

# Caching
CACHE_ENABLED=true
REDIS_ENABLED=false
REDIS_HOST=localhost
REDIS_PORT=6379

# Performance
MAX_TOKENS=4096
TEMPERATURE=0.1
BATCH_SIZE=10
REQUEST_TIMEOUT=30

# Monitoring
LOG_LEVEL=INFO
METRICS_ENABLED=true
ALERT_THRESHOLD_LATENCY_MS=200

# Contract Data
CONTRACT_JSON_PATH=./RFP_parsed.json
```

#### 6. Initialize Vector Store

Create initialization script `init_vector_store.py`:

```python
"""Initialize vector store with contract data."""
from src.data.contract_parser import ContractParser
from src.data.chunking import LegalDocumentChunker
from src.data.preprocessing import ContractPreprocessor
from src.rag.vector_store import VectorStore
from src.monitoring.logger import logger

def main():
    logger.info("Starting vector store initialization...")
    
    # Parse contract
    parser = ContractParser()
    logger.info("Parsing contract JSON...")
    chunks = parser.extract_chunks()
    text_chunks = parser.get_all_text_chunks(min_length=50)
    logger.info(f"Extracted {len(text_chunks)} text chunks")
    
    # Preprocess
    preprocessor = ContractPreprocessor()
    logger.info("Preprocessing chunks...")
    processed_chunks = preprocessor.preprocess_chunks(text_chunks)
    
    # Chunk documents
    chunker = LegalDocumentChunker(chunk_size=1000, chunk_overlap=200)
    logger.info("Chunking documents...")
    chunked_docs = chunker.chunk_text(processed_chunks)
    logger.info(f"Created {len(chunked_docs)} document chunks")
    
    # Add to vector store
    vector_store = VectorStore()
    logger.info("Adding documents to vector store...")
    vector_store.add_documents(chunked_docs)
    
    # Verify
    info = vector_store.get_collection_info()
    logger.info(f"Vector store initialized: {info}")
    print(f"✅ Vector store initialized with {info.get('document_count', 0)} documents")

if __name__ == "__main__":
    main()
```

Run initialization:

```bash
python init_vector_store.py
```

#### 7. Verify Installation

```bash
# Test LM Studio connection
python -c "
from src.llm.lm_studio_client import LMStudioClient
client = LMStudioClient()
response = client.invoke([{'role': 'user', 'content': 'Hello'}])
print('✅ LM Studio connection successful')
print(f'Response: {response[:50]}...')
"

# Test vector store
python -c "
from src.rag.vector_store import VectorStore
store = VectorStore()
info = store.get_collection_info()
print(f'✅ Vector store: {info.get(\"document_count\", 0)} documents')
"
```

#### 8. Run Application

```bash
streamlit run streamlit_app.py
```

Access at `http://localhost:8501`

## Usage Guide

### Chat Interface

#### Starting the Application

```bash
# Basic start
streamlit run streamlit_app.py

# With custom port
streamlit run streamlit_app.py --server.port 8502

# With custom address
streamlit run streamlit_app.py --server.address 0.0.0.0
```

#### Interface Overview

1. **Main Chat Area**: 
   - Type questions in the input box
   - View conversation history
   - See streaming responses in real-time

2. **Sidebar Features**:
   - **Query Suggestions**: Pre-defined example questions
   - **Clear Chat**: Reset conversation history
   - **Export History**: Download chat as text file
   - **View Metrics**: Access performance dashboard

3. **Document Viewer**:
   - Source citations with page numbers
   - Relevant document sections
   - Clause references and metadata

#### Query Types

The system supports various query types:

**Factual Questions**
- "What is the defect liability period?"
- "What are the payment terms?"
- "What is the contract value?"

**Procedural Questions**
- "What happens if the contractor defaults?"
- "How is payment processed?"
- "What is the dispute resolution process?"

**Comparative Questions**
- "Compare payment terms and security requirements"
- "What are the differences between termination clauses?"

**Extraction Queries**
- "Extract all clauses related to liability"
- "List all payment-related sections"
- "Find all termination conditions"

### Example Queries

#### Basic Information Retrieval

```
Q: What is the defect liability period?
A: Based on Article 5.2, the defect liability period is 2 years from 
   the date of completion of the work. During this period, the contractor 
   is responsible for rectifying any defects...

[Sources: Article 5.2, Page 12]
```

#### Complex Multi-Part Questions

```
Q: What are the payment terms and when are they due?
A: Payment terms are specified in Article 7:
   - 30% advance payment upon contract signing
   - 40% upon completion of 50% of work
   - 30% upon final completion and acceptance
   
   Payments are due within 30 days of invoice submission...

[Sources: Article 7.1-7.3, Pages 15-16]
```

#### Clause Extraction

```
Q: Extract all clauses related to termination
A: Termination clauses are found in Article 12:
   
   12.1 Termination by Employer:
   - Non-performance of obligations
   - Material breach of contract
   - Insolvency or bankruptcy
   
   12.2 Termination by Contractor:
   - Non-payment exceeding 60 days
   - Force majeure events
   - Employer's material breach...

[Sources: Article 12, Pages 22-24]
```

### Advanced Usage

#### Programmatic Access

```python
from src.llm.chain import RAGChain
from src.rag.retriever import HybridRetriever
from src.rag.vector_store import VectorStore

# Initialize RAG chain
vector_store = VectorStore()
retriever = HybridRetriever(vector_store)
rag_chain = RAGChain(retriever)

# Query
result = rag_chain.query(
    question="What are the payment terms?",
    n_context=5,  # Number of context documents
    use_keyword=True  # Enable keyword search
)

print(result['answer'])
print(result['sources'])
```

#### Streaming Responses

```python
# Stream response chunks
for chunk_data in rag_chain.stream_query("What is the liability period?"):
    print(chunk_data['chunk'], end='', flush=True)
    # chunk_data also contains 'answer_so_far' and 'sources'
```

#### Clause Classification

```python
clause_text = "The contractor shall be liable for all defects..."
classification = rag_chain.classify_clause(clause_text)
print(classification['classification'])
```

#### Clause Extraction

```python
extracted = rag_chain.extract_clauses(
    query="payment and compensation",
    n_context=10
)
print(extracted['extracted_clauses'])
```

## Evaluation

### Running Benchmarks

The evaluation suite tests system accuracy, latency, and throughput:

```bash
# Run full benchmark suite
python evaluation/run_benchmarks.py

# Expected output:
# ============================================================
# Legal Contract Chatbot - Benchmark Suite
# ============================================================
# 
# Running accuracy benchmarks...
# Running latency benchmarks...
# 
# BENCHMARK RESULTS
# ============================================================
# 
# Accuracy Metrics:
#   Accuracy: 92.50%
#   Average Score: 0.87
#   Pass Rate: 90.00%
#   Tests Passed: 8/8
# 
# Latency Metrics:
#   Average Latency: 185.23ms
#   Min Latency: 120.45ms
#   Max Latency: 245.67ms
#   P95 Latency: 220.15ms
```

### Benchmark Components

1. **Accuracy Tests**:
   - Question-answer matching
   - Keyword presence validation
   - Answer quality scoring
   - Category-specific accuracy

2. **Latency Tests**:
   - Average response time
   - P95/P99 percentiles
   - Throughput measurements
   - Concurrent request handling

3. **Quality Metrics**:
   - BLEU scores for answer quality
   - ROUGE scores for relevance
   - Exact match accuracy
   - Legal terminology correctness

### Custom Test Cases

Add custom test cases in `src/evaluation/test_cases.py`:

```python
LEGAL_TEST_CASES = [
    {
        'id': 'test_custom_001',
        'question': 'Your custom question here',
        'expected_keywords': ['keyword1', 'keyword2'],
        'category': 'your_category',
        'expected_answer_type': 'specific_period'
    },
    # ... more test cases
]
```

### Performance Analysis

Generate detailed analysis:

```bash
python evaluation/results_analysis.py --input results.json --output report.html
```

## Deployment

### Docker Deployment

#### Build Image

```bash
docker build -t contract-chatbot -f docker/Dockerfile .
```

#### Run Container

```bash
docker run -d \
  --name contract-chatbot \
  -p 8501:8501 \
  --env-file .env \
  -v $(pwd)/chroma_db:/app/chroma_db \
  -v $(pwd)/RFP_parsed.json:/app/RFP_parsed.json \
  contract-chatbot
```

#### Verify Deployment

```bash
# Check container status
docker ps

# View logs
docker logs contract-chatbot

# Test endpoint
curl http://localhost:8501/_stcore/health
```

### Docker Compose Deployment

#### Start Services

```bash
# Start all services
docker-compose -f docker/docker-compose.yml up -d

# Start with Redis
docker-compose -f docker/docker-compose.yml --profile redis up -d

# View logs
docker-compose -f docker/docker-compose.yml logs -f

# Stop services
docker-compose -f docker/docker-compose.yml down
```

#### Environment Variables

Update `docker/docker-compose.yml` or use `.env` file:

```yaml
environment:
  - LM_STUDIO_API_URL=${LM_STUDIO_API_URL:-http://192.168.0.4:1601}
  - LM_STUDIO_MODEL=${LM_STUDIO_MODEL:-qwen/qwen3-vl-4b}
  - VECTOR_DB_TYPE=${VECTOR_DB_TYPE:-chromadb}
  - CACHE_ENABLED=${CACHE_ENABLED:-true}
```

### Kubernetes Deployment

#### Prerequisites

- Kubernetes cluster (v1.20+)
- kubectl configured
- Docker image pushed to registry

#### Deploy Application

```bash
# Create namespace
kubectl create namespace contract-chatbot

# Create secret for LM Studio config (if needed)
kubectl create secret generic lm-studio-config \
  --from-literal=api-url=http://192.168.0.4:1601 \
  --from-literal=model=qwen/qwen3-vl-4b \
  -n contract-chatbot

# Deploy application
kubectl apply -f deployment/kubernetes/deployment.yaml

# Check deployment status
kubectl get pods -n contract-chatbot

# View logs
kubectl logs -f deployment/contract-chatbot -n contract-chatbot
```

#### Scaling

```bash
# Scale manually
kubectl scale deployment contract-chatbot --replicas=5 -n contract-chatbot

# HPA automatically scales based on CPU/memory
kubectl get hpa -n contract-chatbot
```

### AWS CloudFormation Deployment

#### Prerequisites

- AWS CLI configured
- ECR repository for Docker image
- VPC and networking setup

#### Deploy Stack

```bash
# Package and upload template
aws cloudformation package \
  --template-file deployment/aws/cloudformation.yaml \
  --s3-bucket your-bucket \
  --output-template-file packaged-template.yaml

# Create stack
aws cloudformation create-stack \
  --stack-name contract-chatbot \
  --template-body file://packaged-template.yaml \
  --parameters \
    ParameterKey=LMStudioAPIUrl,ParameterValue=http://192.168.0.4:1601 \
    ParameterKey=LMStudioModel,ParameterValue=qwen/qwen3-vl-4b \
    ParameterKey=InstanceType,ParameterValue=t3.medium \
  --capabilities CAPABILITY_IAM

# Monitor stack creation
aws cloudformation describe-stacks --stack-name contract-chatbot
```

#### Update Stack

```bash
aws cloudformation update-stack \
  --stack-name contract-chatbot \
  --template-body file://packaged-template.yaml \
  --parameters ParameterKey=LMStudioAPIUrl,ParameterValue=http://new-url:1601
```

### Production Considerations

1. **High Availability**: Deploy multiple replicas across availability zones
2. **Load Balancing**: Use ALB/NLB for traffic distribution
3. **Monitoring**: Set up CloudWatch/Prometheus monitoring
4. **Logging**: Centralized logging with ELK or CloudWatch Logs
5. **Security**: Use secrets management (AWS Secrets Manager, Vault)
6. **Backup**: Regular backups of vector database
7. **Scaling**: Configure HPA for automatic scaling

## Configuration

### Environment Variables

All configuration is done via `.env` file. Create from template:

```bash
cp .env.example .env
```

#### LM Studio Configuration

```bash
# LM Studio API endpoint
LM_STUDIO_API_URL=http://192.168.0.4:1601

# Model name in LM Studio
LM_STUDIO_MODEL=qwen/qwen3-vl-4b

# Model parameters
MAX_TOKENS=4096          # Maximum tokens in response
TEMPERATURE=0.1          # Sampling temperature (0.0-1.0)
```

#### Vector Database Configuration

```bash
# Database type: chromadb or faiss
VECTOR_DB_TYPE=chromadb

# ChromaDB persistence directory
CHROMA_PERSIST_DIRECTORY=./chroma_db

# Embedding model
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

#### Caching Configuration

```bash
# Enable/disable caching
CACHE_ENABLED=true

# Redis configuration (optional)
REDIS_ENABLED=false
REDIS_HOST=localhost
REDIS_PORT=6379
```

#### Performance Configuration

```bash
# Batch processing
BATCH_SIZE=10

# Request timeout (seconds)
REQUEST_TIMEOUT=30

# Maximum retries for failed requests
MAX_RETRIES=3
```

#### Monitoring Configuration

```bash
# Logging level: DEBUG, INFO, WARNING, ERROR
LOG_LEVEL=INFO

# Enable metrics collection
METRICS_ENABLED=true

# Latency alert threshold (milliseconds)
ALERT_THRESHOLD_LATENCY_MS=200
```

#### Fine-Tuning Configuration

```bash
# Enable fine-tuned model
FINETUNING_ENABLED=false

# Path to fine-tuned adapter weights
FINETUNING_MODEL_PATH=

# Base model for fine-tuning
FINETUNING_BASE_MODEL=Qwen/Qwen2.5-7B-Instruct

# LoRA parameters
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05

# Use merged adapters (faster inference)
USE_MERGED_MODEL=false
```

#### Streamlit Configuration

```bash
# Server port
STREAMLIT_SERVER_PORT=8501

# Server address (0.0.0.0 for all interfaces)
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

#### Contract Data Configuration

```bash
# Path to contract JSON file
CONTRACT_JSON_PATH=./RFP_parsed.json
```

### Configuration Validation

Validate your configuration:

```python
from src.config import settings

# Check critical settings
assert settings.lm_studio_api_url, "LM_STUDIO_API_URL not set"
assert settings.lm_studio_model, "LM_STUDIO_MODEL not set"
print("✅ Configuration valid")
```

## Fine-Tuning

The project includes comprehensive PEFT (Parameter-Efficient Fine-Tuning) support using LoRA/QLoRA to fine-tune the Qwen model on legal contract data. Fine-tuning improves domain-specific understanding and answer quality.

### Why Fine-Tune?

- **Domain Adaptation**: Better understanding of legal terminology
- **Improved Accuracy**: 10-30% improvement on legal domain tasks
- **Citation Quality**: More accurate section references
- **Response Structure**: More professional and structured answers

### Quick Start

#### 1. Generate Training Data

```bash
python finetuning/generate_training_data.py \
  --contract_path RFP_parsed.json \
  --output_path data/training/legal_qa.jsonl \
  --augment \
  --num_augmentations 2
```

This creates:
- `legal_qa.jsonl` - Full dataset
- `legal_qa_train.jsonl` - Training set (80%)
- `legal_qa_val.jsonl` - Validation set (10%)
- `legal_qa_test.jsonl` - Test set (10%)

#### 2. Train Model

**Standard LoRA Training**:
```bash
python finetuning/train.py \
  --data_path data/training \
  --output_dir checkpoints/qwen-legal-lora \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --lora_r 16 \
  --lora_alpha 32 \
  --batch_size 4 \
  --learning_rate 2e-4 \
  --num_epochs 3 \
  --gradient_accumulation_steps 4
```

**QLoRA Training** (for limited GPU memory):
```bash
python finetuning/train.py \
  --use_qlora \
  --data_path data/training \
  --output_dir checkpoints/qwen-legal-qlora \
  --batch_size 2 \
  --learning_rate 2e-4 \
  --num_epochs 3
```

**Using Config File**:
```bash
python finetuning/train.py \
  --config finetuning/configs/finetuning_config.yaml \
  --data_path data/training
```

#### 3. Monitor Training

Training logs are saved to TensorBoard:

```bash
tensorboard --logdir checkpoints/qwen-legal-lora
```

Access at `http://localhost:6006`

#### 4. Evaluate Fine-Tuned Model

```bash
python finetuning/train.py \
  --eval_only \
  --data_path data/training \
  --output_dir checkpoints/qwen-legal-lora
```

#### 5. Use Fine-Tuned Model

Update `.env`:
```bash
FINETUNING_ENABLED=true
FINETUNING_MODEL_PATH=checkpoints/qwen-legal-lora
USE_MERGED_MODEL=true
```

Restart the application to load the fine-tuned model.

### Training Data Format

Training data should be JSONL format:

```json
{
  "instruction": "You are an expert legal document analyst...",
  "input": "Contract Context: [retrieved document chunks]\n\nQuestion: What is the defect liability period?",
  "output": "Based on Article 5.2, the defect liability period is 2 years from the date of completion..."
}
```

### Hyperparameter Tuning

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `lora_r` | 16 | 8-64 | LoRA rank (higher = more capacity) |
| `lora_alpha` | 32 | 16-128 | Scaling factor (typically 2x rank) |
| `lora_dropout` | 0.05 | 0.0-0.1 | Dropout rate |
| `learning_rate` | 2e-4 | 1e-4 to 5e-4 | Learning rate |
| `batch_size` | 4 | 1-8 | Batch size (adjust for GPU memory) |
| `num_epochs` | 3 | 1-5 | Training epochs |

### Training Tips

1. **Start Small**: Begin with 100-500 examples to validate setup
2. **Monitor Loss**: Watch for overfitting (validation loss increasing)
3. **Early Stopping**: Use patience=3 to stop if no improvement
4. **Learning Rate**: Use learning rate finder or start with 2e-4
5. **Batch Size**: Use gradient accumulation for effective larger batches

### Troubleshooting Training

**Out of Memory**:
- Use QLoRA (`--use_qlora`)
- Reduce batch size
- Reduce max_length
- Enable gradient checkpointing

**Poor Convergence**:
- Increase learning rate
- Check data quality
- Verify data format
- Try different LoRA rank

See `finetuning/README.md` for comprehensive fine-tuning documentation.

## Project Structure

```
contract/
├── streamlit_app.py          # Main Streamlit app
├── src/
│   ├── data/                 # Data processing
│   ├── rag/                  # RAG system
│   ├── llm/                  # LLM integration
│   ├── finetuning/          # PEFT fine-tuning modules
│   ├── optimization/         # Caching, batching
│   ├── monitoring/           # Logging, metrics
│   ├── evaluation/           # Benchmark tests
│   └── ui/                   # UI components
├── finetuning/              # Training scripts and configs
├── docker/                   # Docker configs
├── deployment/               # Cloud deployment
└── evaluation/               # Benchmark scripts
```

## Development

### Project Structure

```
contract/
├── streamlit_app.py              # Main Streamlit application
├── requirements.txt              # Python dependencies
├── RFP_parsed.json              # Contract data (example)
│
├── src/                         # Source code
│   ├── __init__.py
│   ├── config.py                # Configuration management
│   │
│   ├── data/                    # Data processing
│   │   ├── contract_parser.py   # Parse contract JSON
│   │   ├── chunking.py          # Document chunking
│   │   └── preprocessing.py     # Text preprocessing
│   │
│   ├── rag/                     # RAG system
│   │   ├── retriever.py         # Hybrid retriever
│   │   └── vector_store.py      # Vector database wrapper
│   │
│   ├── llm/                     # LLM integration
│   │   ├── lm_studio_client.py  # LM Studio API client
│   │   ├── chain.py             # RAG chain
│   │   └── prompts.py          # Prompt templates
│   │
│   ├── finetuning/              # Fine-tuning modules
│   │   ├── data_generator.py    # Training data generation
│   │   ├── dataset.py           # Dataset classes
│   │   ├── peft_config.py       # PEFT configuration
│   │   ├── trainer.py           # Training logic
│   │   ├── evaluator.py        # Model evaluation
│   │   ├── model_loader.py     # Model loading utilities
│   │   ├── training_pipeline.py # End-to-end pipeline
│   │   └── utils.py            # Training utilities
│   │
│   ├── optimization/            # Performance optimization
│   │   ├── cache.py             # Caching layer
│   │   ├── batching.py          # Batch processing
│   │   └── rate_limiting.py    # Rate limiting
│   │
│   ├── monitoring/              # Monitoring and logging
│   │   ├── logger.py           # Structured logging
│   │   ├── metrics.py          # Metrics collection
│   │   ├── dashboard.py        # Metrics dashboard
│   │   └── alerting.py         # Alerting system
│   │
│   ├── evaluation/             # Evaluation and testing
│   │   ├── benchmark.py        # Benchmark suite
│   │   ├── metrics.py         # Evaluation metrics
│   │   └── test_cases.py      # Test case definitions
│   │
│   ├── ui/                     # UI components
│   │   ├── chat_interface.py  # Chat UI
│   │   ├── document_viewer.py # Document viewer
│   │   ├── metrics_dashboard.py # Metrics UI
│   │   └── components.py       # Reusable components
│   │
│   └── utils/                  # Utilities
│       └── timing.py           # Timing utilities
│
├── finetuning/                 # Training scripts
│   ├── train.py                # Training entry point
│   ├── generate_training_data.py # Data generation
│   ├── configs/                # Configuration files
│   │   └── finetuning_config.yaml
│   └── README.md               # Fine-tuning docs
│
├── evaluation/                 # Evaluation scripts
│   ├── run_benchmarks.py       # Run benchmarks
│   └── results_analysis.py     # Analyze results
│
├── docker/                     # Docker configs
│   ├── Dockerfile              # Docker image
│   └── docker-compose.yml      # Compose configuration
│
└── deployment/                 # Deployment configs
    ├── aws/                    # AWS CloudFormation
    │   └── cloudformation.yaml
    ├── kubernetes/             # Kubernetes manifests
    │   └── deployment.yaml
    └── scripts/                # Deployment scripts
        └── deploy.sh
```

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd contract

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8 mypy

# Install pre-commit hooks (optional)
pre-commit install
```

### Code Style

```bash
# Format code
black src/ finetuning/ evaluation/

# Lint code
flake8 src/ finetuning/ evaluation/

# Type checking
mypy src/
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test
pytest tests/test_rag_chain.py
```

### Adding New Features

1. **Create feature branch**: `git checkout -b feature/new-feature`
2. **Implement changes**: Follow code style guidelines
3. **Add tests**: Write unit tests for new functionality
4. **Update documentation**: Update README and docstrings
5. **Submit PR**: Create pull request with description

## Performance Tuning

### Performance Targets

- **Latency**: <200ms average response time
- **Accuracy**: >90% on legal benchmark tasks
- **Throughput**: Support 1-100 concurrent instances
- **Cost**: Optimized API usage through caching

### Optimization Strategies

#### 1. Caching

Enable caching for repeated queries:

```bash
CACHE_ENABLED=true
REDIS_ENABLED=true  # For distributed caching
```

#### 2. Batch Processing

Increase batch size for bulk operations:

```bash
BATCH_SIZE=20  # Default: 10
```

#### 3. Vector Store Optimization

- Use FAISS for faster search (if not using ChromaDB)
- Optimize chunk size and overlap
- Index frequently accessed documents

#### 4. Model Optimization

- Use quantized models (QLoRA)
- Merge adapters for faster inference
- Reduce max_tokens for faster responses

#### 5. Concurrent Processing

```python
from concurrent.futures import ThreadPoolExecutor

# Process multiple queries concurrently
with ThreadPoolExecutor(max_workers=5) as executor:
    results = executor.map(rag_chain.query, questions)
```

### Monitoring Performance

#### Metrics Dashboard

Access in Streamlit UI:
1. Click "View Metrics" in sidebar
2. View real-time metrics:
   - Request count
   - Average latency
   - Error rate
   - Function-level stats

#### Logging

Structured logs in `logs/` directory:

```bash
# View logs
tail -f logs/app.log

# Filter errors
grep ERROR logs/app.log
```

#### Performance Profiling

```python
from src.utils.timing import LatencyTracker

tracker = LatencyTracker()
tracker.start()

# Your code here
result = rag_chain.query("question")

latency_ms = tracker.stop()
print(f"Latency: {latency_ms}ms")
```

## Troubleshooting

### Common Issues

#### LM Studio Connection Errors

**Problem**: Cannot connect to LM Studio API

**Solutions**:
```bash
# 1. Verify LM Studio is running
curl http://192.168.0.4:1601/v1/models

# 2. Check firewall settings
# Allow port 1601 in firewall

# 3. Verify URL in .env
LM_STUDIO_API_URL=http://192.168.0.4:1601

# 4. Check network connectivity
ping 192.168.0.4
```

#### Vector Store Empty

**Problem**: No documents in vector store

**Solutions**:
```bash
# Reinitialize vector store
python init_vector_store.py

# Verify contract file exists
ls -la RFP_parsed.json

# Check ChromaDB directory
ls -la chroma_db/
```

#### Out of Memory (Fine-Tuning)

**Problem**: CUDA out of memory during training

**Solutions**:
```bash
# Use QLoRA
python finetuning/train.py --use_qlora

# Reduce batch size
python finetuning/train.py --batch_size 2

# Reduce max_length
python finetuning/train.py --max_length 1024

# Enable gradient checkpointing
# (already enabled by default)
```

#### Slow Response Times

**Problem**: High latency in responses

**Solutions**:
1. Enable caching: `CACHE_ENABLED=true`
2. Reduce `MAX_TOKENS` in config
3. Use smaller model or quantized version
4. Optimize vector store queries
5. Check network latency to LM Studio

#### Import Errors

**Problem**: Module not found errors

**Solutions**:
```bash
# Verify Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Check virtual environment
which python
```

### Getting Help

1. **Check Logs**: Review `logs/app.log` for errors
2. **Verify Configuration**: Run config validation
3. **Test Components**: Test individual components
4. **Review Documentation**: Check relevant README sections
5. **Open Issue**: Create GitHub issue with:
   - Error message
   - Steps to reproduce
   - Environment details
   - Log snippets

## API Reference

### RAGChain

Main interface for querying the RAG system.

```python
from src.llm.chain import RAGChain
from src.rag.retriever import HybridRetriever
from src.rag.vector_store import VectorStore

# Initialize
vector_store = VectorStore()
retriever = HybridRetriever(vector_store)
rag_chain = RAGChain(retriever)
```

#### Methods

**`query(question: str, n_context: int = 5, use_keyword: bool = True) -> Dict`**

Query the RAG system.

**Parameters**:
- `question`: User question
- `n_context`: Number of context documents to retrieve
- `use_keyword`: Enable keyword search

**Returns**:
```python
{
    'question': str,
    'answer': str,
    'context_documents': List[Dict],
    'sources': List[Dict]
}
```

**`stream_query(question: str, n_context: int = 5, use_keyword: bool = True) -> Generator`**

Stream response chunks.

**Returns**: Generator yielding:
```python
{
    'chunk': str,
    'answer_so_far': str,
    'sources': List[Dict]
}
```

**`extract_clauses(query: str, n_context: int = 10) -> Dict`**

Extract specific clauses.

**`classify_clause(clause_text: str) -> Dict`**

Classify a contract clause.

### LMStudioClient

LM Studio API client.

```python
from src.llm.lm_studio_client import LMStudioClient

client = LMStudioClient(
    model="qwen/qwen3-vl-4b",
    temperature=0.1
)
```

#### Methods

**`invoke(messages: List[Dict], system_prompt: Optional[str] = None) -> str`**

Invoke API with messages.

**`stream(messages: List[Dict], system_prompt: Optional[str] = None) -> Generator`**

Stream response.

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Follow code style**: Use `black` for formatting
4. **Write tests**: Add tests for new features
5. **Update docs**: Update README and docstrings
6. **Commit changes**: Use descriptive commit messages
7. **Push to branch**: `git push origin feature/amazing-feature`
8. **Open Pull Request**: Describe changes clearly

### Development Guidelines

- Write clear, documented code
- Add type hints where possible
- Write unit tests for new features
- Update documentation
- Follow PEP 8 style guide
- Keep commits atomic and descriptive

### Reporting Issues

When reporting issues, please include:
- Description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)
- Relevant log snippets
- Screenshots (if applicable)

## Acknowledgments

- [LM Studio](https://lmstudio.ai) for local LLM serving
- [Qwen](https://github.com/QwenLM/Qwen) for the base model
- [LangChain](https://langchain.com) for RAG framework
- [PEFT](https://github.com/huggingface/peft) for efficient fine-tuning
- [Streamlit](https://streamlit.io) for the UI framework
