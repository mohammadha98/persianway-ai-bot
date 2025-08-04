# Persian Agriculture RAG System

## Overview

This document provides a comprehensive guide to the Persian Agriculture Retrieval-Augmented Generation (RAG) system implemented in this project. The system processes agricultural PDF documents in Persian, creates vector embeddings, and provides an API for knowledge retrieval.

## System Architecture

### Components

1. **Document Processor** (`app/services/document_processor.py`)
   - Handles PDF loading and text extraction
   - Splits documents into manageable chunks
   - Creates vector embeddings using OpenAI's embedding model
   - Stores embeddings in ChromaDB

2. **Knowledge Base Service** (`app/services/knowledge_base.py`)
   - Provides RAG functionality using LangChain
   - Retrieves relevant documents for queries
   - Generates context-aware responses in Persian

3. **API Endpoints** (`app/api/routes/knowledge_base.py`)
   - `/api/knowledge/query`: Query the knowledge base
   - `/api/knowledge/process`: Process PDF documents
   - `/api/knowledge/status`: Check processing status

4. **Schemas** (`app/schemas/knowledge_base.py`)
   - Define request/response models for the API

### Data Flow

1. PDF documents in the `docs` folder are processed and converted to text
2. Text is split into chunks and converted to vector embeddings
3. Embeddings are stored in ChromaDB
4. When a query is received, relevant documents are retrieved
5. Retrieved documents and the query are sent to the language model
6. The language model generates a response based on the retrieved context

## Setup and Usage

### Prerequisites

- Python 3.8+
- OpenAI API key
- PDF documents in the `docs` folder

### Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Configure your OpenAI API key in the `.env` file:

```
OPENAI_API_KEY="your-openai-api-key-here"
```

### Initializing the Knowledge Base

Run the initialization script to process all PDF documents and create the vector database:

```bash
python init_knowledge_base.py
```

Alternatively, you can trigger document processing via the API after starting the server:

```bash
curl -X POST "http://localhost:8000/api/knowledge/process"
```

### Starting the Server

```bash
uvicorn main:app --reload
```

### Using the API

#### Query the Knowledge Base

```bash
curl -X POST "http://localhost:8000/api/knowledge/query" \
     -H "Content-Type: application/json" \
     -d '{"question":"انواع کودهای شیمیایی کدامند؟"}'
```

Response format:

```json
{
  "answer": "انواع کودهای شیمیایی عبارتند از: کودهای نیتروژنی، کودهای فسفاته، کودهای پتاسیمی، و کودهای میکرو.",
  "sources": [
    {
      "content": "کودهای شیمیایی به چند دسته تقسیم می‌شوند: کودهای نیتروژنی، کودهای فسفاته، کودهای پتاسیمی، و کودهای میکرو.",
      "source": "fertilization-guide-table.pdf",
      "page": 5
    }
  ]
}
```

#### Check Processing Status

```bash
curl "http://localhost:8000/api/knowledge/status"
```

## System Prompt

The system uses a specialized prompt in Persian that instructs the language model to:

- Answer questions about agricultural topics
- Use information from the knowledge base
- Respond in Persian with a professional tone
- Cite sources when appropriate

The prompt is defined in `app/core/config.py` as `PERSIAN_AGRICULTURE_SYSTEM_PROMPT`.

## Maintenance

### Adding New Documents

1. Add new PDF files to the `docs` directory
2. Run the initialization script or trigger processing via the API

### Modifying the System Prompt

Edit the `PERSIAN_AGRICULTURE_SYSTEM_PROMPT` in `app/core/config.py` to change how the model responds to queries.

### Troubleshooting

- If document processing fails, check that PDFs are not corrupted
- If queries return poor results, you may need to adjust the chunk size or overlap in `DocumentProcessor`
- For OpenAI API errors, verify your API key and quota

## Technical Details

### Vector Database

The system uses ChromaDB as the vector database, with data stored in the `vectordb` directory at the project root.

### Embedding Model

OpenAI's `text-embedding-ada-002` model is used for creating vector embeddings.

### Language Model

The system uses the model specified in `OPENAI_MODEL_NAME` (default: `gpt-3.5-turbo`) for generating responses.

### Document Chunking

Documents are split into chunks of 1000 characters with a 200-character overlap to maintain context between chunks.

## Integration with Existing Project

The RAG system is fully integrated with the existing FastAPI project structure:

- New components follow the established modular architecture
- API endpoints are accessible through the main API router
- The system uses the same configuration and dependency injection patterns

## Future Improvements

- Add support for image extraction from PDFs
- Implement caching for frequent queries
- Add user feedback mechanism to improve responses
- Support additional document formats (DOCX, TXT, etc.)
- Implement document metadata extraction for better source attribution