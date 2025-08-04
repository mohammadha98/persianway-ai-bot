# Persian Way Knowledge Base RAG System

## Overview

This document provides information about the Retrieval-Augmented Generation (RAG) system implemented for the Persian Agriculture Knowledge Base. The system processes PDF documents containing agricultural information, converts them into vector embeddings, and provides an API for querying this knowledge base in Persian.

## Features

- **PDF Processing**: Automatically extracts text from PDF files in the `docs` folder
- **Vector Embeddings**: Converts text into semantic vectors using OpenAI embeddings
- **Vector Database**: Stores embeddings in ChromaDB for efficient retrieval
- **RAG Implementation**: Uses LangChain to implement a retrieval-augmented generation system
- **Persian Language Support**: Designed to work with Persian text throughout the pipeline
- **API Integration**: Seamlessly integrates with the existing FastAPI structure

## System Components

### Document Processor

The `DocumentProcessor` class in `app/services/document_processor.py` handles:

- Loading PDF files from the `docs` directory
- Splitting documents into manageable chunks
- Creating vector embeddings using OpenAI's embedding model
- Storing embeddings in ChromaDB

### Knowledge Base Service

The `KnowledgeBaseService` class in `app/services/knowledge_base.py` provides:

- Integration with the document processor
- A retrieval system that finds relevant documents for a query
- A question-answering system that generates responses based on retrieved documents
- Persian-specific prompting and response formatting

### API Endpoints

The knowledge base API in `app/api/routes/knowledge_base.py` exposes:

- `/api/knowledge/query`: Query the knowledge base with a question
- `/api/knowledge/process`: Process all PDF documents in the `docs` directory
- `/api/knowledge/status`: Check the status of the document processing

## System Prompt

The system includes a Persian-language prompt specifically designed for agricultural questions. This prompt instructs the model to:

- Answer questions about fertilizers, soil improvement, soil textures, pH/EC adjustment, and fertilization methods
- Use information from the knowledge base
- Respond in Persian with a professional tone
- Cite sources when appropriate
- Admit when it doesn't know an answer

## Usage

### Initial Setup

1. Ensure all PDF documents are placed in the `docs` directory
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key in the `.env` file:

```
OPENAI_API_KEY="your-openai-api-key-here"
```

### Processing Documents

Before querying the knowledge base, you need to process the documents:

1. Start the FastAPI server:

```bash
uvicorn main:app --reload
```

2. Trigger document processing via the API:

```bash
curl -X POST "http://localhost:8000/api/knowledge/process"
```

This will start processing all PDF files in the `docs` directory in the background.

3. Check the processing status:

```bash
curl "http://localhost:8000/api/knowledge/status"
```

### Querying the Knowledge Base

Once the documents are processed, you can query the knowledge base:

```bash
curl -X POST "http://localhost:8000/api/knowledge/query" \
     -H "Content-Type: application/json" \
     -d '{"question":"انواع کودهای شیمیایی کدامند؟"}'
```

The response will include:
- An answer to the question in Persian
- Source information from the documents used to generate the answer

## Maintenance

### Adding New Documents

To add new documents to the knowledge base:

1. Place the new PDF files in the `docs` directory
2. Trigger document processing via the API as described above

### Updating the System Prompt

The system prompt can be updated in `app/core/config.py` by modifying the `PERSIAN_AGRICULTURE_SYSTEM_PROMPT` setting.

## Integration with Existing System

The RAG system is fully integrated with the existing FastAPI project structure:

- New components follow the established modular architecture
- API endpoints are accessible through the main API router
- The system uses the same configuration and dependency injection patterns

## Troubleshooting

### Common Issues

- **PDF Processing Errors**: Check that PDF files are not corrupted and are readable
- **OpenAI API Errors**: Verify that your API key is valid and has sufficient quota
- **Vector Database Issues**: If ChromaDB fails to load, try deleting the `vectordb` directory and reprocessing the documents

### Logs

Check the application logs for detailed error messages and debugging information.