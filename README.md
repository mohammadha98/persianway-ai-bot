# ML Model API Server

A professional FastAPI-based web server for deploying machine learning models as RESTful APIs. This project provides a structured and maintainable architecture for serving ML models, with built-in API documentation and health monitoring.

## Features

- **Modular Architecture**: Well-organized folder structure for easy maintenance and extension
- **ML Model Integration**: Simple framework for adding new machine learning models
- **API Documentation**: Auto-generated Swagger/OpenAPI documentation
- **Health Monitoring**: Built-in health check endpoint
- **Scalable Design**: Easily extendable for multiple models and endpoints

## Project Structure

```
├── app/                    # Main application package
│   ├── api/                # API endpoints
│   │   ├── routes/         # API route definitions
│   │   └── dependencies.py # API dependencies
│   ├── core/               # Core application components
│   │   ├── config.py       # Configuration settings
│   │   └── logging.py      # Logging configuration
│   ├── models/             # ML model definitions
│   │   ├── base.py         # Base model interface
│   │   └── example_model.py # Example model implementation
│   ├── schemas/            # Pydantic schemas for request/response validation
│   │   └── prediction.py   # Prediction request/response schemas
│   ├── services/           # Business logic services
│   │   └── model_service.py # Model loading and prediction service
│   └── utils/              # Utility functions
│       └── helpers.py      # Helper functions
├── tests/                  # Test directory
│   ├── api/                # API tests
│   └── models/             # Model tests
├── .env                    # Environment variables (not in version control)
├── .env.example            # Example environment variables
├── .gitignore              # Git ignore file
├── main.py                 # Application entry point
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## Getting Started

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd ml-model-api-server
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Copy the example environment file and modify as needed:

```bash
copy .env.example .env
```

5. Configure your API keys in the .env file:

```
# Choose your preferred API provider: "auto", "openai", or "openrouter"
PREFERRED_API_PROVIDER="auto"

# OpenAI API key (required if using OpenAI directly)
OPENAI_API_KEY="your-openai-api-key-here"

# OpenRouter API key (required if using OpenRouter)
OPENROUTER_API_KEY="your-openrouter-api-key-here"
```

### Using the Chat API

The server provides a chat endpoint that uses LangChain to create a conversational AI system. Each user session maintains its own conversation history.

#### Supported Model Providers

The system supports multiple language model providers:

- **OpenAI** (directly via OpenAI API)
- **Multiple providers via OpenRouter**:
  - OpenAI models: `openai/gpt-4`, `openai/gpt-3.5-turbo`
  - Google models: `google/gemini-pro`
  - Anthropic models: `anthropic/claude-2`
  - Meta models: `meta-llama/llama-2-70b-chat`
  - And many more available through OpenRouter

You can control which provider to use with the `PREFERRED_API_PROVIDER` setting in your `.env` file.

#### Chat Endpoint

- **URL**: `/api/chat/`
- **Method**: POST
- **Request Body**:
  ```json
  {
    "user_id": "unique-user-identifier",
    "message": "Your message here",
    "model": "google/gemini-pro"  // Optional: specify which model to use
  }
  ```
- **Response**:
  ```json
  {
    "response": "AI assistant's response",
    "conversation_history": [
      {"role": "user", "content": "Your message here"},
      {"role": "assistant", "content": "AI assistant's response"}
    ]
  }
  ```

#### Example Request

```bash
curl -X POST "http://localhost:8000/api/chat/" \
     -H "Content-Type: application/json" \
     -d '{"user_id": "user123", "message": "Hello, how can you help me today?"}'
```

### Document Processing and Knowledge Base

The system includes advanced document processing capabilities for both **PDF** and **Word (.docx)** files. Documents are processed, converted to embeddings, and stored in a vector database for semantic search and retrieval.

#### Supported File Formats

- **PDF Files**: Full support for PDF documents with text and table extraction
- **Word Documents (.docx)**: Complete support for Word documents including tables and formatting
- **Persian/RTL Languages**: Native support for Persian, Arabic, and other RTL languages
- **Table Extraction**: Automatic detection and extraction of tables from both PDF and DOCX files

#### Key Features

1. **Automatic Text Extraction**
   - Clean text extraction from PDF and DOCX files
   - Persian text normalization (Arabic to Persian numerals, etc.)
   - Header and structure detection
   - Markdown conversion for easy viewing

2. **Table Processing**
   - Automatic table detection in both formats
   - Conversion to structured DataFrames
   - Markdown table formatting
   - Metadata preservation (rows, columns, page numbers)

3. **Vector Store Integration**
   - Automatic chunking of documents
   - OpenAI embeddings generation
   - ChromaDB vector storage
   - Semantic similarity search

4. **Batch Processing**
   - Process multiple files in a directory
   - Progress tracking and statistics
   - Mixed PDF/DOCX processing
   - Error handling and recovery

#### Quick Start

```python
from app.services.document_processor import get_document_processor

processor = get_document_processor()

# Process a single DOCX file
documents = processor.process_docx("path/to/file.docx")

# Process all documents in a directory (PDF + DOCX)
results = processor.process_all_documents()

# Batch process with statistics
stats = processor.batch_process_mixed_directory(
    input_dir="docs",
    output_dir="processed",
    create_vectors=True
)

# Search processed documents
results = processor.search_documents("your query", k=5)
```

#### Installation

Install the required dependencies:

```bash
pip install python-docx pymupdf pandas tabulate arabic-reshaper python-bidi
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

#### Documentation

For detailed documentation on document processing features, see:
- [DOCX Processing Documentation](docs/DOCX_PROCESSING.md)
- [Examples](examples/docx_processing_example.py)

### Running the Server

```bash
uvicorn main:app --reload
```

The server will start at `http://localhost:8000`.

- API documentation is available at `http://localhost:8000/docs`
- Alternative documentation is at `http://localhost:8000/redoc`
- Health check endpoint is at `http://localhost:8000/health`

## Adding New ML Models

To add a new machine learning model to the API:

1. **Create a new model file** in the `app/models/` directory, implementing the `BaseModel` interface from `app/models/base.py`.

```python
# app/models/your_new_model.py
from app.models.base import BaseModel

class YourNewModel(BaseModel):
    def __init__(self):
        # Initialize your model here
        self.model = None
        self.load()

    def load(self):
        # Load your model here
        # Example: self.model = joblib.load('path/to/model.pkl')
        pass

    def predict(self, data):
        # Make predictions using your model
        # Example: return self.model.predict(data)
        pass
```

2. **Register your model** in the model service (`app/services/model_service.py`).

3. **Create API endpoints** for your model in a new file under `app/api/routes/`.

4. **Add request/response schemas** in the `app/schemas/` directory if needed.

5. **Update the main router** in `app/api/routes/__init__.py` to include your new endpoints.

## Best Practices

- Keep models lightweight and focused on a single task
- Use appropriate input validation for all API endpoints
- Add comprehensive tests for new models and endpoints
- Document all API endpoints with appropriate descriptions and examples
- Use environment variables for configuration settings
- Implement proper error handling and logging

## License

[MIT](LICENSE)