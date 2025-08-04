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