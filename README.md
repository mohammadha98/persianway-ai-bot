# Persian Way AI Consultation Bot

A production-oriented **RAG (Retrieval-Augmented Generation)** chatbot and admin panel for the **Persian Way** platform, covering the **health, beauty and agriculture** domains. It answers user questions from an internal knowledge base, refers low-confidence questions to human experts, and ships with a full admin interface for managing chat, knowledge, models, users and settings.

- **Backend**: FastAPI + LangChain + ChromaDB (vector store) + MongoDB (conversations / feedback / users / config) + Tavily (web search)
- **Frontend**: Angular 19 admin panel (`frontend/ai-panel`)

---

## Features

- **RAG chat** — hybrid retrieval (dense + BM25), re-ranking, confidence scoring, query expansion, context condensation and **streaming** responses
- **Document processing** — PDF, Word (`.docx`) and Excel (`.xls/.xlsx`) ingestion with Persian/RTL normalization and table extraction
- **Knowledge base management** — upload / process / remove documents, list & query knowledge, MongoDB ↔ VectorDB sync
- **AI feedback** — trainers can approve / report / edit responses and add answers to the knowledge base; the prompt and retrieved context are visible per message
- **Spell checker** — Persian spell correction with caching and a custom dictionary
- **Users & auth** — registration, login (JWT), roles/permissions and password management
- **Dynamic configuration** — LLM / RAG / database settings editable at runtime and persisted in MongoDB
- **Human referral** — low-confidence questions are logged for expert review
- **Admin panel** — Angular SPA for chat, knowledge, models, settings, users and tools

## Tech Stack

| Layer | Technology |
|---|---|
| API framework | FastAPI (Python 3.8+, tested on 3.12/3.13) |
| LLM orchestration | LangChain (`0.3.x`) + OpenAI / OpenRouter |
| Vector database | ChromaDB (`0.4.x`) |
| Document database | MongoDB (Motor / PyMongo) |
| Frontend | Angular 19 + Angular Material + SCSS |
| Search | Tavily + BM25 (`rank_bm25`) |
| Auth | PyJWT + bcrypt |

## Project Structure

```
persianway-ai-bot/
├── app/
│   ├── api/
│   │   ├── routes/                 # API endpoints
│   │   │   ├── chat.py                 # chat + streaming + feedback
│   │   │   ├── config.py               # runtime configuration
│   │   │   ├── conversations.py        # conversation history
│   │   │   ├── knowledge_base.py       # knowledge list / query / process
│   │   │   ├── knowledge_contribution.py  # MongoDB <-> VectorDB sync
│   │   │   ├── predictions.py          # ML predictions (legacy)
│   │   │   ├── spell_check.py          # Persian spell correction
│   │   │   ├── ui.py                   # server-rendered (Jinja2) UI
│   │   │   ├── upload.py               # file upload + PDF processing
│   │   │   └── users.py                # auth & user management
│   │   └── dependencies.py         # FastAPI dependencies
│   ├── core/                      # config, logging, Jinja2 templates
│   ├── middleware/                # conversation logging middleware
│   ├── models/                    # (legacy) example ML model interface
│   ├── schemas/                   # Pydantic request/response schemas
│   ├── services/                  # business logic
│   │   ├── chat_service.py            # RAG chat pipeline
│   │   ├── knowledge_base.py          # retrieval & knowledge operations
│   │   ├── hybrid_retrieval.py        # dense + sparse hybrid search
│   │   ├── reranker.py                # re-ranking
│   │   ├── context_condenser.py       # context summarization
│   │   ├── document_processor.py      # PDF / DOCX processing
│   │   ├── excel_processor.py         # Excel QA processing
│   │   ├── feedback_service.py        # AI feedback storage
│   │   ├── conversation_service.py    # conversation persistence
│   │   ├── config_service.py          # dynamic settings (MongoDB)
│   │   ├── model_service.py           # LLM provider management
│   │   ├── spell_corrector.py         # Persian spell checking
│   │   ├── user_service.py            # auth / users
│   │   ├── task_service.py            # async task status
│   │   ├── database.py                # MongoDB / SQLite access
│   │   └── dictionaries/words.json    # spell-checker dictionary
│   ├── static/                    # CSS / JS for the legacy UI
│   ├── templates/                 # Jinja2 templates (legacy UI)
│   └── utils/                     # validators etc.
├── frontend/
│   └── ai-panel/                  # Angular 19 admin panel
├── tests/                         # pytest suite (api, services, unit, tools)
├── examples/                      # usage examples
├── scripts/
│   └── process_documents.py       # document ingestion CLI
├── .env.example                   # example environment variables
├── main.py                        # FastAPI entry point
└── requirements.txt               # Python dependencies
```

## Getting Started

### Prerequisites

- Python 3.8+ (tested on 3.12 / 3.13)
- Node.js 18+ and npm (for the Angular admin panel)
- MongoDB running locally (or a connection string in `.env`)

### Backend

```bash
git clone <repository-url>
cd persianway-ai-bot

# 1. Create & activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# Unix / macOS
# source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment variables
copy .env.example .env   # then fill in your API keys

# 4. Run the server
python main.py
# or
uvicorn main:app --reload
```

The server starts at `http://localhost:8000`:

- Swagger UI: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

### Frontend (admin panel)

```bash
cd frontend/ai-panel
npm install

# Development server (http://localhost:4200)
ng serve

# Production build — served by the FastAPI backend at `/`
ng build --configuration production
```

When `frontend/ai-panel/dist/ai-panel/browser` exists, FastAPI automatically serves the Angular SPA at `/` and `/ui/...`.

## Configuration

Environment variables are loaded from `.env` (see `.env.example` for the full list).

| Variable | Description | Example |
|---|---|---|
| `PREFERRED_API_PROVIDER` | Provider selection: `auto` / `openai` / `openrouter` | `auto` |
| `DEFAULT_MODEL` | Default chat model | `openai/gpt-4.1` |
| `OPENAI_API_KEY` | OpenAI API key | — |
| `OPENAI_EMBEDDING_MODEL` | Embedding model | `text-embedding-3-small` |
| `OPENROUTER_API_KEY` | OpenRouter API key | — |
| `OPENROUTER_API_BASE` | OpenRouter endpoint | `https://openrouter.ai/api/v1` |
| `TAVILY_API_KEY` | Web search API key | — |
| `MONGODB_URL` | MongoDB connection string | `mongodb://localhost:27017` |
| `MONGODB_DATABASE` | MongoDB database name | `persian_way_ai` |
| `TEMPERATURE` / `TOP_P` / `MAX_TOKENS` | LLM generation parameters | `0.7` / `1.0` / `512` |
| `ALLOWED_HOSTS` | CORS allowed origins | `*` |

## API Endpoints

All routes are prefixed with `/api` (plus `/health` and `/ui` from `main.py`).

| Group | Endpoint(s) | Description |
|---|---|---|
| Health | `GET /health` | Health & version check |
| Chat | `POST /api/chat/` | Send a chat message |
|  | `POST` / `GET /api/chat/stream` | Streaming chat |
|  | `GET /api/chat/context/{message_id}` | Message prompt + retrieved context |
| Feedback | `POST` / `GET /api/chat/feedback` | Submit / list AI feedback |
|  | `GET /api/chat/feedback/{id}` · `PATCH .../status` · `DELETE ...` | Manage a single feedback |
|  | `GET /api/chat/feedback/stats/summary` | Feedback statistics |
| Config | `GET` / `PUT /api/config/` · `POST /api/config/reset` | Get / update / reset settings |
|  | `GET /api/config/llm` · `/rag` · `/database` | Per-group settings |
| Conversations | `GET /api/conversations/{user_id}` · `/search` · `/search/advanced` | Conversation history |
|  | `GET /api/conversations/stats/overview` | Usage stats |
| Knowledge | `GET /api/knowledge/knowledge-list` | Paginated knowledge list |
|  | `POST /api/knowledge/query` | Query the knowledge base |
|  | `POST /api/knowledge/process` · `/process-excel` | Ingest documents |
|  | `POST /api/knowledge/contribute` · `DELETE /api/knowledge/remove/{hash_id}` | Add / remove knowledge |
| Upload | `POST /api/upload/` | Upload a file |
|  | `POST /api/upload/pdf/process` | Upload & vectorize a PDF |
| Spell check | `POST /api/spell-check/correct-text` · `/correct-word` · `/suggestions` | Persian spell correction |
| Users | `POST /api/users/login` · `POST /api/users/` · `GET /api/users/me` | Auth & profile |
|  | `GET/PUT/DELETE /api/users/{user_id}` · `.../permissions` · `.../reset-password` | User management |
| Predictions | `POST /api/predictions/` · `GET /api/predictions/models` | Legacy ML predictions |

## Document Processing & Knowledge Base

Documents are ingested, chunked, embedded (OpenAI embeddings) and stored in ChromaDB for semantic retrieval, with BM25 hybrid search and re-ranking on top.

**Supported formats:** PDF, Word (`.docx`), Excel (`.xls` / `.xlsx`), with full **Persian / RTL** support and automatic table extraction.

```python
from app.services.document_processor import get_document_processor

processor = get_document_processor()

# Process a single DOCX file
documents = processor.process_docx("path/to/file.docx")

# Batch process a directory (PDF + DOCX) and build vectors
stats = processor.batch_process_mixed_directory(
    input_dir="docs",
    output_dir="processed",
    create_vectors=True
)

# Search processed documents
results = processor.search_documents("your query", k=5)
```

See `examples/` for runnable scripts.

## Testing

```bash
pytest
```

The test suite (`tests/`) covers API, services, unit and tooling behaviour: retrieval, re-ranking, confidence scoring, document processing, conversation filtering, intent detection and more.

## License

MIT