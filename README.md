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
| Vector database | ChromaDB (`1.x`, NumPy 2 compatible) |
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

> **Deployment caveat.** The bundle is a build artifact, and `.gitignore` excludes `dist/`, so it is *not* part of the repository. On a host deployed from git you must build it there (`cd frontend/ai-panel && npm ci && npm run build`, requires Node.js) or copy an existing `dist/ai-panel/browser` directory into place, **then restart the backend** — `main.py` registers the SPA routes at import time. If the bundle is missing, `/` answers `503` with a JSON hint instead of silently returning `404`.
>
> Also note that `git clean -fd` / `git reset --hard` during a deploy deletes the untracked bundle; either re-run the build afterwards or exclude the path (`git clean -fd -e frontend/ai-panel/dist`).

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

## Deployment

### Start command / port (Runflare and similar managed hosts)

Deploy through the panel **without overriding the start command and port**. Panels inject the
listening port (usually `$PORT`) and run their own supervisor plus health check. Hard-coding a
command such as:

```bash
gunicorn main:app -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 \
  -w 2 --timeout 0 --graceful-timeout 30 --keep-alive 75 --worker-connections 1000
```

produced an endless restart loop (7 restarts, ~100 s lifetime per attempt) while the app itself
stayed healthy: the logs showed an external `Handling signal: term`, **no** traceback, **no**
`WORKER TIMEOUT` and no OOM. The panel's own health check never got a good response from the
container — the hard-coded bind (`0.0.0.0:8000`) was not the port the panel probes, and the two
workers each repeat the cold-start warm-up, so the container lived past the check budget — hence
`SIGTERM` + restart. Clearing the custom command in the panel fixed it.

Guidelines:

- Let the panel build the command/port. If you must set one, keep the bind address, the app port
  and the panel's health-check port identical.
- Point the health check at `GET /health` (defined in `main.py`), not `/` — `/` answers `503`
  when the Angular bundle is missing (see the caveat above).
- Avoid `gunicorn --timeout 0` on hosts that have no external watchdog: it disables gunicorn's
  own worker killer, so a hung worker is never recycled and `WORKER TIMEOUT` never shows up in
  the logs (which hides the real problem). It is still useful when SSE responses are longer than
  the timeout — but then the platform health check must be tuned instead.
- With `-w N` greater than 1, every worker repeats the cold-start warm-up (its own Chroma client
  and BM25 index): budget N× CPU/RAM at boot, or keep a single worker on small plans.

### Chat transport mode & proxy timeouts

`POST/GET /api/chat/stream` stays an SSE endpoint in both of its transport modes, chosen by one
variable:

| `CHAT_STREAMING_ENABLED` | Behaviour |
| --- | --- |
| `true` | token by token: one `data: {"type":"chunk",...}` frame per model token |
| `false` (default) | single frame: the answer is buffered server-side and written once, inside the `done` frame |

Both modes share the whole route: the response still opens with a `: ping` comment frame, keeps
writing comment frames while retrieval/generation run, still forwards `status`, `metadata` and
`error` frames, and still ends with `data: [DONE]`. That is deliberate — those comment frames are
the only thing keeping a proxy from treating a long answer as an idle connection, and the answer
inside the `done` frame is what the frontend renders, so there is one server code path and one
client parser for both modes. Switching modes is an environment change plus a restart; nothing in
the repository has to be edited.

**nginx (or any proxy with an idle read timeout).** nginx's default `proxy_read_timeout` is 60 s: an
answer that stays silent longer than that is reset upstream (`upstream prematurely closed
connection`), which the browser reports as `net::ERR_HTTP2_PROTOCOL_ERROR`. Keep the heartbeats and
raise the timeout past the slowest expected answer:

```nginx
location /api/chat/stream {
    proxy_pass http://127.0.0.1:8000;
    proxy_http_version 1.1;
    proxy_set_header Connection "";
    proxy_buffering off;        # never buffer an SSE body
    proxy_cache off;
    proxy_read_timeout 300s;    # must exceed the slowest expected answer
    proxy_send_timeout 300s;
}
```

**`gunicorn --timeout` is not a per-request deadline here.** With `-k uvicorn.workers.UvicornWorker`
gunicorn hands its timeout to uvicorn as `timeout_notify` and uses it as the interval for
`callback_notify` — the worker's liveness ping back to the gunicorn arbiter
(`uvicorn/workers.py`: `"timeout_notify": self.timeout, "callback_notify": self.callback_notify`;
`uvicorn/server.py::on_tick` fires that callback every `timeout_notify` seconds). The event loop
keeps ticking while an async handler awaits the model, so a 158 s answer does not produce
`WORKER TIMEOUT`; only a *blocking* call inside the handler can starve the tick and cause one. Tune
the proxy instead — the warning above about `--timeout 0` still stands.

### Startup window & observability

The app schedules a background retrieval warm-up after startup (lifespan in `main.py` →
`KnowledgeBaseService.warm_up()`), so early requests are fast, but the first ~1–2 minutes of a
container's life are CPU heavy. Any probe on the container must tolerate that window.

Note that application-level logs (`logging.info(...)` across `app/`) are **not** emitted in
production, because `app/core/logging.py::setup_logging()` is never called: only gunicorn/uvicorn
lines and library warnings reach stdout. Call `setup_logging()` from `main.py` if you want the
warm-up progress (`[KB WARMUP] ...`) visible in the pod logs.

## Testing

```bash
pytest
```

The test suite (`tests/`) covers API, services, unit and tooling behaviour: retrieval, re-ranking, confidence scoring, document processing, conversation filtering, intent detection and more.

## License

MIT