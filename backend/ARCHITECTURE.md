# VEGACHAT Backend Architecture and Usage

This document explains how the backend is built, how requests are processed end‑to‑end, and exactly how to send inputs (including text and images) to get reliable results.

## Overview

- Framework: Flask (Python)
- AI Model: Google Gemini (gemini-2.0-flash-exp)
- Memory:
  - Short-term: last N messages from SQLite per session
  - Long-term: hybrid retrieval
    - Structured: recent history from SQLite
    - Semantic: ChromaDB vector store with sentence-transformers/all-MiniLM-L6-v2
- Agent pattern: two-step "tool-using" flow
  1) Decide: direct answer vs tool call
  2) If tool called: retrieve SQL + vector results, then synthesize final answer

Directory highlights:
- `app.py` — Flask app and REST endpoints
- `agent.py` — Agent orchestration, tool-calling, Gemini prompts
- `database.py` — SQLite persistence (sessions, conversations)
- `vector_store.py` — ChromaDB semantic search and embeddings
- `vegapunk_memory.db` — SQLite database file
- `chroma_db/` — ChromaDB persistence directory

## Data flow (request → response)

1) Frontend sends a JSON payload to the backend.
2) `app.py` extracts text and optional image from the HTML-rich `message` field:
   - Text is obtained by stripping `<img>` tags.
   - Inline base64 `<img src="data:image/...;base64, ...">` is decoded to bytes when present.
3) `app.py` resolves or creates a `session_id` and calls `VegapunkAgent.chat()`.
4) `agent.chat()` builds a short-term memory window, asks the model to either:
   - answer directly, or
   - call the tool to fetch long-term memory (SQL + vector).
5) If tool is used, a second model call synthesizes the final response.
6) Messages are saved to SQLite and embedded into Chroma (unless sub-chat where saving is disabled).
7) JSON response is returned to the frontend.

## Requests: how to send inputs

All chat endpoints accept a JSON body and expect the `message` field to be HTML content. This enables optional inline images via base64 `<img>` tags.

- Text-only example:
  ```json
  { "message": "Hello there" }
  ```

- Text + image example (truncated for brevity):
  ```json
  {
    "message": "Here is my diagram <img src=\"data:image/png;base64,iVBORw0KGgoAAA...\">"
  }
  ```

The backend will extract text for the LLM prompt and decode the first inline base64 image (if present) for multimodal Gemini input.

### Main Chat
- Path: `POST /api/chat/main`
- Request body:
  - `message` (string, required): HTML content.
  - `session_id` (string, optional): If omitted, a new session is created and titled from the first 50 chars of text content.
- Response:
  ```json
  {
    "success": true,
    "response": "model reply",
    "session_id": "uuid",
    "context_used": false,
    "timestamp": "2025-11-08T12:34:56.789Z",
    "chat_type": "main",
    "model": "gemini-2.0-flash-exp"
  }
  ```
  - `context_used` is true when the agent used the long‑term memory tool.

### Sub Chat (Satellite)
- Path: `POST /api/chat/sub`
- Request body:
  - `message` (string, required): HTML content.
  - `subChatId` (number|string, required): Satellite ID label (added to the prompt).
  - `session_id` (string, required): Must reference an existing main chat session.
- Behavior:
  - Can read main chat memory but does not write new memory (`save_to_memory=False`).
- Response shape mirrors main chat with `chat_type: "sub"`.

### Sessions & History
- `GET /api/sessions`
  - Returns all sessions with message counts and timestamps.
- `GET /api/sessions/grouped`
  - Groups sessions by title to show them as one logical entity.
- `GET /api/sessions/group/{group_key}`
  - Returns combined history for all sessions that share the same title.
- `GET /api/sessions/{session_id}`
  - Returns chronological message history for a session.
- `DELETE /api/sessions/{session_id}`
  - Deletes a session and its messages.
- `PUT /api/sessions/group/{group_key}/title`
  - Body: `{ "new_title": "Your Title" }`
  - Renames all sessions in a group.

## Response contracts (high level)

- Success responses include `success: true` and relevant data.
- Error responses include `error` with a message and appropriate HTTP status codes (400/404/500).

## Memory & storage schema

SQLite tables:
- `sessions(session_id PK, title, created_at, updated_at)`
- `conversations(id PK, session_id, role, content, timestamp, metadata)`

ChromaDB:
- Collection: `vegapunk_conversations`
- Each message stored with metadata `{ session_id, message_id, role }` and text content.

## Agent behavior

- Short-term window: last 10 messages via `db.get_session_history(limit=10)`
- Decision prompt: decides between direct answer and tool call (JSON tool request)
- Tool `tool_A` executes:
  - SQL recent history (e.g., last 7 days)
  - Vector semantic search (top-k similar messages)
- Final synthesis prompt crafts the user-facing answer without exposing tool details.

## Run locally (Windows PowerShell)

```powershell
# From VEGAPUNK/backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
# Edit .env and set GEMINI_API_KEY=YOUR_KEY
python app.py
```

Server starts on `http://localhost:5000`.

## Test quickly (PowerShell)

```powershell
# Health
Invoke-RestMethod -Method GET http://localhost:5000/api/health | ConvertTo-Json -Depth 5

# Main chat (creates session automatically)
Invoke-RestMethod -Method POST http://localhost:5000/api/chat/main `
  -ContentType 'application/json' `
  -Body (@{ message = 'Hello VEGACHAT' } | ConvertTo-Json) | ConvertTo-Json -Depth 5

# Sub chat (replace $sid with returned session_id)
$sid = 'REPLACE_WITH_SESSION_ID'
Invoke-RestMethod -Method POST http://localhost:5000/api/chat/sub `
  -ContentType 'application/json' `
  -Body (@{ message = 'A satellite question'; subChatId = 1; session_id = $sid } | ConvertTo-Json) `
  | ConvertTo-Json -Depth 5
```

## Sending images

To include an image, embed it as an inline base64 `<img>` in the `message` HTML.

Example (JavaScript):
```ts
async function toDataImg(file) {
  const buf = await file.arrayBuffer();
  const b64 = btoa(String.fromCharCode(...new Uint8Array(buf)));
  const mime = file.type || 'image/png';
  return `<img src="data:${mime};base64,${b64}">`;
}

// Then send
const html = `Please analyze this ${await toDataImg(file)}`;
fetch('/api/chat/main', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ message: html, session_id })
});
```

Backend will extract the first image and pass it to Gemini together with the prompt text.

## Operational notes & recommendations

- Validation: Consider adding request validation (e.g., Marshmallow) to enforce schemas.
- OpenAPI: Expose a spec via Flask-RESTX/Flask-Smorest for easy client generation.
- Logging: Add structured logs for tool decisions and parse errors (JSON extraction).
- Security: Rate limiting (Flask-Limiter), auth (API keys/JWT) for production.
- CORS: Adjust origins in `app.py` to match your frontend host/port.
- Backups: Periodically back up `vegapunk_memory.db` and `chroma_db/`.
- Migrations: If schemas evolve, use Alembic or simple migration scripts.

## Troubleshooting

- "Gemini API key not configured": Ensure `.env` has `GEMINI_API_KEY` and the venv is active.
- CORS errors: Ensure your frontend origin is whitelisted in `CORS(app, resources=...)`.
- Model always answers directly: Improve JSON parsing robustness and tighten the decision prompt in `agent.py`.
