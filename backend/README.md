# VEGACHAT Backend

Python Flask backend for the VEGACHAT chat application, powered by Google Gemini AI.

For a deep-dive into architecture, endpoints, and exact input formats (including images), see `ARCHITECTURE.md`.

## Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))

### Installation

1. Create a virtual environment:
```powershell
python -m venv venv
```

2. Activate the virtual environment:
```powershell
# Windows PowerShell
.\venv\Scripts\Activate.ps1

# Windows CMD
.\venv\Scripts\activate.bat
```

3. Install dependencies:
```powershell
pip install -r requirements.txt
```

4. Set up environment variables:
```powershell
# Copy the example env file
Copy-Item .env.example .env

# Edit .env with your Gemini API key
```

### Getting Your Gemini API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the API key and paste it in your `.env` file

### Running the Server

```powershell
python app.py
```

The server will start on `http://localhost:5000` by default.

## API Endpoints

### Health Check
- **GET** `/api/health`
- Returns server status and timestamp

### Main Chat
- **POST** `/api/chat/main`
- Body (HTML string in `message`, optional `session_id`):
  ```json
  { "message": "Hello <img src=\"data:image/png;base64,...\">", "session_id": "optional-uuid" }
  ```
- Creates a session when `session_id` is omitted, returns the `session_id` in response.

### Sub Chat (Satellites)
- **POST** `/api/chat/sub`
- Body: `{ "message": "your message", "subChatId": 1, "session_id": "existing-uuid" }`
- Sub-chats read main memory but do not persist new messages.

### Sessions & History
- **GET** `/api/sessions` — list sessions
- **GET** `/api/sessions/grouped` — sessions grouped by title
- **GET** `/api/sessions/group/{group_key}` — combined history for a group
- **GET** `/api/sessions/{session_id}` — session history
- **DELETE** `/api/sessions/{session_id}` — delete session
- **PUT** `/api/sessions/group/{group_key}/title` — rename group (body: `{ "new_title": "..." }`)

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Server port | `5000` |
| `DEBUG` | Debug mode | `True` |
| `GEMINI_API_KEY` | Your Google Gemini API key | Required |
| `GEMINI_MODEL` | Gemini model to use | `gemini-2.0-flash-exp` |
| `ALLOWED_ORIGINS` | CORS allowed origins | Configured in code (app.py) |
| `SECRET_KEY` | Session secret key | Required for production |
| `LOG_LEVEL` | Logging level | `INFO` |

### Available Gemini Models

- `gemini-2.0-flash-exp` — multimodal, fast

## Development

### Testing the API

You can test the endpoints using curl or any HTTP client:

```powershell
# Health check
curl http://localhost:5000/api/health

# Main chat
curl -X POST http://localhost:5000/api/chat/main `
  -H "Content-Type: application/json" `
  -d '{"message": "Hello, VEGAPUNK!"}'

# Sub chat
curl -X POST http://localhost:5000/api/chat/sub `
  -H "Content-Type: application/json" `
  -d '{"message": "Test message", "subChatId": 1}'
```

### CORS Configuration
The backend is configured to accept requests from:
- `http://localhost:5173` (Vite default port)
- `http://localhost:3000` (Alternative dev port)

Add more origins in the `.env` file if needed.

## Production Deployment

For production, use a production WSGI server like Gunicorn:

```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

Remember to:
1. Set `DEBUG=False` in `.env`
2. Use a strong `SECRET_KEY`
3. Configure proper CORS origins
4. Keep your `GEMINI_API_KEY` secure
5. Set up proper logging
6. Consider rate limiting for API calls

## Troubleshooting

### API Key Issues
- Ensure your API key is correctly set in `.env`
- Check if the API key has the necessary permissions
- Verify your API quota hasn't been exceeded

### CORS Errors
- Make sure your frontend URL is listed in `ALLOWED_ORIGINS`
- Check if the frontend is making requests to the correct backend URL

### Module Import Errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check if the virtual environment is activated
