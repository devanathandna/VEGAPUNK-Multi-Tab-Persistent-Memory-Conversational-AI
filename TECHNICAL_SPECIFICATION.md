# VEGAPUNK Technical Specification

## 1. Introduction
VEGAPUNK is an advanced AI chat system designed with a **Hybrid Memory Architecture**. Unlike traditional RAG (Retrieval-Augmented Generation) systems that rely solely on vector similarity, VEGAPUNK integrates three distinct memory types:
1.  **Semantic Memory** (Vector Embeddings via ChromaDB)
2.  **Structural Memory** (Knowledge Graph via NetworkX)
3.  **Episodic Memory** (Relational Session History via SQLite)

This multi-layered approach allows the system to understand context not just by keyword similarity, but by the relationships between entities and the temporal relevance of information.

## 2. System Architecture

The system follows a client-server architecture with a React-based frontend and a Python Flask backend.

### 2.1 Backend (Python/Flask)
The core logic resides in the `backend/` directory. It uses `Google Gemini 2.0 Flash` as the underlying Large Language Model (LLM).

**Key Design Pattern: specialized API Keys**
To handle rate limits and optimize performance, the system uses a distributed key strategy across 4 distinct operations:
- **`main`**: Generates the final user response.
- **`classifier`**: Classifies incoming queries to determine intent.
- **`entity_user`**: Extracts entities from user messages.
- **`entity_model`**: Extracts entities from model responses for graph population.

### 2.2 Frontend (React/Vite)
The user interface is built with:
- **Framework**: React 18 + Vite
- **Language**: TypeScript
- **Styling**: Tailwind CSS + shadcn/ui
- **State Management**: React Hooks
- **Routing**: React Router

## 3. The Fusion Layer (Core Innovation)

The defining feature of VEGAPUNK is the **Fusion Layer** (`backend/fusion_layer.py`). This module is responsible for ranking retrieved memory fragments to provide the most relevant context to the LLM.

### 3.1 Ranking Algorithm
The relevance score ($S$) for any given piece of information is calculated using a weighted formula:

$$ S_{final} = (\alpha \times S_{semantic}) + (\beta \times S_{structural}) + (\gamma \times S_{recency}) $$

Where the default weights are:
- $\alpha$ (Alpha) = **0.4**: Semantic Similarity (how close is the meaning?)
- $\beta$ (Beta) = **0.4**: Structural Connection (how connected is it in the graph?)
- $\gamma$ (Gamma) = **0.2**: Recency (how new is the information?)

### 3.2 Configuration
The `FusionConfig` class allows tuning of:
- **Expansion Hops**: How many steps to traverse in the graph (default: 1).
- **Recency Decay**: The half-life of information relevance (default: 168 hours / 1 week).
- **Top K**: Number of context chunks effectively retrieved (default: 10).

## 4. Memory Subsystems

### 4.1 Graph Store (`backend/graph_store.py`)
Utilizes `networkx` for in-memory graph operations backed by SQLite persistence.
- **Nodes**: `Concept`, `Entity`, `Document`, `Alias`, `Message`.
- **Relationships**: Weighted edges define the strength of connections.
    - `CAUSES` (1.0)
    - `CONTAINS` (0.9)
    - `MENTIONS` (0.8)
    - `RELATED_TO` (0.7)
    - `PART_OF` (0.6)

### 4.2 Vector Store (`backend/vector_store.py`)
Uses `ChromaDB` with `sentence-transformers/all-MiniLM-L6-v2` embeddings.
- Stores semantic representations of all messages and Extracted Entities.
- Enables "fuzzy" matching of concepts even when different words are used.

### 4.3 Relational Database (`backend/database.py`)
Standard SQLite implementation for:
- Session management.
- Raw message logs.
- Chat history retrieval.

## 5. API Reference

### POST `/api/chat`
The primary endpoint for all interactions.

**Request Body:**
```json
{
  "message": "<p>User input here...</p>",
  "images": ["base64_string_optional"]
}
```

**Response:**
```json
{
  "reply": "AI generated response...",
  "sources": [ ... ]
}
```

## 6. Environment Configuration

The application requires the following environment variables in `.env`:

| Variable | Description |
| :--- | :--- |
| `GEMINI_API_KEY_1` | Primary LLM Key |
| `GEMINI_API_KEY_2` | Query Classifier Key |
| `GEMINI_API_KEY_3` | User Entity Extractor Key |
| `GEMINI_API_KEY_4` | Model Entity Extractor Key |
| `PORT` | Backend Server Port (default: 5000) |
| `DEBUG` | Flask Debug Mode (True/False) |


## 7. Setup & Execution

### Backend
1. Install dependencies:
   ```bash
   pip install -r backend/requirements.txt
   ```
2. Start the server:
   ```bash
   python backend/app.py
   ```

### Frontend
1. Install dependencies:
   ```bash
   npm install
   ```
2. Start development server:
   ```bash
   npm run dev
   ```
