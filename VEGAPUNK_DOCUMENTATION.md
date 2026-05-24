# VEGAPUNK: Advanced Hybrid Memory AI Chat System - Comprehensive Technical Documentation

## Table of Contents
1.  **Introduction & Overview**
    *   1.1 What is VEGAPUNK?
    *   1.2 Core Philosophy: Beyond Traditional RAG
    *   1.3 High-Level System Architecture
2.  **The Three Pillars of Hybrid Memory**
    *   2.1 Episodic Memory (Relational SQLite)
    *   2.2 Semantic Memory (ChromaDB Vector Store)
    *   2.3 Structural Memory (NetworkX Knowledge Graph)
3.  **The Secret Sauce: The Fusion Layer**
    *   3.1 The Fusion Algorithm Explained
    *   3.2 Weighting and Tuning ($S_{final}$)
    *   3.3 Recency Decay Mechanics
4.  **System Components & Implementation Details**
    *   4.1 The Frontend (React/Vite/Tailwind)
        *   4.1.1 Main Chat Modality
        *   4.1.2 Sub Chat (Satellite) Modality
        *   4.1.3 Markdown and Rich Media Handling
    *   4.2 The Backend (Python/Flask)
        *   4.2.1 Agent Orchestration & The "Middleman" Classifier
        *   4.2.2 Distributed API Key Management
        *   4.2.3 Tool-Calling Flow
5.  **Concurrency & Resource Management**
    *   5.1 Parallel Retrieval Strategy
    *   5.2 Load Balancing & Rate Limit Mitigation
6.  **Architecture Flow: Lifecycle of a Request**
    *   6.1 Ingestion & Parsing
    *   6.2 Classification & Context Windowing
    *   6.3 Decision & Parallel Retrieval
    *   6.4 Synthesis & Response
7.  **Key Code Walkthroughs**
    *   7.1 App Initialization and Routing (`app.py`)
    *   7.2 The Fusion Layer Calculation (`fusion_layer.py`)
    *   7.3 Parallel Execution (`agent.py`)
    *   7.4 Graph Relationships & Weights (`graph_store.py`)
8.  **Why These Choices Were Made (Design Rationale)**
    *   8.1 Why Hybrid Memory over Pure Vector Search?
    *   8.2 Why SQLite over PostgreSQL/MySQL?
    *   8.3 Why Google Gemini 2.0 Flash?
9.  **Deployment & Environment Configuration**
10. **Conclusion**

---

## 1. Introduction & Overview

### 1.1 What is VEGAPUNK?
VEGAPUNK is a sophisticated, full-stack AI chat application designed to simulate human-like recall and context understanding. At its core, it leverages the Google `gemini-2.0-flash-exp` (or `gemini-2.5-flash`) Large Language Model (LLM). However, the LLM is merely the engine; the true innovation of VEGAPUNK lies in its **Hybrid Memory Architecture**.

It provides a React-based frontend featuring distinct "Main Chat" and "Sub Chat" interfaces, allowing users to maintain a primary contextual timeline while exploring tangential thoughts without polluting the main memory stream.

### 1.2 Core Philosophy: Beyond Traditional RAG
Traditional Retrieval-Augmented Generation (RAG) systems rely almost exclusively on semantic search—converting text into high-dimensional vectors and finding nearest neighbors based on cosine similarity or Euclidean distance. While effective for simple Q&A, pure RAG fails at:
*   **Logical leaps:** Recognizing that if A causes B, and B causes C, A is related to C.
*   **Temporal relevance:** Differentiating between outdated facts and recent, critical context.
*   **Structural context:** Understanding the strict relational hierarchy of concepts (e.g., "React" is a part of the "Frontend Stack").

VEGAPUNK solves this by introducing a multi-layered memory ecosystem that understands context through semantic similarity, relational graph connections, and temporal episodic records.

### 1.3 High-Level System Architecture
The architecture is inherently client-server:
*   **Frontend:** A responsive Single Page Application (SPA) built with React 18, Vite, TypeScript, Tailwind CSS, and `shadcn/ui` components.
*   **Backend:** A Python Flask REST API that orchestrates the LLM, manages the memory stores, and executes the Fusion Layer logic.

---

## 2. The Three Pillars of Hybrid Memory

VEGAPUNK manages state and context across three distinct but synergistic persistence layers.

### 2.1 Episodic Memory (Relational SQLite)
*   **Technology:** SQLite (`vegapunk_memory.db`) via `database.py`.
*   **Purpose:** To act as the system's short-term and explicit historical memory.
*   **Mechanism:** It records the exact conversational flow, managing user sessions, raw message text, timestamps, and explicit chat history. When the agent needs to know exactly what was said 3 messages ago, it queries the Episodic Memory. It provides the strict temporal timeline that semantic searches often scramble.

### 2.2 Semantic Memory (ChromaDB Vector Store)
*   **Technology:** `ChromaDB` running locally, utilizing the HuggingFace `sentence-transformers/all-MiniLM-L6-v2` model (`vector_store.py`).
*   **Purpose:** To provide "fuzzy" conceptual matching.
*   **Mechanism:** Every user message and system response is mapped into a vector space. If a user asks about "browser rendering issues," the semantic memory can retrieve past conversations about "DOM repaints" or "CSS layout thrashing," even if the exact keywords were never used. This handles the broad strokes of topical relevance.

### 2.3 Structural Memory (NetworkX Knowledge Graph)
*   **Technology:** `NetworkX` (in-memory graph operations) backed by SQLite for persistence (`graph_store.py`).
*   **Purpose:** To map explicit, logical relationships between entities and concepts.
*   **Mechanism:** This introduces true "Graph RAG." The system continuously runs entity extraction on both user and model messages in the background. It structures knowledge into Nodes (e.g., `Concept`, `Entity`, `Document`) and Edges representing weighted relationships:
    *   `CAUSES` (1.0 weight)
    *   `CONTAINS` (0.9 weight)
    *   `MENTIONS` (0.8 weight)
    *   `RELATED_TO` (0.7 weight)
    *   `PART_OF` (0.6 weight)
    
    This allows the system to traverse multi-hop logical paths that a vector similarity search would completely miss.

---

## 3. The Secret Sauce: The Fusion Layer

The core innovation of VEGAPUNK is the **Fusion Layer** (`backend/fusion_layer.py`), which acts as the arbiter between the three memory pillars. When a query is made, it doesn't just return the top vector matches; it retrieves candidates from all systems and ranks them holistically.

### 3.1 The Fusion Algorithm Explained
The ranking algorithm calculates a final relevance score ($S_{final}$) for every potential piece of context retrieved from the databases. It evaluates:
1.  **Semantic Score ($S_{semantic}$):** How close is the meaning based on ChromaDB's L2 distance normalization?
2.  **Structural Score ($S_{structural}$):** How highly connected and important is this node in the NetworkX graph? How strong are the edge weights leading to it?
3.  **Recency Score ($S_{recency}$):** How new is this information?

### 3.2 Weighting and Tuning ($S_{final}$)
The formula is a weighted sum:

$$ S_{final} = (\alpha \times S_{semantic}) + (\beta \times S_{structural}) + (\gamma \times S_{recency}) $$

By default, VEGAPUNK is tuned with the following `FusionConfig`:
*   $\alpha$ (Alpha) = **0.4**: Semantic representation is highly important for general language tasks.
*   $\beta$ (Beta) = **0.4**: Structural connections are equally important to ensure logical consistency and prevent hallucination.
*   $\gamma$ (Gamma) = **0.2**: Recency is treated as a tie-breaker or context-refiner, preventing the system from locking onto old paradigms if a user has recently updated their instructions.

### 3.3 Recency Decay Mechanics
Information age is actively managed to prevent "context saturation" where old, frequently discussed topics drown out new instructions. The system uses a linear decay mechanism:

```python
# From fusion_layer.py
age_hours = (now - timestamp).total_seconds() / 3600
decay = age_hours / cfg.recency_decay_hours # Default half-life: 168 hours (1 week)
recency_score = max(cfg.recency_min_score, 1.0 - decay)
```
Data older than one week hits the `min_score_threshold` (e.g., 0.1), meaning it will only be retrieved if its semantic or structural scores are overwhelmingly high.

---

## 4. System Components & Implementation Details

### 4.1 The Frontend (React/Vite/Tailwind)
The UI is engineered for complex knowledge work.

#### 4.1.1 Main Chat Modality (`src/components/MainChat.tsx`)
The primary interface where interactions are considered "canon." Every message sent here is processed by all three backend memory systems (Episodic, Semantic, Structural). This builds the long-term context of the user's project or persona.

#### 4.1.2 Sub Chat (Satellite) Modality (`src/components/SubChat.tsx`)
A critical feature for complex debugging or tangental exploration. Sub Chats inherit the context and `session_id` of the Main Chat, allowing the LLM to read the entire history. However, interactions within a Sub Chat **do not save** new memory (`save_to_memory=False` in `agent.py`).
*   **Why?** This allows users to paste massive error logs, go down a rapid-fire debugging rabbit hole, or ask "stupid questions" without that chaotic data permanently polluting the semantic and structural graph of the main project.

#### 4.1.3 Markdown and Rich Media Handling (`src/components/ChatMessage.tsx`)
Messages are rendered using `remark-gfm` for tables and code blocks. VEGAPUNK supports multimodal input. When a user pastes an image, the frontend embeds it as a base64 string within an `<img>` HTML tag inside the `message` payload. The backend parses this, extracts the text, and routes the byte data to Gemini's vision pipeline.

### 4.2 The Backend (Python/Flask)
The backend acts as an orchestrator rather than just a pass-through proxy.

#### 4.2.1 Agent Orchestration & The "Middleman" Classifier
Before a user's prompt ever reaches the main text generation model, it goes through a lightweight classifier (`_classify_query` in `agent.py`).
*   **Categories Output:** `RECENT_CHAT`, `GENERAL_KNOWLEDGE`, `TOPIC_SEARCH`, etc.
*   **Action Taken:** If classified as `RECENT_CHAT`, the context window restricts episodic history fetch to only the last few messages to save tokens and improve latency. If classified as `TOPIC_SEARCH`, it aggressively queries the graph and vector stores.

#### 4.2.2 Distributed API Key Management
To handle heavy extraction loads, background processing, and aggressive rate limits imposed by LLM providers, VEGAPUNK requires passing an array of 4 distinct API keys.
*   `main`: Synthesizes final response.
*   `classifier`: Fast query categorizations.
*   `entity_user`: Background extraction of graph nodes from user prompts.
*   `entity_model`: Background extraction of graph nodes from the LLM outputs.

#### 4.2.3 Tool-Calling Flow
VEGAPUNK utilizes the Gemini Tool-Calling API. The LLM is provided with a description of the `_hybrid_search_tool`. 
1. The LLM analyzes the query and the short-term episodic history provided in the prompt.
2. It decides: "Can I answer this directly, or do I need to look into long-term memory?"
3. If it chooses the latter, it halts generation, outputs a structured JSON function call, the backend executes the Hybrid Search in Python, returns the Fusion Ranked data as a JSON string back to the LLM, and the LLM synthesizes the final response.

---

## 5. Concurrency & Resource Management

### 5.1 Parallel Retrieval Strategy
A pure sequential search (Query SQL -> Query Chroma -> Query NetworkX -> Rank) is far too slow for real-time chat. VEGAPUNK heavily utilizes Python's `concurrent.futures.ThreadPoolExecutor` to execute I/O bound queries simultaneously.

```python
# From agent.py orchestration
with ThreadPoolExecutor(max_workers=3) as executor:
    sql_future = executor.submit(self.db.search_messages, session_id, query, hours_ago=168)
    vector_future = executor.submit(self.vector_store.semantic_search, query, session_id)
    graph_future = executor.submit(self._graph_rag_search, query, session_id, fusion_config)
    
    # Execution is non-blocking until the results are needed for the Fusion Layer
```
This reduces retrieval latency from ~1200ms (sequential) down to the latency of the slowest individual query (usually ~400ms for ChromaDB depending on hardware).

### 5.2 Load Balancing & Rate Limit Mitigation
The extraction of entities for the Knowledge Graph is token-intensive. By routing `entity_user` and `entity_model` extraction tasks to dedicated API keys (`GEMINI_API_KEY_3` and `GEMINI_API_KEY_4`), the system ensures that the critical `main` key never hits 429 Too Many Requests errors. Furthermore, the Sub Chat modality implicitly saves tokens and API calls by bypassing the entity extraction pipelines entirely.

---

## 6. Architecture Flow: Lifecycle of a Request

Below is step-by-step breakdown of how data moves through VEGAPUNK.

### 6.1 Ingestion & Parsing
1.  User clicks "Send" in `MainChat.tsx`.
2.  An HTTP POST is made to `/api/chat/main` with payload `{ "message": "HTML content..." }`.
3.  `app.py` receives the request. It uses Regex to strip `<img>` tags to extract pure text for the prompt, and extracts base64 data to decode into raw bytes if multimodal vision is required.

### 6.2 Classification & Context Windowing
4.  `app.py` passes the data to `VegapunkAgent.chat()`.
5.  The agent calls the classifier LLM using Key 2. The query "How do I fix the React Three Fiber bug we discussed yesterday?" is classified as `TOPIC_SEARCH`.
6.  The agent prepares the initial prompt, injecting the last $N$ lines of Episodic SQLite history to give immediate context.

### 6.3 Decision & Parallel Retrieval
7.  The prompt is sent to the main Gemini model (Key 1) with the `_hybrid_search_tool` available.
8.  Gemini recognizes "discussed yesterday" and issues a function call to search memory.
9.  The `ThreadPoolExecutor` spins up 3 threads. Vector, Graph, and SQL stores are queried in parallel.
10. Results are dumped into the `FusionLayer.rank_and_fuse()` method.
11. The mathematics (Alpha, Beta, Gamma) runs, sorting the disparate data types into a single, unified list of the top 10 most relevant context chunks.

### 6.4 Synthesis & Response
12. The ranked context is formatted into a "System Context Observation" string and passed back to Gemini as the tool response.
13. Gemini synthesizes the final answer using the highly-curated context array.
14. The response string is returned to `app.py`.
15. *Background Spawn:* Before returning to the user, `app.py` spawns background threads using Keys 3 & 4 to run Entity Extraction on the new user prompt and the LLM response, updating the NetworkX graph and ChromaDB vectors asynchronously.
16. JSON is returned to the frontend. `MainChat.tsx` renders the Markdown.

---

## 7. Key Code Walkthroughs

### 7.1 App Initialization and Routing (`app.py`)
This snippet demonstrates the load distribution implementation and how the backend acts as a centralized orchestrator.

```python
# app.py snippet
# Build API keys dictionary for load balancing across 4 keys
api_keys = {
    'main': GEMINI_API_KEY_1,           # Main LLM response
    'classifier': GEMINI_API_KEY_2,     # Query classification
    'entity_user': GEMINI_API_KEY_3,    # Entity extraction (user messages)
    'entity_model': GEMINI_API_KEY_4,   # Entity extraction (model responses)
}

# Initialize agent and database connections ONCE (Singleton pattern)
if GEMINI_API_KEY_1:
    print("🧠 Initializing VEGAPUNK Agent with Hybrid Memory...")
    db = VegapunkDatabase()
    vector_store = VegapunkVectorStore()
    agent = VegapunkAgent(api_keys=api_keys, db=db, vector_store=vector_store)

def extract_image_data(html_content):
    """Extract image data from HTML content to support visual model inputs"""
    img_match = re.search(r'<img[^>]+src="data:image/[^;]+;base64,([^"]+)"', html_content)
    if img_match:
        return base64.b64decode(img_match.group(1))
    return None
```

### 7.2 The Fusion Layer Calculation (`fusion_layer.py`)
This is the mathematical core of the system. Notice how structural values (NetworkX pagerank/centrality) and semantic values (Chroma L2 distance) are normalized before the weighted sum is applied.

```python
# fusion_layer.py snippet
@dataclass
class FusionConfig:
    alpha: float = 0.4  # Semantic weight
    beta: float = 0.4   # Structural weight
    gamma: float = 0.2  # Recency weight
    recency_decay_hours: float = 168.0  # 1 week

# Inside rank() loop:
# Recency calculation
age_hours = (now - timestamp).total_seconds() / 3600
decay = age_hours / cfg.recency_decay_hours
recency_score = max(cfg.recency_min_score, 1.0 - decay)

# Note: semantic_score and structural_score are normalized between 0-1 prior to this step
final_score = (
    cfg.alpha * semantic_score +
    cfg.beta * structural_score +
    cfg.gamma * recency_score
)
```

### 7.3 Graph Relationships & Weights (`graph_store.py`)
This defines how the Knowledge Graph interprets logical importance. An edge denoting cause/effect is mathematically more valuable than one denoting a passing mention.

```python
# graph_store.py snippet
# Relationship weights for ranking (higher = more important during structural scoring)
RELATION_WEIGHTS = {
    RelationType.CAUSES.value: 1.0,
    RelationType.CONTAINS.value: 0.9,
    RelationType.MENTIONS.value: 0.8,
    RelationType.RELATED_TO.value: 0.7,
    RelationType.PART_OF.value: 0.6,
    RelationType.DERIVED_FROM.value: 0.5,
    RelationType.ALIAS_OF.value: 0.4,
}
```

---

## 8. Why These Choices Were Made (Design Rationale)

### 8.1 Why Hybrid Memory over Pure Vector Search?
Pure vector search (ChromaDB alone) suffers from the "Lost in the Middle" problem and "Conceptual Drift." If a user types "Make it a button", a vector search might pull up a 3-month-old conversation about a CSS button, missing that the user is currently discussing an IoT physical button setup. 

By injecting Episodic (short-term SQLite history restricts the immediate context to "IoT devices") and Structural (Graph RAG notes `IoT Device -> CONTAINS -> Button Component`), the LLM receives context that is temporally and logically accurate, not just syntactically similar.

### 8.2 Why SQLite over PostgreSQL/MySQL?
VEGAPUNK is designed as a personal, deeply contextual AI agent, not a multi-tenant SaaS application (currently). SQLite requires zero external infrastructure, runs entirely in-process, and allows the `vegapunk_memory.db` and the ChromaDB SQLite mappings to exist locally. This aligns with the principles of local-first AI experimentation and rapid prototyping.

### 8.3 Why Google Gemini 2.0 Flash?
Gemini 2.0 Flash provides a massive context window (up to 1M tokens), native multimodal support (vision processing is handled on the same API call as text generation), and extremely fast time-to-first-token. Crucially, its Tool-Calling reliability is high, which is strictly required for the dynamic Hybrid Retrieval routing logic.

---

## 9. Deployment & Environment Configuration

### Prerequisites
*   Node.js & npm (Frontend)
*   Python 3.10+ (Backend)
*   4x Google Gemini API Keys.

### Environment Setup (`.env`)
```bash
GEMINI_API_KEY_1=your_main_key
GEMINI_API_KEY_2=your_classifier_key
GEMINI_API_KEY_3=your_user_entity_key
GEMINI_API_KEY_4=your_model_entity_key
PORT=5000
DEBUG=True
```

### Installation 
**Frontend:**
```bash
cd <project_root>
npm install
npm run dev
```

**Backend:**
```powershell
cd backend
python -m venv vegapunk
.\vegapunk\Scripts\activate
pip install -r requirements.txt
python app.py
```

---

## 10. Conclusion

VEGAPUNK represents a significant architectural leap in local AI memory management. By decoupling the memory persistence layer from the standard vector-only RAG paradigm and introducing a mathematics-driven Fusion Layer to balance Semantic, Structural, and Episodic memory mathematically, it creates an AI interaction model that feels functionally closer to human recall. The implementation of Satellite Sub Chats further refines the UX, protecting the integrity of the Knowledge Graph while allowing uninhibited complex debugging workflows.