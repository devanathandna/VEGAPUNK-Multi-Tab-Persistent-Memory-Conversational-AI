from groq import Groq
from typing import List, Dict, Optional
from database import VegapunkDatabase
from vector_store import VegapunkVectorStore
from graph_store import GraphStore, NodeType, RelationType
from fusion_layer import FusionLayer, FusionConfig
from entity_extractor import EntityExtractor, ConversationExtractor, ExtractionMode
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ── Model identifiers ──────────────────────────────────────────────────────────
MAIN_MODEL       = "llama-3.3-70b-versatile"      # 70B versatile for main chat responses
CLASSIFIER_MODEL = "llama-3.1-8b-instant"       # Fast 8B for classification
ENTITY_MODEL     = "llama-3.3-70b-versatile"    # 70B for accurate entity extraction

# ── Payload cap (chars) sent to LLM as tool result ────────────────────────────
MAX_TOOL_PAYLOAD_CHARS = 4000

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are VEGAPUNK, a helpful and knowledgeable AI assistant with advanced hybrid memory.

When answering questions:
1. If the question relates to past conversations or stored knowledge, use the 'search_database' tool to retrieve relevant context.
2. Pay attention to 'graph_context' in search results — it contains:
   - 'nodes': Direct semantic matches
   - 'neighbors': Related concepts from the knowledge graph
   - 'relations': How concepts connect (e.g., "A CAUSES B")
   - 'context_texts': Actual content from matched nodes
3. Use graph relationships to give more complete, well-reasoned answers.
4. Only answer from your own knowledge if the query is pure general knowledge.

Your memory and knowledge graph are your primary tools for personalized, contextual responses."""

# ── Tool schema (OpenAI-compatible for Groq) ────────────────────────────────────
SEARCH_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_database",
            "description": "Search hybrid memory: SQL episodic history, ChromaDB semantic vectors, and NetworkX knowledge graph.",
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "The current chat session ID"
                    },
                    "query": {
                        "type": "string",
                        "description": "The search query to look up in memory"
                    },
                    "query_type": {
                        "type": "string",
                        "enum": ["GENERAL_KNOWLEDGE", "RECENT_CHAT", "KEYWORD_SEARCH", "TOPIC_SEARCH"],
                        "description": "Query type to optimize retrieval strategy"
                    }
                },
                "required": ["session_id", "query", "query_type"]
            }
        }
    }
]

class VegapunkAgent:
    def __init__(self, api_keys: Dict[str, str], db: VegapunkDatabase, vector_store: VegapunkVectorStore, graph_store: Optional[GraphStore] = None):
        """
        Initialize the VEGAPUNK agent with Groq clients for load balancing.

        Args:
            api_keys: Dict with keys 'main', 'classifier', 'entity_user', 'entity_model'.
                     Each maps to a Groq API key to distribute rate limits.
        """
        self.api_keys = api_keys
        self.main_api_key = api_keys.get('main') or api_keys.get('default')
        self.classifier_api_key = api_keys.get('classifier') or self.main_api_key
        self.entity_user_api_key = api_keys.get('entity_user') or self.main_api_key
        self.entity_model_api_key = api_keys.get('entity_model') or self.main_api_key

        print(f"🔑 API Key Distribution:")
        print(f"   Main Model:        ...{self.main_api_key[-8:]}")
        print(f"   Classifier:        ...{self.classifier_api_key[-8:]}")
        print(f"   Entity (User):     ...{self.entity_user_api_key[-8:]}")
        print(f"   Entity (Model):    ...{self.entity_model_api_key[-8:]}")

        self.db = db
        self.vector_store = vector_store

        # Initialize Groq clients (one per role for rate-limit distribution)
        self.main_client = Groq(api_key=self.main_api_key)
        self.classifier_client = Groq(api_key=self.classifier_api_key)

        # Initialize Graph RAG components
        self.graph_store = graph_store or GraphStore(db_path=db.db_path)
        self.fusion_layer = FusionLayer(self.graph_store, self.vector_store)
        self.use_graph_rag = True

        # Initialize Entity Extractors — 70B model for accuracy, separate keys
        self.entity_extractor_user = EntityExtractor(self.entity_user_api_key, model=ENTITY_MODEL)
        self.entity_extractor_model = EntityExtractor(self.entity_model_api_key, model=ENTITY_MODEL)
        self.conversation_extractor = ConversationExtractor(self.entity_user_api_key)
        self.auto_extract_entities = True

    def _classify_query(self, query: str) -> str:
        """
        Groq-based query classification to determine search strategy.
        Returns: GENERAL_KNOWLEDGE, RECENT_CHAT, KEYWORD_SEARCH, or TOPIC_SEARCH
        """
        classification_prompt = f"""Classify this user query into ONE category:

Query: "{query}"

Categories:
- GENERAL_KNOWLEDGE: Questions about facts or concepts not needing memory
- RECENT_CHAT: References to recent/previous messages ("just said", "earlier", "before")
- KEYWORD_SEARCH: Explicit search requests ("find", "show me", "search for")
- TOPIC_SEARCH: Questions about past conversations or stored knowledge

Examples:
"What is Python?" -> GENERAL_KNOWLEDGE
"What did you just say?" -> RECENT_CHAT
"Find all messages about the hello project" -> KEYWORD_SEARCH
"What result did we get for the hello project?" -> TOPIC_SEARCH

Respond with ONLY the category name.
Category:"""

        try:
            print(f"\n📤 [CLASSIFIER] Groq API CALL")
            print(f"   Model: {CLASSIFIER_MODEL}")
            print(f"   Query: {query[:60]}...")

            response = self.classifier_client.chat.completions.create(
                model=CLASSIFIER_MODEL,
                messages=[{"role": "user", "content": classification_prompt}],
                max_tokens=20,
                temperature=0
            )
            raw = response.choices[0].message.content.strip().upper()

            for cat in ["GENERAL_KNOWLEDGE", "RECENT_CHAT", "KEYWORD_SEARCH", "TOPIC_SEARCH"]:
                if cat in raw:
                    print(f"🧠 Query classified as: {cat}")
                    return cat

            print(f"⚠️ Unexpected classification '{raw}', defaulting to TOPIC_SEARCH")
            return "TOPIC_SEARCH"

        except Exception as e:
            print(f"⚠️ Classification error: {e}, defaulting to TOPIC_SEARCH")
            return "TOPIC_SEARCH"


    def _hybrid_search_tool(self, session_id: str, query: str, query_type: str) -> str:
        """
        Executes a hybrid search by combining:
        1. Recent structured history (SQL)
        2. Long-term semantic memory (Vector)
        3. Knowledge graph context (Graph RAG) - NEW!
        
        All searches run IN PARALLEL for performance.
        """
        print(f"🤖 Agent executing tool: search_database with query: '{query}'")
        
        # Determine fusion config based on query type
        fusion_config = self._get_fusion_config(query_type)
        
        # Execute searches in parallel
        sql_results = []
        vector_results = []
        graph_context = None
        
        try:
            with ThreadPoolExecutor(max_workers=3) as executor:
                # Submit all searches simultaneously
                sql_future = executor.submit(
                    self.db.search_messages,
                    session_id,
                    query,
                    hours_ago=168  # 7 days
                )
                vector_future = executor.submit(
                    self.vector_store.semantic_search, 
                    query, 
                    session_id
                )
                
                # Graph RAG search (if enabled)
                graph_future = None
                if self.use_graph_rag:
                    graph_future = executor.submit(
                        self._graph_rag_search,
                        query,
                        session_id,
                        fusion_config
                    )
                
                # Collect results as they complete
                futures = [sql_future, vector_future]
                if graph_future:
                    futures.append(graph_future)
                
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if future == sql_future:
                            sql_results = result
                            print(f"✓ SQL search completed: {len(sql_results)} results")
                        elif future == vector_future:
                            vector_results = result
                            print(f"✓ Vector search completed: {len(vector_results)} results")
                        elif future == graph_future:
                            graph_context = result
                            print(f"✓ Graph RAG completed: {len(graph_context.get('nodes', []))} seed nodes, {len(graph_context.get('neighbors', []))} neighbors")
                    except Exception as e:
                        print(f"⚠️ Search error: {e}")
        
        except Exception as e:
            print(f"❌ Parallel search execution failed: {e}")
            # Fallback to sequential search
            try:
                sql_results = self.db.search_messages(session_id, query, hours_ago=168)
            except Exception as sql_err:
                print(f"❌ SQL search failed: {sql_err}")
            
            try:
                vector_results = self.vector_store.semantic_search(query, session_id)
            except Exception as vec_err:
                print(f"❌ Vector search failed: {vec_err}")
        
        # Combine and de-duplicate results
        # NOTE: SQL rows use 'id' (int PK). Vector results use no 'id' — fall back to content hash.
        all_results = []
        seen_ids = set()

        # Add SQL results (keyed by integer PK 'id')
        for result in sql_results:
            result_id = result.get('id')
            if result_id is not None and result_id not in seen_ids:
                all_results.append({**result, 'source': 'SQL'})
                seen_ids.add(result_id)

        # Add vector results — semantic_search() returns 'content'/'similarity_score', not 'id'.
        # Use content as dedup key so SQL hits are not duplicated.
        for result in vector_results:
            content = result.get('content', '')
            dedup_key = f"vec::{hash(content)}"
            if dedup_key not in seen_ids:
                all_results.append({**result, 'source': 'Vector'})
                seen_ids.add(dedup_key)

        # ──────────────────────────────────────────────────────────────────────
        # 📊 RETRIEVAL LOG — what each memory layer returned
        # ──────────────────────────────────────────────────────────────────────
        print("\n" + "█"*70)
        print("📊  RETRIEVED CONTEXT BEING SENT TO LLM")
        print("█"*70)

        # ── 1. SQLite results ─────────────────────────────────────────────────
        print(f"\n┌── 📦 SQLite (Episodic History) — {len(sql_results)} result(s)")
        if sql_results:
            for i, r in enumerate(sql_results, 1):
                role  = r.get('role', '?').upper()
                ts    = str(r.get('timestamp', ''))[:19]
                snip  = str(r.get('content', ''))[:200].replace('\n', ' ')
                print(f"│   [{i}] [{role}] {ts}")
                print(f"│       {snip}{'...' if len(str(r.get('content',''))) > 200 else ''}")
        else:
            print("│   (no results)")

        # ── 2. Vector / ChromaDB results ──────────────────────────────────────
        print(f"├── 🔍 Vector Store (ChromaDB Semantic) — {len(vector_results)} result(s)")
        if vector_results:
            for i, r in enumerate(vector_results, 1):
                score = r.get('similarity_score', r.get('score', r.get('distance', '?')))
                snip  = str(r.get('content', r.get('text', ''))).strip()[:200].replace('\n', ' ')
                print(f"│   [{i}] similarity={score:.4f}" if isinstance(score, float) else f"│   [{i}] similarity={score}")
                print(f"│       {snip}{'...' if len(str(r.get('content', r.get('text','')))) > 200 else ''}")
        else:
            print("│   (no results)")

        # ── 3. NetworkX / GraphRAG context ────────────────────────────────────
        gc_nodes     = graph_context.get('nodes', [])     if graph_context else []
        gc_neighbors = graph_context.get('neighbors', []) if graph_context else []
        gc_relations = graph_context.get('relations', []) if graph_context else []
        gc_texts     = graph_context.get('context_texts', []) if graph_context else []
        print(f"├── 🕸  NetworkX (Graph RAG) — {len(gc_nodes)} seed node(s), "
              f"{len(gc_neighbors)} neighbor(s), {len(gc_relations)} relation(s)")
        if gc_nodes:
            for n in gc_nodes[:5]:
                print(f"│   Seed  : {n}")
        if gc_neighbors:
            for n in gc_neighbors[:5]:
                print(f"│   Nbr   : {n}")
        if gc_relations:
            for rel in gc_relations[:5]:
                print(f"│   Rel   : {rel}")
        if gc_texts:
            print(f"│   Context texts ({len(gc_texts)}):")
            for t in gc_texts[:3]:
                snip = str(t)[:200].replace('\n', ' ')
                print(f"│     • {snip}")
        if not gc_nodes and not gc_texts:
            print("│   (no graph context)")

        # Build final response with Graph RAG context
        response_data = {
            "search_results": all_results,
            "graph_context": graph_context
        }

        # Serialise and cap payload to avoid overwhelming the 8B model
        tool_results_full = json.dumps(response_data, indent=2, default=str)
        if len(tool_results_full) > MAX_TOOL_PAYLOAD_CHARS:
            tool_results = tool_results_full[:MAX_TOOL_PAYLOAD_CHARS] + "\n... [TRUNCATED for LLM context limit]"
            print(f"└── ⚠️  Payload capped: {len(tool_results_full)} → {MAX_TOOL_PAYLOAD_CHARS} chars")
        else:
            tool_results = tool_results_full
            print(f"└── ✅ Payload size: {len(tool_results)} chars (within limit)")

        print("█"*70 + "\n")

        return tool_results
    
    def _graph_rag_search(
        self, 
        query: str, 
        session_id: str,
        config: FusionConfig
    ) -> Dict:
        """
        Execute Graph RAG search using the fusion layer.
        
        Flow:
        1. Semantic search via Chroma
        2. Expand via knowledge graph
        3. Rank with fusion formula
        4. Build LLM context
        """
        try:
            print(f"\n🔍 [GRAPH RAG] Starting search for: '{query[:50]}...'")
            
            # Query the fusion layer
            results = self.fusion_layer.query(
                query_text=query,
                session_id=session_id,
                config=config
            )
            
            print(f"   Fusion results: {len(results)} nodes")
            
            if not results:
                print("   ⚠️ No fusion results, returning empty context")
                return {"nodes": [], "neighbors": [], "relations": [], "context_texts": []}
            
            # Build context for LLM
            context = self.fusion_layer.build_llm_context(
                results=results,
                include_relations=True,
                include_scores=False
            )
            
            # Add prompt-ready text context
            context["prompt_context"] = self.fusion_layer.build_prompt_context(results)
            
            # Add score information for debugging
            context["top_results"] = [
                {
                    "node_id": r.node_id,
                    "score": r.final_score,
                    "is_seed": r.is_seed,
                    "type": r.node_type
                }
                for r in results[:5]
            ]
            
            print(f"   Context built: {len(context.get('nodes', []))} nodes, {len(context.get('relations', []))} relations")
            
            return context
            
        except Exception as e:
            print(f"⚠️ Graph RAG search error: {e}")
            import traceback
            traceback.print_exc()
            return {"nodes": [], "neighbors": [], "relations": [], "context_texts": [], "error": str(e)}
    
    def _get_fusion_config(self, query_type: str) -> FusionConfig:
        """
        Get optimal fusion config based on query classification.
        """
        if query_type == "RECENT_CHAT":
            # Favor recency for recent conversation queries
            return FusionConfig(alpha=0.3, beta=0.2, gamma=0.5)
        elif query_type == "GENERAL_KNOWLEDGE":
            # Favor semantic for factual queries
            return FusionConfig(alpha=0.5, beta=0.4, gamma=0.1)
        elif query_type == "KEYWORD_SEARCH":
            # Balanced for explicit searches
            return FusionConfig(alpha=0.4, beta=0.4, gamma=0.2)
        else:  # TOPIC_SEARCH
            # Favor structure for topic exploration
            return FusionConfig(alpha=0.35, beta=0.45, gamma=0.2)

    def chat(self, session_id: str, user_message: str, image_data: Optional[bytes] = None, save_to_memory: bool = True) -> Dict:
        """
        Main chat method using native tool-calling agent architecture.
        """
        print("\n" + "="*80)
        print(f"[AGENT STEP 1] 📥 AGENT RECEIVED USER MESSAGE")
        print(f"               Message: {user_message}")
        print(f"               Session ID: {session_id}")
        print(f"               Has image: {image_data is not None}")
        print(f"               Save to memory: {save_to_memory}")
        print("="*80)
        
        # [AGENT STEP 3.5] 🧠 MIDDLEMAN CLASSIFICATION
        query_type = self._classify_query(user_message)
        print(f"               🧠 Middleman Classification: {query_type}")

        # DYNAMIC CONTEXT WINDOW: Decide how much history to fetch based on query type
        history_limit = 0
        if query_type == "RECENT_CHAT":
            history_limit = 10
            print(f"               📚 Context Strategy: RECENT_CHAT (Fetching last {history_limit} messages)")
        elif query_type == "GENERAL_KNOWLEDGE":
            history_limit = 0
            print(f"               📚 Context Strategy: GENERAL_KNOWLEDGE (Zero history for focus)")
        else: # TOPIC_SEARCH, KEYWORD_SEARCH
            history_limit = 2
            print(f"               📚 Context Strategy: MEMORY SEARCH (Minimal history for context)")

        # [AGENT STEP 2.5] 💾 SAVE USER MESSAGE TO MEMORY
        if save_to_memory:
            msg_id = self._add_message(session_id, 'user', user_message)
            # Add to vector store with session context
            self.vector_store.add_message(
                session_id=session_id,
                message_id=msg_id or 0,
                role='user',
                content=user_message
            )
            print(f"               ✓ User message saved to memory.")
            
            # [GRAPH RAG] Extract entities — fire in background thread so response is not blocked
            # NOTE: We always extract entities regardless of query_type.
            # Messages classified as GENERAL_KNOWLEDGE may still contain facts the user is asserting
            # (e.g. "the green ball has a red triangle") that must be persisted to the graph.
            if self.auto_extract_entities and len(user_message) >= 5:  # lowered from >20 so short facts like "A is the father of B" are captured
                # Capture variables NOW — thread holds them as closed-over locals, no DB lookup needed
                _text    = user_message
                _sid     = session_id
                _role    = 'user'
                _msg_id  = msg_id
                print(f"               🚀 Entity extraction fired in background thread (user | query_type={query_type})")
                t = threading.Thread(
                    target=self._extract_entities_async,
                    args=(_text, _sid, _role),
                    kwargs={'message_id': _msg_id},
                    daemon=True   # Thread dies if main process exits — no hanging threads
                )
                t.start()
            else:
                print(f"               ⏭️  Entity extraction skipped (message too short or auto_extract disabled)")

        # IMPORTANT: User message was already saved above, so history now contains it.
        # Fetch limit+1 and drop the last entry (current msg) to avoid sending it twice.
        print(f"\n[AGENT STEP 2] 📚 FETCHING SHORT-TERM MEMORY FROM DATABASE")
        if history_limit > 0:
            # Fetch limit+1 to account for the current user message that was just saved.
            # The current message is always the last row — drop it so it isn't sent twice.
            raw_history = self.db.get_session_history(session_id, limit=history_limit + 1)
            if raw_history and raw_history[-1].get('content') == user_message and raw_history[-1].get('role') == 'user':
                chat_history = raw_history[:-1]   # strip the current message
            else:
                chat_history = raw_history[-history_limit:]   # safety fallback
        else:
            chat_history = []

        # Build Groq-format messages: system + history + current user message
        # Inject the real session_id so the LLM doesn't hallucinate it in tool calls
        system_with_session = SYSTEM_PROMPT + f"\n\nCurrent session_id: {session_id} — always use this exact value when calling search_database."
        messages = [{"role": "system", "content": system_with_session}]
        for msg in chat_history:
            role = "user" if msg['role'] == "user" else "assistant"
            messages.append({"role": role, "content": msg['content']})
        messages.append({"role": "user", "content": user_message})

        # ================================================================
        # [LLM CALL 1] EXACT PAYLOAD SENT TO MAIN MODEL (FIRST CALL)
        # ================================================================
        print("\n" + "=" * 60)
        print("[LLM CALL 1] 📤 EXACT DATA SENT TO MAIN MODEL (1st call)")
        print(f"             Model          : {MAIN_MODEL}")
        print(f"             Temperature    : 0.3")
        print(f"             Max tokens     : 2048")
        print(f"             Tool choice    : auto")
        print(f"             Total messages : {len(messages)}")
        print("-" * 60)
        for i, m in enumerate(messages):
            role = m.get('role', '?').upper()
            content = m.get('content') or ''
            print(f"  [{i}] {role}")
            print(content)
            print("-" * 60)
        print("=" * 60)

        # Call Groq with tool support
        final_response_text = ""
        try:
            response = self.main_client.chat.completions.create(
                model=MAIN_MODEL,
                messages=messages,
                tools=SEARCH_TOOLS,
                tool_choice="auto",
                max_tokens=2048,
                temperature=0.3   # ↓ was 0.7 — lower temp reduces hallucinations
            )

            choice = response.choices[0]

            if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
                tool_call = choice.message.tool_calls[0]
                tool_args = json.loads(tool_call.function.arguments)

                print(f"Tool call: {tool_call.function.name} with args: {tool_args}")

                tool_result_json = self._hybrid_search_tool(
                    session_id=session_id,
                    query=tool_args.get('query', user_message),
                    query_type=query_type
                )

                # Append assistant tool_call message + tool result in OpenAI format
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments
                            }
                        }
                    ]
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result_json
                })

                # ================================================================
                # [LLM CALL 2] EXACT PAYLOAD SENT TO MAIN MODEL (SYNTHESIS CALL)
                # This is the call that uses SQL + ChromaDB + NetworkX data.
                # If hallucination happens, the tool_result below is the cause.
                # ================================================================
                print("\n" + "=" * 60)
                print("[LLM CALL 2] 📤 EXACT DATA SENT TO MAIN MODEL (synthesis call)")
                print(f"             Total messages : {len(messages)}")
                print("-" * 60)
                for i, m in enumerate(messages):
                    role = m.get('role', '?').upper()
                    if role == 'TOOL':
                        raw = m.get('content', '')
                        print(f"  [{i}] 🔴 TOOL RESULT (retrieval payload sent to LLM)")
                        print(f"       Total chars : {len(raw)}")
                        print("-" * 40)
                        try:
                            parsed = json.loads(raw)
                            sql_hits = parsed.get('search_results', [])
                            graph    = parsed.get('graph_context', {})
                            print(f"  ├─ SQL + Vector hits : {len(sql_hits)}")
                            for j, r in enumerate(sql_hits, 1):
                                src = r.get('source', '?')
                                print(f"  │  [{j}] [{src}]")
                                print(r.get('content', ''))
                                print("  │")
                            print(f"  ├─ Graph nodes     : {graph.get('nodes', [])}")
                            print(f"  ├─ Graph neighbors : {graph.get('neighbors', [])}")
                            print(f"  ├─ Graph relations :")
                            for rel in graph.get('relations', []):
                                print(f"       {rel}")
                            print(f"  └─ Context texts :")
                            for ct in graph.get('context_texts', []):
                                print(f"       {ct}")
                        except Exception:
                            print(raw)
                        print("-" * 40)
                    else:
                        content = m.get('content') or ''
                        print(f"  [{i}] {role}")
                        print(content)
                        print("-" * 60)
                print("=" * 60)

                # Get final synthesized response
                final_resp = self.main_client.chat.completions.create(
                    model=MAIN_MODEL,
                    messages=messages,
                    max_tokens=2048,
                    temperature=0.3   # ↓ was 0.7 — grounded synthesis
                )
                final_response_text = final_resp.choices[0].message.content
                print(f"               ✓ Synthesized final answer from tool results.")

            else:
                final_response_text = choice.message.content
                print(f"\n[AGENT STEP 6] ✅ DIRECT ANSWER")
                print(f"               Response: {str(final_response_text)[:100]}")

        except Exception as e:
            print(f"❌ Error during response generation: {e}")
            final_response_text = "I'm sorry, I encountered an error while processing your request."

        # [AGENT STEP 7] 💾 SAVING TO MEMORY
        if save_to_memory and final_response_text:
            print(f"\n[AGENT STEP 7] 💾 SAVING RESPONSE TO MEMORY")
            try:
                # 1. Save the assistant's response to the database
                msg_id = self._add_message(session_id, 'model', final_response_text)
                
                # 2. Update the vector store
                self.vector_store.add_message(
                    session_id=session_id,
                    message_id=msg_id or 0,
                    role='model',
                    content=final_response_text
                )
                
                print(f"               ✓ Response saved to memory.")
                
                # [GRAPH RAG] Extract entities from model response — fire in background thread
                # NOTE: Extract for all query types so factual responses are also stored in graph.
                if self.auto_extract_entities and len(final_response_text) > 50:
                    # Capture variables NOW before thread starts
                    _text    = final_response_text
                    _sid     = session_id
                    _role    = 'model'
                    _msg_id  = msg_id
                    print(f"               🚀 Entity extraction fired in background thread (model | query_type={query_type})")
                    t = threading.Thread(
                        target=self._extract_entities_async,
                        args=(_text, _sid, _role),
                        kwargs={'message_id': _msg_id},
                        daemon=True   # Thread dies if main process exits — no hanging threads
                    )
                    t.start()
                else:
                    print(f"               ⏭️  Entity extraction skipped (response too short or auto_extract disabled)")
                        
            except Exception as e:
                print(f"⚠️ Error saving to memory: {e}")

        return {
            "response": final_response_text,
            "context_used": query_type  # Include the query classification type
        }
        
    def _add_message(self, session_id: str, role: str, content: str) -> Optional[int]:
        """
        Saves a message to the database and optionally to the knowledge graph.
        
        Args:
            session_id: The ID of the current chat session.
            role: The role of the message sender ('user' or 'model').
            content: The text content of the message.
            
        Returns:
            The message ID if successful, None otherwise.
        """
        try:
            return self.db.save_message(session_id, role, content)
        except Exception as e:
            print(f"⚠️ Error saving message to database: {e}")
            return None
    
    def add_to_knowledge_graph(
        self,
        node_id: str,
        content: str,
        node_type: str = "Concept",
        label: Optional[str] = None,
        importance: float = 0.5,
        related_nodes: Optional[List[Dict]] = None,
        session_id: Optional[str] = None
    ) -> bool:
        """
        Add a concept/entity to the knowledge graph with optional relationships.
        
        Args:
            node_id: Unique identifier for the node
            content: Text content to embed in vector store
            node_type: Type of node (Concept, Entity, Document, etc.)
            label: Human-readable label
            importance: Importance score (0-1)
            related_nodes: List of {"node_id": str, "relation": str} dicts
            session_id: Optional session for scoping
            
        Returns:
            True if successful
        """
        try:
            # Add node to graph
            self.graph_store.add_node(
                node_id=node_id,
                node_type=node_type,
                label=label or node_id,
                importance=importance
            )
            
            # Add to vector store with graph link
            doc_id = f"graph_{node_id}"
            self.vector_store.add_document(
                doc_id=doc_id,
                content=content,
                node_id=node_id,
                node_type=node_type,
                metadata={"session_id": session_id} if session_id else None
            )
            
            # Add relationships if provided
            if related_nodes:
                for rel in related_nodes:
                    target_id = rel.get("node_id")
                    relation = rel.get("relation", "RELATED_TO")
                    if target_id:
                        self.graph_store.add_edge(
                            source=node_id,
                            target=target_id,
                            relation=relation
                        )
            
            print(f"✓ Added to knowledge graph: {node_id} ({node_type})")
            return True
            
        except Exception as e:
            print(f"⚠️ Error adding to knowledge graph: {e}")
            return False
    
    def get_graph_stats(self) -> Dict:
        """Get statistics about the knowledge graph."""
        return self.graph_store.get_stats()
    
    def toggle_graph_rag(self, enabled: bool):
        """Enable or disable Graph RAG mode."""
        self.use_graph_rag = enabled
        print(f"🔧 Graph RAG mode: {'ENABLED' if enabled else 'DISABLED'}")
    
    def toggle_entity_extraction(self, enabled: bool):
        """Enable or disable automatic entity extraction."""
        self.auto_extract_entities = enabled
        print(f"🔧 Auto entity extraction: {'ENABLED' if enabled else 'DISABLED'}")
    
    def _extract_entities_async(self, text: str, session_id: str, role: str, message_id: Optional[int] = None):
        """
        Extract entities from text and add to knowledge graph.
        Uses different API keys based on role to distribute rate limits.
        
        Args:
            text: Text to extract from
            session_id: Current session ID
            role: 'user' or 'model'
        """
        try:
            # Use different extractor based on role (different API keys)
            if role == 'user':
                extractor = self.entity_extractor_user
                api_key = self.entity_user_api_key
            else:
                extractor = self.entity_extractor_model
                api_key = self.entity_model_api_key
            
            print(f"\n📤 [ENTITY EXTRACTION - {role.upper()}]")
            print(f"   API Key: ...{api_key[-8:]}")
            print(f"   Text: {text[:50]}...")
            
            result = extractor.extract(text)
            
            if result.entities:
                # Add entities to graph store
                for entity in result.entities:
                    self.graph_store.add_node(
                        node_id=entity.id,
                        label=entity.description or entity.name,
                        node_type=entity.type,
                        importance=entity.importance,
                        metadata={'session_id': session_id, 'role': role}
                    )
                
                # Add relationships to graph store
                # ExtractedRelation is always a dataclass — access attributes directly, not via .get()
                for relation in result.relations:
                    src      = relation.source_id
                    tgt      = relation.target_id
                    rel_type = relation.relation_type
                    if src and tgt:
                        self.graph_store.add_edge(
                            source=src,
                            target=tgt,
                            relation=rel_type,
                            metadata={'session_id': session_id}
                        )

                # Link created nodes to vector store so Fusion can find seed nodes
                if message_id:
                    doc_id = f"{session_id}_{message_id}"
                    first_node_id = None
                    for i, entity in enumerate(result.entities):
                        node_id = entity.id
                        node_type = entity.type
                        # create a short document linked to the node so semantic search returns node-linked docs
                        per_doc_id = f"{node_id}_{message_id}"
                        content = entity.description or entity.name
                        try:
                            self.vector_store.add_document(
                                doc_id=per_doc_id,
                                content=content,
                                node_id=node_id,
                                node_type=node_type,
                                metadata={"session_id": session_id, "source_doc": doc_id}
                            )
                        except Exception as e:
                            print(f"               ⚠️ Failed to add per-entity document for {node_id}: {e}")
                        if i == 0:
                            first_node_id = node_id
                    # update original message doc to link to the first node (best-fit)
                    if first_node_id:
                        try:
                            self.vector_store.update_node_link(doc_id, first_node_id, node_type=result.entities[0].type)
                            print(f"               ✓ Linked message {doc_id} to node {first_node_id}")
                        except Exception as e:
                            print(f"               ⚠️ Failed to update node link for {doc_id}: {e}")

                entity_names = [e.name for e in result.entities]
                print(f"               🧠 Extracted entities: {', '.join(entity_names)}")
                
        except Exception as e:
            print(f"               ⚠️ Entity extraction error: {e}")
    
    def extract_from_text(
        self,
        text: str,
        session_id: Optional[str] = None,
        mode: str = "standard"
    ) -> Dict:
        """
        Manually extract entities from text and add to knowledge graph.
        
        Args:
            text: Text to extract from
            session_id: Optional session context
            mode: "minimal", "standard", or "comprehensive"
            
        Returns:
            Extraction result dict
        """
        extraction_mode = {
            "minimal": ExtractionMode.MINIMAL,
            "standard": ExtractionMode.STANDARD,
            "comprehensive": ExtractionMode.COMPREHENSIVE
        }.get(mode, ExtractionMode.STANDARD)
        
        result = self.entity_extractor_user.extract_and_populate(
            text=text,
            graph_store=self.graph_store,
            vector_store=self.vector_store,
            session_id=session_id,
            mode=extraction_mode
        )
        
        return result.to_dict()
    
    def ingest_document(
        self,
        content: str,
        doc_id: str,
        title: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Dict:
        """
        Ingest a document: extract entities and add to knowledge graph.
        
        Args:
            content: Document content
            doc_id: Unique document identifier
            title: Optional document title
            session_id: Optional session for scoping
            
        Returns:
            Dict with ingestion results
        """
        # Add document node
        self.graph_store.add_node(
            node_id=doc_id,
            node_type="Document",
            label=title or doc_id,
            importance=0.7
        )
        
        # Add to vector store
        self.vector_store.add_document(
            doc_id=doc_id,
            content=content,
            node_id=doc_id,
            node_type="Document",
            metadata={"title": title, "session_id": session_id}
        )
        
        # Extract entities with comprehensive mode for documents
        result = self.entity_extractor_user.extract_and_populate(
            text=content,
            graph_store=self.graph_store,
            vector_store=self.vector_store,
            session_id=session_id,
            mode=ExtractionMode.COMPREHENSIVE
        )
        
        # Link extracted entities to document
        for entity in result.entities:
            self.graph_store.add_edge(
                source=doc_id,
                target=entity.id,
                relation="CONTAINS"
            )
        
        return {
            "doc_id": doc_id,
            "title": title,
            "entities_extracted": len(result.entities),
            "relations_extracted": len(result.relations),
            "entities": [e.name for e in result.entities]
        }