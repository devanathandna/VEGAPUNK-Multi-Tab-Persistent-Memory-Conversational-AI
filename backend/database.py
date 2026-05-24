import sqlite3
from datetime import datetime
from typing import List, Dict, Optional
import json

class VegapunkDatabase:
    def __init__(self, db_path: str = "vegapunk_memory.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create conversations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        ''')
        
        # Create sessions table for tracking main chats
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                title TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create index for faster queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_session_timestamp 
            ON conversations(session_id, timestamp)
        ''')
        
        # ==================== GRAPH TABLES ====================
        
        # Graph nodes table - stores concepts, entities, documents, aliases
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS graph_nodes (
                node_id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                label TEXT,
                importance REAL DEFAULT 0.5,
                metadata TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Graph edges table - stores relationships between nodes
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS graph_edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                target TEXT NOT NULL,
                relation TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                metadata TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source) REFERENCES graph_nodes(node_id),
                FOREIGN KEY (target) REFERENCES graph_nodes(node_id)
            )
        ''')
        
        # Graph indexes for faster lookups
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_edges_source 
            ON graph_edges(source)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_edges_target 
            ON graph_edges(target)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_edges_relation 
            ON graph_edges(relation)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_nodes_type 
            ON graph_nodes(type)
        ''')
        
        conn.commit()
        conn.close()
    
    def create_session(self, session_id: str, title: str = "New Chat") -> bool:
        """Create a new chat session"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO sessions (session_id, title) VALUES (?, ?)",
                (session_id, title)
            )
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError:
            return False
    
    def save_message(self, session_id: str, role: str, content: str, metadata: Optional[Dict] = None) -> int:
        """Save a message to the database"""
        print("\n" + "=" * 60)
        print("[DB STEP 1] 💾 SQL — save_message CALLED")
        print(f"            session_id : {session_id}")
        print(f"            role       : {role}")
        print(f"            content len: {len(content)} chars")
        print(f"            preview    : {content[:120].replace(chr(10),' ')}{'...' if len(content)>120 else ''}")
        print(f"            metadata   : {metadata}")
        print("=" * 60)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        metadata_json = json.dumps(metadata) if metadata else None

        cursor.execute(
            """INSERT INTO conversations (session_id, role, content, metadata)
               VALUES (?, ?, ?, ?)""",
            (session_id, role, content, metadata_json)
        )

        message_id = cursor.lastrowid

        # Update session's updated_at timestamp
        cursor.execute(
            "UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
            (session_id,)
        )

        conn.commit()
        conn.close()

        print(f"[DB STEP 2] ✅ SQL — Message committed  →  message_id = {message_id}")
        print("=" * 60 + "\n")
        return message_id
    
    def get_session_history(self, session_id: str, limit: Optional[int] = None) -> List[Dict]:
        """Get conversation history for a specific session"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        if limit:
            # For limited results, get the most recent messages and return in chronological order
            query = """
                SELECT id, role, content, timestamp, metadata 
                FROM (
                    SELECT id, role, content, timestamp, metadata 
                    FROM conversations 
                    WHERE session_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ) 
                ORDER BY timestamp ASC
            """
            cursor.execute(query, (session_id, limit))
        else:
            # For all messages, return in chronological order
            query = """
                SELECT id, role, content, timestamp, metadata 
                FROM conversations 
                WHERE session_id = ? 
                ORDER BY timestamp ASC
            """
            cursor.execute(query, (session_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_all_sessions(self) -> List[Dict]:
        """Get all chat sessions"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT s.session_id, s.title, s.created_at, s.updated_at,
                   COUNT(c.id) as message_count
            FROM sessions s
            LEFT JOIN conversations c ON s.session_id = c.session_id
            GROUP BY s.session_id
            ORDER BY s.updated_at DESC
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def search_messages(self, session_id: str, query: str, hours_ago: int = 24) -> List[Dict]:
        """Search messages with a query string within a time range"""
        print("\n" + "=" * 60)
        print("[DB STEP 1] 🔎 SQL — search_messages CALLED")
        print(f"            session_id : {session_id}")
        print(f"            query      : {query[:80]}")
        print(f"            hours_ago  : {hours_ago} hours  (window = last {hours_ago/24:.1f} days)")
        print(f"            LIKE term  : %{query[:40]}%")
        print("=" * 60)

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        search_term = f"%{query}%"

        cursor.execute("""
            SELECT id, role, content, timestamp
            FROM conversations
            WHERE session_id = ?
            AND content LIKE ?
            AND datetime(timestamp) >= datetime('now', '-' || ? || ' hours')
            ORDER BY timestamp ASC
        """, (session_id, search_term, hours_ago))

        rows = cursor.fetchall()
        conn.close()

        results = [dict(row) for row in rows]

        print(f"[DB STEP 2] 📦 SQL — Rows matched : {len(results)}")
        for i, r in enumerate(results, 1):
            snip = str(r.get('content', ''))[:120].replace('\n', ' ')
            print(f"            [{i}] id={r.get('id')} role={r.get('role')} ts={str(r.get('timestamp',''))[:19]}")
            print(f"                 {snip}{'...' if len(str(r.get('content','')))>120 else ''}")
        if not results:
            print("            (no rows matched)")
        print("=" * 60 + "\n")
        return results
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its messages"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))
            cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error deleting session: {e}")
            return False
    
    def update_session_title(self, session_id: str, new_title: str) -> bool:
        """Update the title of a session"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "UPDATE sessions SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
                (new_title, session_id)
            )
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error updating session title: {e}")
            return False
