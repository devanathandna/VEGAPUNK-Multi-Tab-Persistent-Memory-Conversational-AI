"""
Test script to inspect the graph store database and verify data is being saved.
"""
import sqlite3
import os

def inspect_graph_db():
    db_path = "vegapunk_memory.db"
    
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("=" * 60)
    print("🔍 GRAPH DATABASE INSPECTION")
    print("=" * 60)
    
    # Check graph_nodes table
    print("\n📊 GRAPH NODES:")
    print("-" * 40)
    try:
        cursor.execute("SELECT COUNT(*) FROM graph_nodes")
        count = cursor.fetchone()[0]
        print(f"   Total nodes: {count}")
        
        if count > 0:
            cursor.execute("SELECT node_id, type, label, importance FROM graph_nodes LIMIT 10")
            rows = cursor.fetchall()
            for row in rows:
                print(f"   - {row[0]}: {row[2]} (type: {row[1]}, importance: {row[3]})")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Check graph_edges table
    print("\n🔗 GRAPH EDGES:")
    print("-" * 40)
    try:
        cursor.execute("SELECT COUNT(*) FROM graph_edges")
        count = cursor.fetchone()[0]
        print(f"   Total edges: {count}")
        
        if count > 0:
            cursor.execute("SELECT source, target, relation FROM graph_edges LIMIT 10")
            rows = cursor.fetchall()
            for row in rows:
                print(f"   - {row[0]} --[{row[2]}]--> {row[1]}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Check messages table (actually called 'conversations')
    print("\n💬 CONVERSATIONS:")
    print("-" * 40)
    try:
        cursor.execute("SELECT COUNT(*) FROM conversations")
        count = cursor.fetchone()[0]
        print(f"   Total conversations: {count}")
        
        if count > 0:
            cursor.execute("SELECT id, session_id, role, content FROM conversations ORDER BY id DESC LIMIT 5")
            rows = cursor.fetchall()
            for row in rows:
                content_preview = row[3][:50] + "..." if len(row[3]) > 50 else row[3]
                print(f"   - [{row[2]}] {content_preview}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Check sessions table
    print("\n📁 SESSIONS:")
    print("-" * 40)
    try:
        cursor.execute("SELECT COUNT(*) FROM sessions")
        count = cursor.fetchone()[0]
        print(f"   Total sessions: {count}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # List all tables
    print("\n📋 ALL TABLES:")
    print("-" * 40)
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
        count = cursor.fetchone()[0]
        print(f"   - {table[0]}: {count} rows")
    
    conn.close()
    print("\n" + "=" * 60)

if __name__ == "__main__":
    inspect_graph_db()
