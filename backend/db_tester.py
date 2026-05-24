import sqlite3
import os
import chromadb

def read_data_from_sqlite(db_path, table_name):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Execute query
        cursor.execute(f"SELECT * FROM {table_name};")

        # Fetch all rows
        rows = cursor.fetchall()

        # Get column names
        columns = [description[0] for description in cursor.description]

        # Print header
        print(f"\n=== {table_name.upper()} TABLE ===")
        print(" | ".join(columns))
        print("-" * 80)

        # Print each row
        if rows:
            for row in rows:
                print(" | ".join(str(value) for value in row))
        else:
            print("No data found in this table.")

    except sqlite3.Error as e:
        print(f"Error reading data from SQLite: {e}")

    finally:
        if conn:
            conn.close()

def check_vegapunk_database():
    """Check VEGAPUNK database and display all data"""
    db_file = "vegapunk_memory.db"  # VEGAPUNK database file
    
    if not os.path.exists(db_file):
        print(f"❌ Database file '{db_file}' not found!")
        print("Make sure the backend has been run at least once to create the database.")
        return
    
    print(f"🔍 Reading data from VEGAPUNK database: {db_file}")
    print("=" * 80)
    
    # Read from conversations table
    read_data_from_sqlite(db_file, "conversations")
    
    # Read from sessions table
    read_data_from_sqlite(db_file, "sessions")
    
    # Check if ChromaDB directory exists
    chroma_dir = "./chroma_db"
    if os.path.exists(chroma_dir):
        print(f"\n=== VECTOR STORE INFO ===")
        print(f"✅ ChromaDB directory found: {chroma_dir}")
        
        # List files in ChromaDB directory
        chroma_files = os.listdir(chroma_dir)
        print(f"📁 Files in vector store: {chroma_files}")
        
        # Try to get collection info (requires chromadb to be installed)
        try:
            # import chromadb
            client = chromadb.PersistentClient(path=chroma_dir)
            collections = client.list_collections()
            
            if collections:
                print(f"📚 Collections found: {len(collections)}")
                for collection in collections:
                    print(f"   - {collection.name}")
                    count = collection.count()
                    print(f"     Documents: {count}")
            else:
                print("📚 No collections found in vector store")
                
        except ImportError:
            print("⚠️  ChromaDB not available for detailed inspection")
        except Exception as e:
            print(f"⚠️  Error reading ChromaDB: {e}")
    else:
        print(f"\n❌ ChromaDB directory not found: {chroma_dir}")
        print("Vector store hasn't been initialized yet.")

def show_database_stats():
    """Show statistics about the VEGAPUNK database"""
    db_file = "vegapunk_memory.db"
    
    if not os.path.exists(db_file):
        return
    
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        
        print(f"\n=== DATABASE STATISTICS ===")
        
        # Count sessions
        cursor.execute("SELECT COUNT(*) FROM sessions;")
        session_count = cursor.fetchone()[0]
        print(f"📊 Total sessions: {session_count}")
        
    
        cursor.execute("SELECT COUNT(*) FROM conversations;")
        message_count = cursor.fetchone()[0]
        print(f"💬 Total messages: {message_count}")
        
        # Count by role
        cursor.execute("SELECT role, COUNT(*) FROM conversations GROUP BY role;")
        role_counts = cursor.fetchall()
        for role, count in role_counts:
            print(f"   - {role}: {count}")
        
        # Recent activity
        cursor.execute("""
            SELECT session_id, COUNT(*) as msg_count, MAX(timestamp) as last_activity 
            FROM conversations 
            GROUP BY session_id 
            ORDER BY last_activity DESC 
            LIMIT 5;
        """)
        recent = cursor.fetchall()
        
        if recent:
            print(f"\n🕒 Recent sessions:")
            for session_id, msg_count, last_activity in recent:
                print(f"   - {session_id[:8]}... ({msg_count} messages, last: {last_activity})")
        
        conn.close()
        
    except sqlite3.Error as e:
        print(f"Error reading statistics: {e}")

# Main execution
if __name__ == "__main__":
    print("🧠 VEGAPUNK Database Inspector")
    print("=" * 50)
    
    # Check and display all database content
    check_vegapunk_database()
    
    # Show statistics
    show_database_stats()
    
    print("\n" + "=" * 80)
    print("✅ Database inspection complete!")