from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
from datetime import datetime
import uuid
from agent import VegapunkAgent
from database import VegapunkDatabase
from vector_store import VegapunkVectorStore
import re
import base64

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Enable CORS for all routes
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:8080", "http://localhost:3000"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Configuration from environment variables
GROQ_API_KEY_1 = os.getenv('GROQ_API_KEY_1', '')
GROQ_API_KEY_2 = os.getenv('GROQ_API_KEY_2', '')
GROQ_API_KEY_3 = os.getenv('GROQ_API_KEY_3', '')
GROQ_API_KEY_4 = os.getenv('GROQ_API_KEY_4', '')
PORT = int(os.getenv('PORT', 5000))
DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'

# Build API keys dictionary for load balancing across 4 keys
api_keys = {
    'main': GROQ_API_KEY_1,           # Main LLM response
    'classifier': GROQ_API_KEY_2,     # Query classification
    'entity_user': GROQ_API_KEY_3,    # Entity extraction (user messages)
    'entity_model': GROQ_API_KEY_4,   # Entity extraction (model responses)
}

# Initialize agent and database connections ONCE
if GROQ_API_KEY_1:
    print("🧠 Initializing VEGAPUNK Agent with Hybrid Memory...")
    print("🔑 Using Groq (llama-3.1-8b-instant) with 4 API keys for distributed rate limits")
    db = VegapunkDatabase()
    vector_store = VegapunkVectorStore()
    agent = VegapunkAgent(api_keys=api_keys, db=db, vector_store=vector_store)
    print("✅ Agent ready!")
else:
    print("⚠️  Warning: GROQ_API_KEY_1 not set. Please add it to .env file")
    agent = None
    db = None
    vector_store = None

def extract_image_data(html_content):
    """Extract image data from HTML content"""
    img_match = re.search(r'<img[^>]+src="data:image/[^;]+;base64,([^"]+)"', html_content)
    if img_match:
        try:
            image_data = base64.b64decode(img_match.group(1))
            return image_data
        except Exception as e:
            print(f"Error decoding image: {e}")
    return None

def extract_text_content(html_content):
    """Extract text content from HTML, removing image tags"""
    text_content = re.sub(r'<img[^>]*>', '', html_content).strip()
    return text_content

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    is_ready = agent is not None and db is not None and vector_store is not None
    return jsonify({
        'status': 'healthy' if is_ready else 'initializing',
        'ready': is_ready,
        'timestamp': datetime.now().isoformat(),
        'service': 'VEGAPUNK Backend',
        'components': {
            'agent': agent is not None,
            'database': db is not None,
            'vector_store': vector_store is not None
        }
    }), 200 if is_ready else 503

@app.route('/api/chat/main', methods=['POST'])
def main_chat():
    """Handle main chat messages with hybrid retrieval"""
    try:
        print("\n" + "#"*80)
        print("[STEP 1] 📥 INPUT FROM API RECEIVED")
        data = request.get_json()
        print(f"         Raw data: {data}")
        
        if not data or 'message' not in data:
            print("         ❌ ERROR: Message is required")
            return jsonify({'error': 'Message is required'}), 400
        
        user_message_html = data['message']
        session_id = data.get('session_id')
        print(f"         Message: {user_message_html[:100]}...")
        print(f"         Session ID: {session_id}")
        
        if not agent:
            print("         ❌ ERROR: Gemini API key not configured")
            return jsonify({'error': 'Gemini API key not configured'}), 500
        
        # Create new session if not provided
        if not session_id:
            session_id = str(uuid.uuid4())
            text_content = extract_text_content(user_message_html)
            title = text_content[:50] if text_content else "New Chat"
            db.create_session(session_id, title)
            print(f"         ✓ Created new session: {session_id}")
        
        # Extract text and image
        text_content = extract_text_content(user_message_html)
        image_data = extract_image_data(user_message_html)
        print(f"         ✓ Extracted text ({len(text_content)} chars)")
        if image_data:
            print(f"         ✓ Extracted image ({len(image_data)} bytes)")
        
        if not text_content:
            print("         ❌ ERROR: Empty message received")
            return jsonify({'error': 'Empty message received'}), 400

        print("\n[STEP 2] 🚀 LOADING INPUT TO GEMINI")
        print(f"         Calling agent.chat()...")
        # Use agent for intelligent response with hybrid retrieval
        result = agent.chat(session_id, text_content, image_data)
        
        print("\n[STEP 3] ✅ CONTENT FROM GEMINI FETCHED")
        print(f"         Response length: {len(result['response'])} chars")
        print(f"         Context used: {result['context_used']}")
        print(f"         Response preview: {result['response'][:100]}...")
        
        print("\n[STEP 4] 📤 SENDING RESPONSE TO FRONTEND")
        response_data = {
            'success': True,
            'response': result['response'],
            'session_id': session_id,
            'context_used': result['context_used'],
            'timestamp': datetime.now().isoformat(),
            'chat_type': 'main',
            'model': 'llama-3.1-8b-instant'
        }
        print(f"         ✓ Response prepared")
        print("#"*80 + "\n")
        return jsonify(response_data), 200
        
    except Exception as e:
        print(f"\n❌❌❌ ERROR IN MAIN CHAT: {str(e)}")
        import traceback
        traceback.print_exc()
        print("#"*80 + "\n")
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat/sub', methods=['POST'])
def sub_chat():
    """Handle sub chat (satellite) messages - inherits context from main chat"""
    try:
        print("\n" + "#"*80)
        print("[STEP 1] 📥 INPUT FROM API RECEIVED (SUB CHAT)")
        data = request.get_json()
        
        if not data or 'message' not in data or 'subChatId' not in data:
            print("         ❌ ERROR: Message and subChatId are required")
            return jsonify({'error': 'Message and subChatId are required'}), 400
        
        user_message_html = data['message']
        sub_chat_id = data['subChatId']
        session_id = data.get('session_id')
        print(f"         Sub Chat ID: {sub_chat_id}")
        print(f"         Session ID: {session_id}")
        
        if not agent or not session_id:
            print("         ❌ ERROR: Session ID required for sub chats")
            return jsonify({'error': 'Session ID required for sub chats'}), 400
        
        # Extract text and image
        text_content = extract_text_content(user_message_html)
        image_data = extract_image_data(user_message_html)
        print(f"         ✓ Extracted text ({len(text_content)} chars)")

        if not text_content:
            print("         ❌ ERROR: Empty message received")
            return jsonify({'error': 'Empty message received'}), 400
            
        # Add satellite context
        text_content = f"[Satellite Chat #{sub_chat_id}] {text_content}"
        print(f"         ✓ Added satellite context")

        print("\n[STEP 2] 🚀 LOADING INPUT TO GEMINI (SUB CHAT)")
        # Sub-chats can access main chat memory but DON'T save their conversations
        result = agent.chat(session_id, text_content, image_data, save_to_memory=False)
        
        print("\n[STEP 3] ✅ CONTENT FROM GEMINI FETCHED (SUB CHAT)")
        print(f"         Response length: {len(result['response'])} chars")
        
        print("\n[STEP 4] 📤 SENDING RESPONSE TO FRONTEND (SUB CHAT)")
        response_data = {
            'success': True,
            'response': result['response'],
            'subChatId': sub_chat_id,
            'session_id': session_id,
            'context_used': result['context_used'],
            'timestamp': datetime.now().isoformat(),
            'chat_type': 'sub',
            'model': 'llama-3.1-8b-instant'
        }
        print("#"*80 + "\n")
        return jsonify(response_data), 200
        
    except Exception as e:
        print(f"\n❌❌❌ ERROR IN SUB CHAT: {str(e)}")
        import traceback
        traceback.print_exc()
        print("#"*80 + "\n")
        return jsonify({'error': str(e)}), 500

@app.route('/api/sessions', methods=['GET'])
def get_sessions():
    """Get all chat sessions"""
    try:
        if not db:
            return jsonify({'error': 'Database not initialized'}), 500
        
        sessions = db.get_all_sessions()
        return jsonify({
            'success': True,
            'sessions': sessions
        }), 200
    except Exception as e:
        print(f"Error getting sessions: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/sessions/grouped', methods=['GET'])
def get_grouped_sessions():
    """Return sessions grouped by title (useful for collapsing duplicates into one entity)"""
    try:
        if not db:
            return jsonify({'error': 'Database not initialized'}), 500

        sessions = db.get_all_sessions()
        groups = {}
        for s in sessions:
            key = s.get('title') or 'Untitled Chat'
            if key not in groups:
                groups[key] = {
                    'group_key': key,
                    'title': key,
                    'sessions': [],
                    'message_count': 0,
                    'updated_at': s.get('updated_at')
                }
            groups[key]['sessions'].append({'session_id': s['session_id'], 'updated_at': s.get('updated_at')})
            groups[key]['message_count'] += s.get('message_count', 0) or 0
            # keep latest updated_at
            if s.get('updated_at') and s.get('updated_at') > groups[key]['updated_at']:
                groups[key]['updated_at'] = s.get('updated_at')

        grouped_list = list(groups.values())
        # sort by latest updated
        grouped_list.sort(key=lambda x: x.get('updated_at') or '', reverse=True)

        return jsonify({'success': True, 'groups': grouped_list}), 200
    except Exception as e:
        print(f"Error getting grouped sessions: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/sessions/group/<path:group_key>', methods=['GET', 'DELETE'])
def handle_group(group_key):
    """Handle GET and DELETE for a session group"""
    try:
        if not db:
            return jsonify({'error': 'Database not initialized'}), 500

        # Find all sessions matching the group key (title)
        sessions = db.get_all_sessions()
        matching_sessions = [s for s in sessions if (s.get('title') or 'Untitled Chat') == group_key]

        if not matching_sessions:
            return jsonify({'error': 'Group not found'}), 404

        if request.method == 'GET':
            # --- Existing GET Logic ---
            combined = []
            for s in matching_sessions:
                hist = db.get_session_history(s['session_id'])
                for m in hist:
                    m['_session_id'] = s['session_id']
                    combined.append(m)
            combined.sort(key=lambda x: x.get('timestamp') or '')
            return jsonify({'success': True, 'history': combined, 'sessions': [s['session_id'] for s in matching_sessions]}), 200

        elif request.method == 'DELETE':
            # --- New DELETE Logic ---
            deleted_count = 0
            for session in matching_sessions:
                # Also delete from vector store
                agent.vector_store.delete_session(session['session_id'])
                # Delete from SQL database
                success = db.delete_session(session['session_id'])
                if success:
                    deleted_count += 1
            
            print(f"Deleted {deleted_count} sessions for group '{group_key}'")
            return jsonify({'success': True, 'deleted_count': deleted_count}), 200

    except Exception as e:
        print(f"Error handling group '{group_key}': {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/sessions/<session_id>', methods=['GET'])
def get_session_history(session_id):
    """Get conversation history for a specific session"""
    try:
        if not db:
            return jsonify({'error': 'Database not initialized'}), 500
            
        history = db.get_session_history(session_id)
        return jsonify({
            'success': True,
            'history': history,
            'session_id': session_id
        }), 200
    except Exception as e:
        print(f"Error getting session history: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/sessions/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    """Delete a session and all its messages"""
    try:
        if not db:
            return jsonify({'error': 'Database not initialized'}), 500
            
        success = db.delete_session(session_id)
        return jsonify({
            'success': success
        }), 200
    except Exception as e:
        print(f"Error deleting session: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/sessions/group/<path:group_key>/title', methods=['PUT'])
def update_group_title(group_key):
    """Update the title for all sessions in a group"""
    try:
        if not db:
            return jsonify({'error': 'Database not initialized'}), 500
        
        data = request.get_json()
        new_title = data.get('new_title', '').strip()
        
        if not new_title:
            return jsonify({'error': 'New title is required'}), 400
        
        # Get all sessions with the old title
        sessions = db.get_all_sessions()
        matching_sessions = [s for s in sessions if (s.get('title') or 'Untitled Chat') == group_key]
        
        if not matching_sessions:
            return jsonify({'error': 'No sessions found with that title'}), 404
        
        # Update title for all matching sessions
        updated_count = 0
        for session in matching_sessions:
            success = db.update_session_title(session['session_id'], new_title)
            if success:
                updated_count += 1
        
        return jsonify({
            'success': True,
            'updated_count': updated_count,
            'new_title': new_title
        }), 200
    except Exception as e:
        print(f"Error updating group title: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 VEGAPUNK Backend Server Starting...")
    print("=" * 60)
    print("📊 Hybrid Memory System Initialized:")
    print("   ├── SQL Database: vegapunk_memory.db")
    print("   ├── Vector Store: ChromaDB")
    print("   └── Embeddings: sentence-transformers/all-MiniLM-L6-v2")
    print("🧠 Agent ready with parallel thinking capabilities")
    print(f"🔧 Debug mode: {DEBUG}")
    print(f"🌐 Server running on port {PORT}")
    print("=" * 60)
    try:
        app.run(host='0.0.0.0', port=PORT, debug=DEBUG, use_reloader=False)
    except Exception as e:
        print(f"\n❌❌❌ FATAL ERROR STARTING SERVER:")
        print(f"{e}")
        import traceback
        traceback.print_exc()
        raise
