from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from typing import List, Dict, Optional
from datetime import datetime

class VegapunkVectorStore:
    def __init__(self, persist_directory: str = "./chroma_db"):
        """Initialize the vector store with HuggingFace embeddings"""
        print("🔄 Loading sentence-transformers model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        self.vectorstore = Chroma(
            collection_name="vegapunk_conversations",
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )
        print("✅ Vector store initialized")
    
    def add_message(
        self, 
        session_id: str, 
        message_id: int, 
        role: str, 
        content: str, 
        metadata: Optional[Dict] = None,
        node_id: Optional[str] = None,
        node_type: Optional[str] = None
    ) -> str:
        """
        Add a message to the vector store.
        
        Args:
            session_id: The session ID for the conversation
            message_id: Unique message identifier
            role: 'user' or 'assistant'
            content: The message content
            metadata: Additional metadata
            node_id: Graph node ID to link this embedding to (for Graph RAG)
            node_type: Type of the linked node (Concept, Entity, etc.)
            
        Returns:
            The document ID
        """
        doc_id = f"{session_id}_{message_id}"

        doc_metadata = {
            "session_id": session_id,
            "message_id": message_id,
            "role": role,
            "timestamp": datetime.now().isoformat(),
            **(metadata or {})
        }

        # Add graph node linking if provided
        if node_id:
            doc_metadata["node_id"] = node_id
        if node_type:
            doc_metadata["node_type"] = node_type

        print("\n" + "=" * 60)
        print("[VECTOR STEP 1] 📌 ChromaDB — add_message CALLED")
        print(f"               doc_id      : {doc_id}")
        print(f"               role        : {role}")
        print(f"               content len : {len(content)} chars")
        print(f"               preview     : {content[:120].replace(chr(10),' ')}{'...' if len(content)>120 else ''}")
        print(f"               metadata    : {doc_metadata}")
        print(f"               node_id     : {node_id or 'None (plain message, no graph link)'}")
        print("=" * 60)

        self.vectorstore.add_texts(
            texts=[content],
            metadatas=[doc_metadata],
            ids=[doc_id]
        )

        print(f"[VECTOR STEP 2] ✅ ChromaDB — Embedding stored  →  id = {doc_id}")
        print("=" * 60 + "\n")
        return doc_id
    
    def add_document(
        self,
        doc_id: str,
        content: str,
        node_id: str,
        node_type: str = "Document",
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Add a document/knowledge chunk to the vector store linked to a graph node.
        
        Args:
            doc_id: Unique document identifier
            content: The document content
            node_id: Graph node ID this document is linked to
            node_type: Type of the linked node
            metadata: Additional metadata
            
        Returns:
            The document ID
        """
        doc_metadata = {
            "doc_id": doc_id,
            "node_id": node_id,
            "node_type": node_type,
            "timestamp": datetime.now().isoformat(),
            **(metadata or {})
        }

        print("\n" + "=" * 60)
        print("[VECTOR STEP 1] 📖 ChromaDB — add_document CALLED")
        print(f"               doc_id      : {doc_id}")
        print(f"               node_id     : {node_id}")
        print(f"               node_type   : {node_type}")
        print(f"               content len : {len(content)} chars")
        print(f"               preview     : {content[:120].replace(chr(10),' ')}{'...' if len(content)>120 else ''}")
        print(f"               metadata    : {doc_metadata}")
        print("=" * 60)

        self.vectorstore.add_texts(
            texts=[content],
            metadatas=[doc_metadata],
            ids=[doc_id]
        )

        print(f"[VECTOR STEP 2] ✅ ChromaDB — Document embedding stored  →  id = {doc_id}")
        print("=" * 60 + "\n")
        return doc_id
    
    def semantic_search(
        self, 
        query: str, 
        session_id: Optional[str] = None, 
        k: int = 5,
        include_node_ids: bool = False
    ) -> List[Dict]:
        """
        Perform semantic search on the vector store.
        
        Args:
            query: Search query
            session_id: Optional session filter
            k: Number of results to return
            include_node_ids: If True, only returns results that have node_id
            
        Returns:
            List of results with content, metadata, and similarity_score
        """
        filter_dict = None

        if session_id and include_node_ids:
            # Chroma doesn't support complex AND filters easily,
            # so we'll filter post-query
            filter_dict = {"session_id": session_id}
        elif session_id:
            filter_dict = {"session_id": session_id}

        fetch_k = k * 2 if include_node_ids else k

        print("\n" + "=" * 60)
        print("[VECTOR STEP 1] 🔍 ChromaDB — semantic_search CALLED")
        print(f"               query          : {query[:80]}{'...' if len(query)>80 else ''}")
        print(f"               session_id     : {session_id or 'None (global search)'}")
        print(f"               k requested    : {k}")
        print(f"               fetch_k actual : {fetch_k}  (doubled={'YES' if include_node_ids else 'NO'})")
        print(f"               include_node_ids: {include_node_ids}")
        print(f"               filter applied : {filter_dict}")
        print("=" * 60)

        results = self.vectorstore.similarity_search_with_score(
            query=query,
            k=fetch_k,
            filter=filter_dict
        )

        print(f"[VECTOR STEP 2] 📤 ChromaDB — Raw results returned: {len(results)} docs")
        for i, (doc, score) in enumerate(results, 1):
            nid = doc.metadata.get('node_id', 'None')
            snip = doc.page_content[:100].replace('\n', ' ')
            print(f"               [{i}] score={score:.4f}  node_id={nid}")
            print(f"                    {snip}{'...' if len(doc.page_content)>100 else ''}")
        print("=" * 60)

        formatted_results = []
        kept = 0
        skipped = 0
        for doc, score in results:
            result = {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": float(score),
                "node_id": doc.metadata.get("node_id"),
                "node_type": doc.metadata.get("node_type")
            }

            # Filter for node_ids if requested
            if include_node_ids:
                if result["node_id"]:
                    formatted_results.append(result)
                    kept += 1
                else:
                    skipped += 1
            else:
                formatted_results.append(result)
                kept += 1

            if len(formatted_results) >= k:
                break

        print(f"[VECTOR STEP 3] ✅ ChromaDB — After node_id filter: kept={kept}, skipped={skipped}")
        print(f"               Final result count: {len(formatted_results)}")
        print("=" * 60 + "\n")
        return formatted_results
    
    def search_by_node_ids(self, query: str, node_ids: List[str], k: int = 5) -> List[Dict]:
        """
        Search within documents linked to specific graph nodes.
        
        Args:
            query: Search query
            node_ids: List of node IDs to search within
            k: Number of results per node
            
        Returns:
            List of results with content, metadata, and similarity_score
        """
        all_results = []
        
        for node_id in node_ids:
            try:
                results = self.vectorstore.similarity_search_with_score(
                    query=query,
                    k=k,
                    filter={"node_id": node_id}
                )
                
                for doc, score in results:
                    all_results.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "similarity_score": float(score),
                        "node_id": node_id,
                        "node_type": doc.metadata.get("node_type")
                    })
            except Exception as e:
                print(f"Error searching node {node_id}: {e}")
                continue
        
        # Sort by similarity score (lower is better for Chroma's L2 distance)
        all_results.sort(key=lambda x: x["similarity_score"])
        
        return all_results[:k * len(node_ids)]
    
    def get_by_node_id(self, node_id: str) -> List[Dict]:
        """
        Get all documents linked to a specific graph node.
        
        Args:
            node_id: The graph node ID
            
        Returns:
            List of documents linked to this node
        """
        try:
            results = self.vectorstore.get(where={"node_id": node_id})
            
            documents = []
            if results and results.get('ids'):
                for i, doc_id in enumerate(results['ids']):
                    documents.append({
                        "doc_id": doc_id,
                        "content": results['documents'][i] if results.get('documents') else None,
                        "metadata": results['metadatas'][i] if results.get('metadatas') else {}
                    })
            
            return documents
        except Exception as e:
            print(f"Error getting documents for node {node_id}: {e}")
            return []
    
    def update_node_link(self, doc_id: str, node_id: str, node_type: Optional[str] = None) -> bool:
        """
        Update the graph node link for an existing document.
        
        Args:
            doc_id: The document ID to update
            node_id: The new node ID to link to
            node_type: Optional node type
            
        Returns:
            True if successful
        """
        try:
            # Get existing document
            results = self.vectorstore.get(ids=[doc_id])
            
            if not results or not results.get('ids'):
                return False
            
            # Update metadata
            metadata = results['metadatas'][0] if results.get('metadatas') else {}
            metadata['node_id'] = node_id
            if node_type:
                metadata['node_type'] = node_type
            
            # Chroma doesn't have a direct update, so we delete and re-add
            content = results['documents'][0] if results.get('documents') else ""
            
            self.vectorstore.delete(ids=[doc_id])
            self.vectorstore.add_texts(
                texts=[content],
                metadatas=[metadata],
                ids=[doc_id]
            )
            
            return True
        except Exception as e:
            print(f"Error updating node link for {doc_id}: {e}")
            return False
    
    def delete_session(self, session_id: str):
        """Delete all vectors for a specific session using a metadata filter."""
        try:
            # First, get the IDs of all documents matching the session_id
            results = self.vectorstore.get(where={"session_id": session_id})
            ids_to_delete = results.get('ids', [])
            
            if not ids_to_delete:
                print(f"No vectors found for session_id '{session_id}' to delete.")
                return

            # Now, delete the documents using their IDs
            self.vectorstore.delete(ids=ids_to_delete)
            print(f"Deleted {len(ids_to_delete)} vectors for session_id '{session_id}'.")
        except Exception as e:
            print(f"Error deleting vectors for session {session_id}: {e}")
