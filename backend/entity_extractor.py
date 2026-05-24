"""
VEGAPUNK Entity Extractor - Automatic Knowledge Graph Population

This module uses LLM to extract entities, concepts, and relationships
from text to automatically populate the knowledge graph.

Extraction pipeline:
1. Text Input (message, document, etc.)
2. LLM Analysis (extract entities + relations)
3. Graph Population (add nodes + edges)
4. Vector Linking (connect embeddings to graph nodes)
"""

from groq import Groq
from typing import List, Dict, Optional, Tuple
import json
import re
import hashlib
from dataclasses import dataclass, field
from enum import Enum


class ExtractionMode(Enum):
    """Extraction detail levels"""
    MINIMAL = "minimal"      # Only key entities
    STANDARD = "standard"    # Entities + main relationships
    COMPREHENSIVE = "comprehensive"  # Everything including implicit relations


@dataclass
class ExtractedEntity:
    """An extracted entity or concept"""
    id: str
    name: str
    type: str  # Concept, Entity, Person, Project, Technology, etc.
    description: Optional[str] = None
    importance: float = 0.5
    aliases: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "importance": self.importance,
            "aliases": self.aliases
        }


@dataclass
class ExtractedRelation:
    """An extracted relationship between entities"""
    source_id: str
    target_id: str
    relation_type: str  # MENTIONS, CAUSES, RELATED_TO, etc.
    description: Optional[str] = None
    confidence: float = 0.8
    
    def to_dict(self) -> Dict:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type,
            "description": self.description,
            "confidence": self.confidence
        }


@dataclass
class ExtractionResult:
    """Result of entity extraction"""
    entities: List[ExtractedEntity]
    relations: List[ExtractedRelation]
    source_text: str
    extraction_mode: str
    
    def to_dict(self) -> Dict:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "relations": [r.to_dict() for r in self.relations],
            "entity_count": len(self.entities),
            "relation_count": len(self.relations),
            "extraction_mode": self.extraction_mode
        }


class EntityExtractor:
    """
    Extracts entities and relationships from text using LLM.
    
    Usage:
        extractor = EntityExtractor(api_key="...")
        result = extractor.extract("Python is a programming language created by Guido van Rossum")
        # result.entities = [Python (Technology), Guido van Rossum (Person)]
        # result.relations = [(Python, CREATED_BY, Guido van Rossum)]
    """
    
    # Valid relation types that map to graph_store RelationType
    # Example relation types shown to the LLM as guidance — NOT enforced as a whitelist.
    # Groq is free to output any semantically appropriate UPPER_SNAKE_CASE relation.
    EXAMPLE_RELATIONS = [
        # General
        "MENTIONS", "CAUSES", "RELATED_TO", "ALIAS_OF",
        "CONTAINS", "PART_OF", "DERIVED_FROM", "CREATED_BY",
        "USES", "DEPENDS_ON", "SIMILAR_TO", "OPPOSITE_OF",
        # Family / social
        "IS_PARENT_OF", "IS_CHILD_OF", "IS_SIBLING_OF",
        "IS_SPOUSE_OF", "IS_ANCESTOR_OF", "KNOWS",
        # Location / affiliation
        "WORKS_AT", "STUDIES_AT", "LOCATED_IN",
    ]
    
    # Valid entity types
    VALID_ENTITY_TYPES = [
        "Concept", "Entity", "Person", "Organization", "Project",
        "Technology", "Location", "Event", "Document", "Topic"
    ]
    
    def __init__(self, api_key: str, model: str = "llama-3.1-8b-instant"):
        """Initialize the extractor with a Groq API client."""
        self.api_key = api_key
        self.model = model
        self.client = Groq(api_key=api_key)

    def _get_client(self) -> Groq:
        """Return the Groq client (kept for structural compatibility)."""
        return self.client
    
    def extract(
        self,
        text: str,
        mode: ExtractionMode = ExtractionMode.STANDARD,
        context: Optional[str] = None
    ) -> ExtractionResult:
        """
        Extract entities and relationships from text.
        
        Args:
            text: The text to extract from
            mode: Extraction detail level
            context: Optional context about the conversation/document
            
        Returns:
            ExtractionResult with entities and relations
        """
        if not text or len(text.strip()) < 10:
            return ExtractionResult(
                entities=[],
                relations=[],
                source_text=text,
                extraction_mode=mode.value
            )
        
        prompt = self._build_extraction_prompt(text, mode, context)

        try:
            print(f"\n📤 [ENTITY EXTRACTOR] Groq API CALL")
            print(f"   Model: {self.model}")
            print(f"   API Key: ...{self.api_key[-8:]}")
            print(f"   Text length: {len(text)} chars")
            print(f"   Mode: {mode.value}")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0
            )
            result_text = response.choices[0].message.content.strip()

            print(f"   ✓ Response received: {len(result_text)} chars")

            entities, relations = self._parse_extraction_response(result_text)

            return ExtractionResult(
                entities=entities,
                relations=relations,
                source_text=text,
                extraction_mode=mode.value
            )

        except Exception as e:
            print(f"⚠️ Entity extraction error: {e}")
            return ExtractionResult(
                entities=[],
                relations=[],
                source_text=text,
                extraction_mode=mode.value
            )
    
    def extract_and_populate(
        self,
        text: str,
        graph_store,
        vector_store,
        session_id: Optional[str] = None,
        mode: ExtractionMode = ExtractionMode.STANDARD
    ) -> ExtractionResult:
        """
        Extract entities and directly populate the graph and vector stores.
        
        Args:
            text: Text to extract from
            graph_store: GraphStore instance
            vector_store: VegapunkVectorStore instance
            session_id: Optional session for scoping
            mode: Extraction detail level
            
        Returns:
            ExtractionResult with what was extracted
        """
        result = self.extract(text, mode)
        
        if not result.entities:
            return result
        
        # Add entities to graph
        for entity in result.entities:
            graph_store.add_node(
                node_id=entity.id,
                node_type=entity.type,
                label=entity.name,
                importance=entity.importance,
                metadata={
                    "description": entity.description,
                    "aliases": entity.aliases,
                    "session_id": session_id
                }
            )
            
            # Add to vector store with graph link
            content = entity.description or entity.name
            doc_id = f"entity_{entity.id}"
            
            try:
                vector_store.add_document(
                    doc_id=doc_id,
                    content=content,
                    node_id=entity.id,
                    node_type=entity.type,
                    metadata={"session_id": session_id} if session_id else None
                )
            except Exception as e:
                print(f"⚠️ Error adding entity to vector store: {e}")
        
        # Add relationships to graph
        for relation in result.relations:
            # Only add if both nodes exist
            if graph_store.get_node(relation.source_id) and graph_store.get_node(relation.target_id):
                graph_store.add_edge(
                    source=relation.source_id,
                    target=relation.target_id,
                    relation=relation.relation_type,
                    metadata={
                        "description": relation.description,
                        "confidence": relation.confidence
                    }
                )
        
        print(f"✓ Extracted and added: {len(result.entities)} entities, {len(result.relations)} relations")
        
        return result
    
    def _build_extraction_prompt(
        self,
        text: str,
        mode: ExtractionMode,
        context: Optional[str]
    ) -> str:
        """Build the extraction prompt for the LLM."""
        
        mode_instructions = {
            ExtractionMode.MINIMAL: "Extract only the most important entities (max 3). Skip common/generic terms.",
            ExtractionMode.STANDARD: "Extract key entities and their main relationships (max 5 entities, max 5 relations).",
            ExtractionMode.COMPREHENSIVE: "Extract all entities, concepts, and relationships including implicit ones (max 10 entities, max 10 relations)."
        }
        
        context_section = f"\nContext: {context}\n" if context else ""
        
        prompt = f"""Extract entities and relationships from the following text.
{context_section}
Text: "{text}"

Instructions: {mode_instructions[mode]}

Valid entity types: {', '.join(self.VALID_ENTITY_TYPES)}

Respond with ONLY valid JSON in this exact format:
{{
  "entities": [
    {{
      "name": "Entity Name",
      "type": "Concept|Entity|Person|Organization|Project|Technology|Location|Event|Document|Topic",
      "description": "Brief description",
      "importance": 0.5,
      "aliases": ["alias1", "alias2"]
    }}
  ],
  "relations": [
    {{
      "source": "Source Entity Name",
      "target": "Target Entity Name",
      "relation": "YOUR_INFERRED_RELATION_TYPE",
      "description": "Brief description of the relationship"
    }}
  ]
}}

Rules:
1. Entity names should be normalized (proper case, no extra whitespace)
2. Use existing entity names as source/target in relations
3. importance: 0.3 (minor mention) to 1.0 (central topic)
4. Only include entities that are meaningful for knowledge retrieval
5. Skip generic terms like "it", "this", "thing"
6. For the "relation" field: INVENT the most semantically precise UPPER_SNAKE_CASE
   relation type that describes the actual relationship. Do NOT default to RELATED_TO
   unless nothing more specific fits.
   Examples of good relation types:
   - "A is the father of B"   → IS_PARENT_OF
   - "B is the child of A"    → IS_CHILD_OF
   - "X and Y are siblings"   → IS_SIBLING_OF
   - "Alice works at Acme"    → WORKS_AT
   - "Bob studies at MIT"     → STUDIES_AT
   - "Paris is in France"     → LOCATED_IN
   - "X causes Y"             → CAUSES
   - "A owns B"               → OWNS
   - "A married B"            → IS_SPOUSE_OF
   - "A founded B"            → FOUNDED
   You are NOT limited to these — use any descriptive UPPER_SNAKE_CASE string.

JSON:"""
        
        return prompt
    
    def _parse_extraction_response(
        self,
        response_text: str
    ) -> Tuple[List[ExtractedEntity], List[ExtractedRelation]]:
        """Parse the LLM response into structured entities and relations."""
        
        entities = []
        relations = []
        
        try:
            # Clean the response - remove markdown code blocks if present
            cleaned = response_text.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r'^```(?:json)?\n?', '', cleaned)
                cleaned = re.sub(r'\n?```$', '', cleaned)
            
            data = json.loads(cleaned)
            
            # Parse entities
            entity_name_to_id = {}  # Map names to IDs for relation linking
            
            for e in data.get("entities", []):
                name = e.get("name", "").strip()
                if not name:
                    continue
                
                # Generate stable ID from name
                entity_id = self._generate_entity_id(name)
                entity_name_to_id[name.lower()] = entity_id
                
                # Validate and normalize type
                entity_type = e.get("type", "Concept")
                if entity_type not in self.VALID_ENTITY_TYPES:
                    entity_type = "Concept"
                
                entities.append(ExtractedEntity(
                    id=entity_id,
                    name=name,
                    type=entity_type,
                    description=e.get("description"),
                    importance=min(1.0, max(0.1, float(e.get("importance", 0.5)))),
                    aliases=e.get("aliases", [])
                ))
            
            # Parse relations
            for r in data.get("relations", []):
                source_name = r.get("source", "").strip().lower()
                target_name = r.get("target", "").strip().lower()
                
                source_id = entity_name_to_id.get(source_name)
                target_id = entity_name_to_id.get(target_name)
                
                if not source_id or not target_id:
                    continue
                
                # Sanitize relation type: UPPER_SNAKE_CASE, no spaces.
                # No whitelist — Groq freely infers semantically accurate relation types.
                relation_type = r.get("relation", "RELATED_TO").upper().strip()
                relation_type = re.sub(r'\s+', '_', relation_type)          # spaces → underscores
                relation_type = re.sub(r'[^A-Z0-9_]', '', relation_type)    # strip non-alphanumeric
                if not relation_type:
                    relation_type = "RELATED_TO"

                relations.append(ExtractedRelation(
                    source_id=source_id,
                    target_id=target_id,
                    relation_type=relation_type,
                    description=r.get("description"),
                    confidence=0.8
                ))
            
        except json.JSONDecodeError as e:
            print(f"⚠️ Failed to parse extraction response: {e}")
        except Exception as e:
            print(f"⚠️ Error processing extraction: {e}")
        
        return entities, relations
    
    def _generate_entity_id(self, name: str) -> str:
        """Generate a stable, URL-safe ID from entity name."""
        # Normalize the name
        normalized = name.lower().strip()
        normalized = re.sub(r'[^a-z0-9\s]', '', normalized)
        normalized = re.sub(r'\s+', '_', normalized)
        
        # Add a short hash for uniqueness
        hash_suffix = hashlib.md5(name.encode()).hexdigest()[:6]
        
        return f"{normalized}_{hash_suffix}"
    
    def batch_extract(
        self,
        texts: List[str],
        mode: ExtractionMode = ExtractionMode.STANDARD
    ) -> List[ExtractionResult]:
        """
        Extract from multiple texts.
        
        Args:
            texts: List of texts to extract from
            mode: Extraction detail level
            
        Returns:
            List of ExtractionResult
        """
        results = []
        for text in texts:
            result = self.extract(text, mode)
            results.append(result)
        return results
    
    def merge_extractions(
        self,
        results: List[ExtractionResult]
    ) -> ExtractionResult:
        """
        Merge multiple extraction results, deduplicating entities.
        
        Args:
            results: List of ExtractionResult to merge
            
        Returns:
            Merged ExtractionResult
        """
        seen_entity_ids = set()
        seen_relation_keys = set()
        
        merged_entities = []
        merged_relations = []
        
        for result in results:
            for entity in result.entities:
                if entity.id not in seen_entity_ids:
                    merged_entities.append(entity)
                    seen_entity_ids.add(entity.id)
            
            for relation in result.relations:
                key = f"{relation.source_id}_{relation.relation_type}_{relation.target_id}"
                if key not in seen_relation_keys:
                    merged_relations.append(relation)
                    seen_relation_keys.add(key)
        
        return ExtractionResult(
            entities=merged_entities,
            relations=merged_relations,
            source_text="[merged]",
            extraction_mode="merged"
        )


class ConversationExtractor:
    """
    Specialized extractor for conversation messages.
    Tracks entities across a conversation for better context.
    """
    
    def __init__(self, api_key: str, model: str = "llama-3.1-8b-instant"):
        self.extractor = EntityExtractor(api_key, model=model)
        self.session_entities: Dict[str, set] = {}  # session_id -> entity_ids
    
    def process_message(
        self,
        message: str,
        session_id: str,
        graph_store,
        vector_store,
        role: str = "user"
    ) -> ExtractionResult:
        """
        Process a conversation message and extract/link entities.
        
        Args:
            message: The message text
            session_id: Conversation session ID
            graph_store: GraphStore instance
            vector_store: VegapunkVectorStore instance
            role: 'user' or 'model'
            
        Returns:
            ExtractionResult
        """
        # Use minimal mode for short messages, standard for longer
        mode = ExtractionMode.MINIMAL if len(message) < 100 else ExtractionMode.STANDARD
        
        # Get existing entities for context
        existing_entities = self.session_entities.get(session_id, set())
        context = None
        if existing_entities:
            context = f"Previously mentioned entities in this conversation: {', '.join(list(existing_entities)[:10])}"
        
        result = self.extractor.extract(message, mode, context)
        
        if result.entities:
            # Track new entities
            if session_id not in self.session_entities:
                self.session_entities[session_id] = set()
            
            for entity in result.entities:
                self.session_entities[session_id].add(entity.name)
                
                # Add to graph with session context
                graph_store.add_node(
                    node_id=entity.id,
                    node_type=entity.type,
                    label=entity.name,
                    importance=entity.importance,
                    metadata={
                        "description": entity.description,
                        "session_id": session_id,
                        "source_role": role
                    }
                )
            
            # Add relations
            for relation in result.relations:
                if graph_store.get_node(relation.source_id) and graph_store.get_node(relation.target_id):
                    graph_store.add_edge(
                        source=relation.source_id,
                        target=relation.target_id,
                        relation=relation.relation_type
                    )
        
        return result
    
    def clear_session(self, session_id: str):
        """Clear tracked entities for a session."""
        if session_id in self.session_entities:
            del self.session_entities[session_id]


# ==================== TESTING ====================

if __name__ == "__main__":
    import os
    
    api_key = os.getenv("GROQ_API_KEY_1")
    if not api_key:
        print("Set GROQ_API_KEY_1 environment variable to test")
        exit(1)
    
    extractor = EntityExtractor(api_key)
    
    test_text = """
    Python is a programming language created by Guido van Rossum at CWI in the Netherlands.
    It was first released in 1991. Python is used extensively in machine learning and data science.
    Popular frameworks like TensorFlow and PyTorch are built on Python.
    """
    
    print("Testing entity extraction...")
    result = extractor.extract(test_text, ExtractionMode.STANDARD)
    
    print("\n--- Extracted Entities ---")
    for entity in result.entities:
        print(f"  • {entity.name} ({entity.type}) - importance: {entity.importance}")
    
    print("\n--- Extracted Relations ---")
    for relation in result.relations:
        print(f"  • {relation.source_id} --[{relation.relation_type}]--> {relation.target_id}")
    
    print("\n✅ Entity extraction test complete!")
