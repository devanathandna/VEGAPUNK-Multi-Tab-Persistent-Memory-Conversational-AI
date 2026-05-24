"""
VEGAPUNK Graph Store - NetworkX-based Knowledge Graph Management

This module provides graph-based knowledge storage and retrieval using NetworkX
with SQLite persistence. It handles:
- Node management (Concepts, Entities, Documents, Aliases)
- Relationship management (MENTIONS, CAUSES, RELATED_TO, etc.)
- Graph persistence to SQLite
- In-memory graph operations via NetworkX
"""

import sqlite3
import networkx as nx
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum


class NodeType(Enum):
    """Types of nodes in the knowledge graph"""
    CONCEPT = "Concept"
    ENTITY = "Entity"
    DOCUMENT = "Document"
    ALIAS = "Alias"
    MESSAGE = "Message"


# ── Dynamic relation weight lookup ────────────────────────────────────────────
# Relations are now free-form strings inferred by Groq AI.
# Known high-value types get explicit weights; anything Groq invents gets 0.60.
_KNOWN_RELATION_WEIGHTS: dict = {
    "CAUSES":           1.0,
    "IS_PARENT_OF":     0.95,
    "IS_ANCESTOR_OF":   0.92,
    "CONTAINS":         0.90,
    "IS_CHILD_OF":      0.88,
    "IS_SIBLING_OF":    0.85,
    "IS_SPOUSE_OF":     0.85,
    "FOUNDED":          0.83,
    "WORKS_AT":         0.82,
    "STUDIES_AT":       0.82,
    "LOCATED_IN":       0.80,
    "OWNS":             0.78,
    "KNOWS":            0.78,
    "MENTIONS":         0.75,
    "CREATED_BY":       0.73,
    "USES":             0.70,
    "DEPENDS_ON":       0.68,
    "RELATED_TO":       0.65,
    "SIMILAR_TO":       0.60,
    "PART_OF":          0.58,
    "DERIVED_FROM":     0.55,
    "OPPOSITE_OF":      0.50,
    "ALIAS_OF":         0.40,
}


def get_relation_weight(relation: str) -> float:
    """Return the semantic weight for any relation type string.

    Known types get their explicit weight.
    Any novel relation Groq invents defaults to 0.60 (above RELATED_TO baseline).
    """
    return _KNOWN_RELATION_WEIGHTS.get(relation.upper(), 0.60)


# Legacy alias kept for backward compatibility
RELATION_WEIGHTS = _KNOWN_RELATION_WEIGHTS


class GraphStore:
    """
    Knowledge Graph Store using NetworkX with SQLite persistence.
    
    SQLite is the source of truth. NetworkX graph is loaded in-memory
    for fast graph operations like traversal and PageRank.
    """
    
    def __init__(self, db_path: str = "vegapunk_memory.db"):
        self.db_path = db_path
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self._init_graph_tables()
        self._load_graph_from_db()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection"""
        return sqlite3.connect(self.db_path)
    
    def _init_graph_tables(self):
        """Initialize graph-related tables in SQLite"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Nodes table
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
        
        # Edges table
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
        
        # Indexes for faster lookups
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
    
    def _load_graph_from_db(self):
        """Load the graph from SQLite into NetworkX (in-memory)"""
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Clear existing graph
        self.graph.clear()
        
        # Load nodes
        cursor.execute("""
            SELECT node_id, type, label, importance, metadata, created_at 
            FROM graph_nodes
        """)
        for row in cursor.fetchall():
            self.graph.add_node(
                row['node_id'],
                type=row['type'],
                label=row['label'],
                importance=row['importance'],
                metadata=row['metadata'],
                created_at=row['created_at']
            )
        
        # Load edges
        cursor.execute("""
            SELECT id, source, target, relation, weight, metadata, created_at 
            FROM graph_edges
        """)
        for row in cursor.fetchall():
            self.graph.add_edge(
                row['source'],
                row['target'],
                key=row['id'],
                relation=row['relation'],
                weight=row['weight'],
                metadata=row['metadata'],
                created_at=row['created_at']
            )
        
        conn.close()
        print(f"[GraphStore] Loaded {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
    
    # ==================== NODE OPERATIONS ====================
    
    def add_node(
        self, 
        node_id: str, 
        node_type: str, 
        label: Optional[str] = None,
        importance: float = 0.5,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Add a node to the graph and persist to SQLite.
        
        Args:
            node_id: Unique identifier for the node
            node_type: Type of node (Concept, Entity, Document, Alias, Message)
            label: Human-readable label for the node
            importance: Importance score (0.0 to 1.0)
            metadata: Additional metadata as dict
            
        Returns:
            True if successful, False otherwise
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            import json
            metadata_json = json.dumps(metadata) if metadata else None
            
            cursor.execute("""
                INSERT OR REPLACE INTO graph_nodes 
                (node_id, type, label, importance, metadata, updated_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (node_id, node_type, label or node_id, importance, metadata_json))
            
            conn.commit()
            conn.close()
            
            # Update in-memory graph
            self.graph.add_node(
                node_id,
                type=node_type,
                label=label or node_id,
                importance=importance,
                metadata=metadata_json,
                created_at=datetime.now().isoformat()
            )
            
            return True
        except Exception as e:
            print(f"[GraphStore] Error adding node: {e}")
            return False
    
    def get_node(self, node_id: str) -> Optional[Dict]:
        """Get a node by its ID"""
        if node_id in self.graph:
            return {"node_id": node_id, **self.graph.nodes[node_id]}
        return None
    
    def get_nodes_by_type(self, node_type: str) -> List[Dict]:
        """Get all nodes of a specific type"""
        nodes = []
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get('type') == node_type:
                nodes.append({"node_id": node_id, **attrs})
        return nodes
    
    def update_node_importance(self, node_id: str, importance: float) -> bool:
        """Update the importance score of a node"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE graph_nodes 
                SET importance = ?, updated_at = CURRENT_TIMESTAMP
                WHERE node_id = ?
            """, (importance, node_id))
            
            conn.commit()
            conn.close()
            
            if node_id in self.graph:
                self.graph.nodes[node_id]['importance'] = importance
            
            return True
        except Exception as e:
            print(f"[GraphStore] Error updating node importance: {e}")
            return False
    
    def delete_node(self, node_id: str) -> bool:
        """Delete a node and all its edges"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Delete edges first
            cursor.execute("DELETE FROM graph_edges WHERE source = ? OR target = ?", 
                          (node_id, node_id))
            # Delete node
            cursor.execute("DELETE FROM graph_nodes WHERE node_id = ?", (node_id,))
            
            conn.commit()
            conn.close()
            
            # Update in-memory graph
            if node_id in self.graph:
                self.graph.remove_node(node_id)
            
            return True
        except Exception as e:
            print(f"[GraphStore] Error deleting node: {e}")
            return False
    
    # ==================== EDGE OPERATIONS ====================
    
    def add_edge(
        self,
        source: str,
        target: str,
        relation: str,
        weight: Optional[float] = None,
        metadata: Optional[Dict] = None
    ) -> Optional[int]:
        """
        Add an edge between two nodes.
        
        Args:
            source: Source node ID
            target: Target node ID
            relation: Type of relationship (MENTIONS, CAUSES, etc.)
            weight: Edge weight (if None, uses default from RELATION_WEIGHTS)
            metadata: Additional metadata
            
        Returns:
            Edge ID if successful, None otherwise
        """
        try:
            # Use dynamic weight lookup — handles any Groq-inferred relation type
            if weight is None:
                weight = get_relation_weight(relation)

            conn = self._get_connection()
            cursor = conn.cursor()

            # ── DEDUPLICATION: skip if identical (source, target, relation) already exists ──
            cursor.execute("""
                SELECT id FROM graph_edges
                WHERE source = ? AND target = ? AND relation = ?
                LIMIT 1
            """, (source, target, relation))
            existing = cursor.fetchone()
            if existing:
                conn.close()
                print(f"[GraphStore] Edge {source} -[{relation}]→ {target} already exists (id={existing[0]}), skipping duplicate")
                return existing[0]   # return existing id so callers still get a valid id

            import json
            metadata_json = json.dumps(metadata) if metadata else None

            cursor.execute("""
                INSERT INTO graph_edges (source, target, relation, weight, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (source, target, relation, weight, metadata_json))

            edge_id = cursor.lastrowid
            conn.commit()
            conn.close()

            # Update in-memory graph
            self.graph.add_edge(
                source,
                target,
                key=edge_id,
                relation=relation,
                weight=weight,
                metadata=metadata_json,
                created_at=datetime.now().isoformat()
            )

            return edge_id
        except Exception as e:
            print(f"[GraphStore] Error adding edge: {e}")
            return None
    
    def get_edges(self, node_id: str, direction: str = "both") -> List[Dict]:
        """
        Get edges connected to a node.
        
        Args:
            node_id: The node to get edges for
            direction: 'outgoing', 'incoming', or 'both'
            
        Returns:
            List of edge dictionaries
        """
        edges = []
        
        if direction in ("outgoing", "both"):
            for _, target, key, data in self.graph.out_edges(node_id, keys=True, data=True):
                edges.append({
                    "id": key,
                    "source": node_id,
                    "target": target,
                    **data
                })
        
        if direction in ("incoming", "both"):
            for source, _, key, data in self.graph.in_edges(node_id, keys=True, data=True):
                edges.append({
                    "id": key,
                    "source": source,
                    "target": node_id,
                    **data
                })
        
        return edges
    
    def get_edges_by_relation(self, relation: str) -> List[Dict]:
        """Get all edges of a specific relation type"""
        edges = []
        for source, target, key, data in self.graph.edges(keys=True, data=True):
            if data.get('relation') == relation:
                edges.append({
                    "id": key,
                    "source": source,
                    "target": target,
                    **data
                })
        return edges
    
    def delete_edge(self, edge_id: int) -> bool:
        """Delete an edge by its ID"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Get edge info first
            cursor.execute(
                "SELECT source, target FROM graph_edges WHERE id = ?", 
                (edge_id,)
            )
            row = cursor.fetchone()
            
            if row:
                cursor.execute("DELETE FROM graph_edges WHERE id = ?", (edge_id,))
                conn.commit()
                
                # Update in-memory graph
                source, target = row
                if self.graph.has_edge(source, target, key=edge_id):
                    self.graph.remove_edge(source, target, key=edge_id)
            
            conn.close()
            return True
        except Exception as e:
            print(f"[GraphStore] Error deleting edge: {e}")
            return False
    
    # ==================== GRAPH TRAVERSAL ====================
    
    def get_neighbors(
        self, 
        node_id: str, 
        hops: int = 1,
        direction: str = "both"
    ) -> Set[str]:
        """
        Get neighboring nodes within n hops.
        
        Args:
            node_id: Starting node
            hops: Maximum number of hops (1 or 2 recommended)
            direction: 'outgoing', 'incoming', or 'both'
            
        Returns:
            Set of neighboring node IDs
        """
        if node_id not in self.graph:
            return set()
        
        neighbors = set()
        
        if direction == "outgoing":
            paths = nx.single_source_shortest_path(self.graph, node_id, cutoff=hops)
        elif direction == "incoming":
            paths = nx.single_source_shortest_path(self.graph.reverse(), node_id, cutoff=hops)
        else:  # both
            # Get undirected view for bidirectional traversal
            undirected = self.graph.to_undirected()
            paths = nx.single_source_shortest_path(undirected, node_id, cutoff=hops)
        
        neighbors = set(paths.keys())
        neighbors.discard(node_id)  # Remove the starting node
        
        return neighbors
    
    def expand_nodes(
        self,
        seed_nodes: List[str],
        hops: int = 1,
        max_neighbors: int = 10,
        relation_filter: Optional[List[str]] = None
    ) -> Dict[str, Dict]:
        """
        Expand from seed nodes to get neighbors with detailed info.
        
        Args:
            seed_nodes: Starting nodes from semantic search
            hops: Number of hops (1 or 2)
            max_neighbors: Maximum neighbors per seed node
            relation_filter: Only follow these relation types (e.g., ['CAUSES', 'MENTIONS'])
            
        Returns:
            Dict mapping node_id to node info with expansion details
        """
        expanded = {}
        
        for seed in seed_nodes:
            if seed not in self.graph:
                continue
            
            # Add seed node
            if seed not in expanded:
                expanded[seed] = {
                    **self.graph.nodes[seed],
                    "node_id": seed,
                    "is_seed": True,
                    "hop_distance": 0,
                    "relations_from_seed": []
                }
            
            # Get neighbors at each hop level
            current_level = {seed}
            
            for hop in range(1, hops + 1):
                next_level = set()
                
                for node in current_level:
                    # Get outgoing edges
                    for _, target, data in self.graph.out_edges(node, data=True):
                        relation = data.get('relation', 'UNKNOWN')
                        
                        # Apply relation filter if specified
                        if relation_filter and relation not in relation_filter:
                            continue
                        
                        if target not in expanded:
                            expanded[target] = {
                                **self.graph.nodes.get(target, {}),
                                "node_id": target,
                                "is_seed": False,
                                "hop_distance": hop,
                                "relations_from_seed": []
                            }
                        
                        # Track how we got here
                        expanded[target]["relations_from_seed"].append({
                            "from": node,
                            "relation": relation,
                            "weight": data.get('weight', 0.5)
                        })
                        
                        next_level.add(target)
                    
                    # Get incoming edges
                    for source, _, data in self.graph.in_edges(node, data=True):
                        relation = data.get('relation', 'UNKNOWN')
                        
                        if relation_filter and relation not in relation_filter:
                            continue
                        
                        if source not in expanded:
                            expanded[source] = {
                                **self.graph.nodes.get(source, {}),
                                "node_id": source,
                                "is_seed": False,
                                "hop_distance": hop,
                                "relations_from_seed": []
                            }
                        
                        expanded[source]["relations_from_seed"].append({
                            "from": node,
                            "relation": f"INV_{relation}",  # Inverse relation
                            "weight": data.get('weight', 0.5)
                        })
                        
                        next_level.add(source)
                
                current_level = next_level
                
                # Limit neighbors per hop
                if len(expanded) > max_neighbors * len(seed_nodes):
                    break
        
        return expanded
    
    def get_expansion_context(
        self,
        seed_nodes: List[str],
        hops: int = 1
    ) -> Dict:
        """
        Build a context dictionary for LLM consumption.
        
        This is the format the LLM expects:
        {
            "nodes": ["A"],
            "neighbors": ["H", "B", "F"],
            "relations": [
                {"A": "MENTIONS → B"},
                {"A": "RELATED_TO → H"}
            ]
        }
        
        Args:
            seed_nodes: The seed nodes from semantic search
            hops: Number of hops to expand
            
        Returns:
            Context dict for LLM
        """
        context = {
            "nodes": seed_nodes,
            "neighbors": [],
            "relations": []
        }

        seen_neighbors = set()
        seen_relations: set = set()   # dedup: (source, relation_str, target)

        def _add_relation(src: str, rel_str: str):
            """Add relation only if not already seen."""
            key = (src, rel_str)
            if key not in seen_relations:
                seen_relations.add(key)
                context["relations"].append({src: rel_str})

        for seed in seed_nodes:
            if seed not in self.graph:
                continue

            # Get outgoing relations
            for _, target, data in self.graph.out_edges(seed, data=True):
                relation = data.get('relation', 'RELATED_TO')
                _add_relation(seed, f"{relation} → {target}")
                if target not in seed_nodes and target not in seen_neighbors:
                    seen_neighbors.add(target)

            # Get incoming relations
            for source, _, data in self.graph.in_edges(seed, data=True):
                relation = data.get('relation', 'RELATED_TO')
                _add_relation(source, f"{relation} → {seed}")
                if source not in seed_nodes and source not in seen_neighbors:
                    seen_neighbors.add(source)

            # 2-hop expansion if requested
            if hops >= 2:
                for neighbor in list(seen_neighbors):
                    if neighbor in self.graph:
                        for _, target2, data in self.graph.out_edges(neighbor, data=True):
                            if target2 not in seed_nodes:
                                relation = data.get('relation', 'RELATED_TO')
                                _add_relation(neighbor, f"{relation} → {target2}")
                                seen_neighbors.add(target2)

        context["neighbors"] = list(seen_neighbors)

        return context
    
    def get_relation_chain(
        self,
        source: str,
        target: str,
        max_depth: int = 3
    ) -> Optional[List[Dict]]:
        """
        Find the chain of relations between two nodes.
        
        Args:
            source: Starting node
            target: Target node
            max_depth: Maximum path length
            
        Returns:
            List of relation steps, or None if no path
        """
        try:
            path = nx.shortest_path(self.graph, source, target)
            
            if len(path) > max_depth + 1:
                return None
            
            chain = []
            for i in range(len(path) - 1):
                # Get edge data
                edge_data = self.graph.get_edge_data(path[i], path[i + 1])
                if edge_data:
                    # MultiDiGraph returns dict of edges
                    first_edge = list(edge_data.values())[0]
                    chain.append({
                        "from": path[i],
                        "to": path[i + 1],
                        "relation": first_edge.get('relation', 'RELATED_TO'),
                        "weight": first_edge.get('weight', 0.5)
                    })
            
            return chain
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
    
    def get_subgraph(self, node_ids: List[str]) -> Dict:
        """
        Get a subgraph containing only the specified nodes and their edges.
        
        Returns:
            Dict with 'nodes' and 'edges' lists
        """
        subgraph = self.graph.subgraph(node_ids)
        
        nodes = []
        for node_id in subgraph.nodes():
            nodes.append({"node_id": node_id, **subgraph.nodes[node_id]})
        
        edges = []
        for source, target, key, data in subgraph.edges(keys=True, data=True):
            edges.append({
                "id": key,
                "source": source,
                "target": target,
                **data
            })
        
        return {"nodes": nodes, "edges": edges}
    
    def get_path(self, source: str, target: str) -> Optional[List[str]]:
        """Get the shortest path between two nodes"""
        try:
            return nx.shortest_path(self.graph, source, target)
        except nx.NetworkXNoPath:
            return None
        except nx.NodeNotFound:
            return None
    
    # ==================== GRAPH ANALYTICS & RANKING ====================
    
    def compute_pagerank(self, alpha: float = 0.85) -> Dict[str, float]:
        """
        Compute PageRank for all nodes.
        
        Args:
            alpha: Damping factor (default 0.85)
            
        Returns:
            Dict mapping node_id to PageRank score
        """
        if self.graph.number_of_nodes() == 0:
            return {}
        
        try:
            return nx.pagerank(self.graph, alpha=alpha)
        except Exception as e:
            print(f"[GraphStore] Error computing PageRank: {e}")
            return {}
    
    def update_importance_from_pagerank(self):
        """Update all node importance scores based on PageRank"""
        pagerank = self.compute_pagerank()
        
        if not pagerank:
            return
        
        # Normalize PageRank scores to 0-1 range
        max_pr = max(pagerank.values())
        min_pr = min(pagerank.values())
        range_pr = max_pr - min_pr if max_pr != min_pr else 1.0
        
        for node_id, pr in pagerank.items():
            normalized = (pr - min_pr) / range_pr
            self.update_node_importance(node_id, normalized)
    
    def get_node_centrality(self, node_id: str) -> float:
        """Get the centrality score for a specific node"""
        pagerank = self.compute_pagerank()
        return pagerank.get(node_id, 0.0)
    
    def compute_structural_score(self, node_id: str) -> float:
        """
        Compute the structural score for a node based on:
        - PageRank centrality
        - Degree centrality
        - Edge weights
        
        Args:
            node_id: The node to score
            
        Returns:
            Structural score between 0 and 1
        """
        if node_id not in self.graph:
            return 0.0
        
        scores = []
        
        # 1. PageRank score (normalized)
        pagerank = self.compute_pagerank()
        if pagerank:
            max_pr = max(pagerank.values())
            min_pr = min(pagerank.values())
            range_pr = max_pr - min_pr if max_pr != min_pr else 1.0
            pr_score = (pagerank.get(node_id, 0) - min_pr) / range_pr
            scores.append(pr_score)
        
        # 2. Degree centrality (normalized)
        in_degree = self.graph.in_degree(node_id)
        out_degree = self.graph.out_degree(node_id)
        total_degree = in_degree + out_degree
        max_possible = self.graph.number_of_nodes() * 2
        degree_score = total_degree / max_possible if max_possible > 0 else 0
        scores.append(min(degree_score * 2, 1.0))  # Scale up, cap at 1
        
        # 3. Average edge weight
        edge_weights = []
        for _, _, data in self.graph.out_edges(node_id, data=True):
            edge_weights.append(data.get('weight', 0.5))
        for _, _, data in self.graph.in_edges(node_id, data=True):
            edge_weights.append(data.get('weight', 0.5))
        
        weight_score = sum(edge_weights) / len(edge_weights) if edge_weights else 0.5
        scores.append(weight_score)
        
        # Average all scores
        return sum(scores) / len(scores) if scores else 0.0
    
    def compute_relation_weight_score(self, node_id: str, from_node: Optional[str] = None) -> float:
        """
        Compute score based on relation types and their weights.
        
        Higher weight relations (CAUSES) contribute more than lower (ALIAS_OF).
        
        Args:
            node_id: The node to score
            from_node: If provided, only consider relations from this node
            
        Returns:
            Relation weight score between 0 and 1
        """
        if node_id not in self.graph:
            return 0.0
        
        weights = []
        
        # Check incoming edges
        for source, _, data in self.graph.in_edges(node_id, data=True):
            if from_node and source != from_node:
                continue
            relation = data.get('relation', 'RELATED_TO')
            weight = RELATION_WEIGHTS.get(relation, 0.5)
            weights.append(weight)
        
        # Check outgoing edges
        for _, target, data in self.graph.out_edges(node_id, data=True):
            if from_node and target != from_node:
                continue
            relation = data.get('relation', 'RELATED_TO')
            weight = RELATION_WEIGHTS.get(relation, 0.5)
            weights.append(weight)
        
        return sum(weights) / len(weights) if weights else 0.5
    
    def rank_nodes(
        self,
        node_ids: List[str],
        semantic_scores: Optional[Dict[str, float]] = None,
        timestamps: Optional[Dict[str, str]] = None,
        alpha: float = 0.4,
        beta: float = 0.4,
        gamma: float = 0.2
    ) -> List[Tuple[str, float, Dict]]:
        """
        Rank nodes using the fusion formula:
        final_score = α * semantic + β * structural + γ * recency
        
        Args:
            node_ids: List of node IDs to rank
            semantic_scores: Dict mapping node_id to semantic similarity score (0-1)
            timestamps: Dict mapping node_id to ISO timestamp string
            alpha: Weight for semantic score (default 0.4)
            beta: Weight for structural score (default 0.4)
            gamma: Weight for recency score (default 0.2)
            
        Returns:
            List of (node_id, final_score, score_breakdown) sorted by score descending
        """
        from datetime import datetime
        
        ranked = []
        
        for node_id in node_ids:
            # Semantic score (from Chroma similarity, inverted since lower is better in L2)
            if semantic_scores and node_id in semantic_scores:
                # Chroma uses L2 distance, so lower = better
                # Convert to 0-1 where 1 is best
                raw_semantic = semantic_scores[node_id]
                # Assuming typical L2 distances range from 0-2
                semantic = max(0, 1 - (raw_semantic / 2))
            else:
                semantic = 0.5  # Default if no semantic score
            
            # Structural score (PageRank + degree + edge weights)
            structural = self.compute_structural_score(node_id)
            
            # Recency score (based on timestamp)
            recency = 0.5  # Default
            if timestamps and node_id in timestamps:
                try:
                    ts = datetime.fromisoformat(timestamps[node_id].replace('Z', '+00:00'))
                    now = datetime.now(ts.tzinfo) if ts.tzinfo else datetime.now()
                    age_hours = (now - ts).total_seconds() / 3600
                    
                    # Decay function: recent = 1.0, 24h ago = 0.5, 1 week = 0.1
                    recency = max(0.1, 1.0 - (age_hours / 168))  # 168 hours = 1 week
                except Exception:
                    recency = 0.5
            
            # Fusion formula
            final_score = alpha * semantic + beta * structural + gamma * recency
            
            breakdown = {
                "semantic": round(semantic, 4),
                "structural": round(structural, 4),
                "recency": round(recency, 4),
                "weights": {"alpha": alpha, "beta": beta, "gamma": gamma}
            }
            
            ranked.append((node_id, final_score, breakdown))
        
        # Sort by final score (descending)
        ranked.sort(key=lambda x: x[1], reverse=True)
        
        return ranked
    
    def get_top_ranked_nodes(
        self,
        seed_node_ids: List[str],
        semantic_results: List[Dict],
        hops: int = 1,
        top_k: int = 10,
        alpha: float = 0.4,
        beta: float = 0.4,
        gamma: float = 0.2
    ) -> List[Dict]:
        """
        Get top-ranked nodes by combining semantic search with graph expansion.
        
        This is the main ranking function for Graph RAG queries.
        
        Args:
            seed_node_ids: Initial nodes from semantic search
            semantic_results: Results from vector_store.semantic_search()
            hops: Number of hops to expand
            top_k: Number of top results to return
            alpha, beta, gamma: Fusion weights
            
        Returns:
            List of ranked node dicts with scores
        """
        # Build semantic scores map from Chroma results
        semantic_scores = {}
        timestamps = {}
        
        for result in semantic_results:
            node_id = result.get("node_id")
            if node_id:
                semantic_scores[node_id] = result.get("similarity_score", 1.0)
                if result.get("metadata", {}).get("timestamp"):
                    timestamps[node_id] = result["metadata"]["timestamp"]
        
        # Expand to get neighbors
        all_nodes = set(seed_node_ids)
        for seed in seed_node_ids:
            neighbors = self.get_neighbors(seed, hops=hops)
            all_nodes.update(neighbors)
        
        # Assign default semantic scores to expanded nodes (not directly matched)
        for node_id in all_nodes:
            if node_id not in semantic_scores:
                # Neighbors get a penalized semantic score
                semantic_scores[node_id] = 1.5  # Higher = worse in L2 distance
        
        # Rank all nodes
        ranked = self.rank_nodes(
            list(all_nodes),
            semantic_scores=semantic_scores,
            timestamps=timestamps,
            alpha=alpha,
            beta=beta,
            gamma=gamma
        )
        
        # Build result with node details
        results = []
        for node_id, score, breakdown in ranked[:top_k]:
            node_data = self.get_node(node_id) or {"node_id": node_id}
            results.append({
                **node_data,
                "final_score": round(score, 4),
                "score_breakdown": breakdown,
                "is_seed": node_id in seed_node_ids
            })
        
        return results
    
    # ==================== UTILITY METHODS ====================
    
    def get_stats(self) -> Dict:
        """Get graph statistics"""
        return {
            "node_count": self.graph.number_of_nodes(),
            "edge_count": self.graph.number_of_edges(),
            "node_types": self._count_by_attribute("type"),
            "relation_types": self._count_edges_by_relation(),
            "is_connected": nx.is_weakly_connected(self.graph) if self.graph.number_of_nodes() > 0 else True
        }
    
    def _count_by_attribute(self, attr: str) -> Dict[str, int]:
        """Count nodes by a specific attribute"""
        counts = {}
        for _, attrs in self.graph.nodes(data=True):
            value = attrs.get(attr, "unknown")
            counts[value] = counts.get(value, 0) + 1
        return counts
    
    def _count_edges_by_relation(self) -> Dict[str, int]:
        """Count edges by relation type"""
        counts = {}
        for _, _, data in self.graph.edges(data=True):
            relation = data.get('relation', 'unknown')
            counts[relation] = counts.get(relation, 0) + 1
        return counts
    
    def clear_graph(self) -> bool:
        """Clear all nodes and edges from the graph"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM graph_edges")
            cursor.execute("DELETE FROM graph_nodes")
            
            conn.commit()
            conn.close()
            
            self.graph.clear()
            
            return True
        except Exception as e:
            print(f"[GraphStore] Error clearing graph: {e}")
            return False
    
    def refresh_from_db(self):
        """Reload the graph from SQLite (useful if external changes were made)"""
        self._load_graph_from_db()


# ==================== TESTING ====================

if __name__ == "__main__":
    # Quick test
    store = GraphStore(db_path="test_graph.db")
    
    # Add some test nodes
    store.add_node("python", NodeType.CONCEPT.value, label="Python Programming")
    store.add_node("flask", NodeType.ENTITY.value, label="Flask Framework")
    store.add_node("django", NodeType.ENTITY.value, label="Django Framework")
    store.add_node("web_dev", NodeType.CONCEPT.value, label="Web Development")
    
    # Add relationships
    store.add_edge("flask", "python", RelationType.DERIVED_FROM.value)
    store.add_edge("django", "python", RelationType.DERIVED_FROM.value)
    store.add_edge("flask", "web_dev", RelationType.RELATED_TO.value)
    store.add_edge("django", "web_dev", RelationType.RELATED_TO.value)
    
    # Test traversal
    print("\n--- Graph Stats ---")
    print(store.get_stats())
    
    print("\n--- Neighbors of 'python' (1 hop) ---")
    print(store.get_neighbors("python", hops=1))
    
    print("\n--- Neighbors of 'python' (2 hops) ---")
    print(store.get_neighbors("python", hops=2))
    
    print("\n--- PageRank ---")
    print(store.compute_pagerank())
    
    print("\n--- Path from 'flask' to 'django' ---")
    print(store.get_path("flask", "django"))
    
    # Cleanup
    import os
    os.remove("test_graph.db")
    print("\n✅ All tests passed!")
