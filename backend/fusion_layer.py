"""
VEGAPUNK Fusion Layer - The SECRET SAUCE

This module combines semantic similarity from Chroma with structural 
information from the knowledge graph to produce optimally ranked results.

The fusion formula:
    final_score = α * semantic + β * structural + γ * recency

Default weights: α=0.4, β=0.4, γ=0.2
"""

from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
import json


@dataclass
class FusionConfig:
    """Configuration for the fusion layer"""
    # Fusion weights (must sum to 1.0)
    alpha: float = 0.4  # Semantic weight
    beta: float = 0.4   # Structural weight
    gamma: float = 0.2  # Recency weight
    
    # Expansion settings
    expansion_hops: int = 1
    max_neighbors: int = 10
    
    # Ranking settings
    top_k: int = 10
    min_score_threshold: float = 0.1
    
    # Recency decay settings
    recency_decay_hours: float = 168.0  # 1 week
    recency_min_score: float = 0.1
    
    def validate(self) -> bool:
        """Validate that weights sum to 1.0"""
        total = self.alpha + self.beta + self.gamma
        return abs(total - 1.0) < 0.001


@dataclass
class FusionResult:
    """Result from the fusion layer"""
    node_id: str
    final_score: float
    semantic_score: float
    structural_score: float
    recency_score: float
    is_seed: bool
    hop_distance: int
    content: Optional[str] = None
    node_type: Optional[str] = None
    label: Optional[str] = None
    relations: List[Dict] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "node_id": self.node_id,
            "final_score": round(self.final_score, 4),
            "scores": {
                "semantic": round(self.semantic_score, 4),
                "structural": round(self.structural_score, 4),
                "recency": round(self.recency_score, 4)
            },
            "is_seed": self.is_seed,
            "hop_distance": self.hop_distance,
            "content": self.content,
            "node_type": self.node_type,
            "label": self.label,
            "relations": self.relations,
            "metadata": self.metadata
        }


class FusionLayer:
    """
    Fusion Layer for Graph RAG
    
    Combines three sources of relevance:
    1. Semantic similarity (from Chroma/vector store)
    2. Structural importance (from NetworkX graph)
    3. Recency (from timestamps)
    
    Usage:
        fusion = FusionLayer(graph_store, vector_store)
        results = fusion.query("What causes climate change?")
    """
    
    def __init__(
        self, 
        graph_store, 
        vector_store,
        config: Optional[FusionConfig] = None
    ):
        """
        Initialize the fusion layer.
        
        Args:
            graph_store: GraphStore instance
            vector_store: VegapunkVectorStore instance
            config: FusionConfig with custom settings
        """
        self.graph = graph_store
        self.vector = vector_store
        self.config = config or FusionConfig()
        
        if not self.config.validate():
            raise ValueError("Fusion weights must sum to 1.0")
    
    def query(
        self,
        query_text: str,
        session_id: Optional[str] = None,
        config: Optional[FusionConfig] = None
    ) -> List[FusionResult]:
        """
        Main query method - combines semantic search with graph expansion.
        
        Args:
            query_text: The user's query
            session_id: Optional session filter for Chroma
            config: Override default config for this query
            
        Returns:
            List of FusionResult sorted by final_score
        """
        cfg = config or self.config

        print("\n" + "=" * 60)
        print("[FUSION STEP 1] ✨ FusionLayer.query() CALLED")
        print(f"               query      : {query_text[:80]}{'...' if len(query_text)>80 else ''}")
        print(f"               session_id : {session_id or 'None'}")
        print(f"               α (semantic)  = {cfg.alpha}")
        print(f"               β (structural) = {cfg.beta}")
        print(f"               γ (recency)    = {cfg.gamma}")
        print(f"               top_k      : {cfg.top_k}")
        print(f"               min_score  : {cfg.min_score_threshold}")
        print(f"               hops       : {cfg.expansion_hops}")
        print(f"               max_nbrs   : {cfg.max_neighbors}")
        print("=" * 60)

        # Step 1: Semantic search via Chroma
        semantic_results = self.vector.semantic_search(
            query=query_text,
            session_id=session_id,
            k=cfg.top_k * 2,  # Get more for filtering
            include_node_ids=True
        )

        print(f"\n[FUSION STEP 2] 🔍 Semantic search returned {len(semantic_results)} docs (with node_id links)")

        # Extract seed nodes (those with node_id links)
        seed_nodes = []
        semantic_scores = {}
        timestamps = {}
        content_map = {}

        for result in semantic_results:
            node_id = result.get("node_id")
            if node_id:
                seed_nodes.append(node_id)
                semantic_scores[node_id] = result.get("similarity_score", 1.0)
                content_map[node_id] = result.get("content")

                ts = result.get("metadata", {}).get("timestamp")
                if ts:
                    timestamps[node_id] = ts

        print(f"[FUSION STEP 3] 🌱 Seed nodes extracted: {len(seed_nodes)}")
        for nid in seed_nodes[:8]:
            print(f"               ‣ {nid}  (raw_score={semantic_scores.get(nid, '?'):.4f})")

        if not seed_nodes:
            # No graph-linked results, return semantic results as-is
            print("[FUSION STEP 3] ⚠️  No seed nodes found — falling back to semantic-only path")
            print("=" * 60 + "\n")
            return self._convert_semantic_only(semantic_results, cfg)

        # Step 2: Graph expansion
        expanded = self.graph.expand_nodes(
            seed_nodes=seed_nodes,
            hops=cfg.expansion_hops,
            max_neighbors=cfg.max_neighbors
        )

        print(f"\n[FUSION STEP 4] 🕸️  Graph expansion: {len(expanded)} total nodes (seeds + neighbors)")
        seed_count = sum(1 for v in expanded.values() if v.get("is_seed"))
        nbr_count  = len(expanded) - seed_count
        print(f"               ├─ seed nodes    : {seed_count}")
        print(f"               └─ neighbor nodes: {nbr_count}")

        # Step 3: Compute scores and fuse
        print(f"\n[FUSION STEP 5] 🧩 Scoring all {len(expanded)} nodes …")
        print(f"               Formula: final = {cfg.alpha}*semantic + {cfg.beta}*structural + {cfg.gamma}*recency")
        results = []
        skipped_threshold = 0

        for node_id, node_info in expanded.items():
            # Semantic score (convert L2 distance to similarity)
            raw_semantic = semantic_scores.get(node_id, 1.5)  # Default penalty for non-seeds
            semantic = self._normalize_semantic_score(raw_semantic)

            # Structural score
            structural = self.graph.compute_structural_score(node_id)

            # Recency score
            recency = self._compute_recency_score(timestamps.get(node_id), cfg)

            # Fusion formula
            final_score = (
                cfg.alpha * semantic +
                cfg.beta * structural +
                cfg.gamma * recency
            )

            is_seed = node_info.get("is_seed", False)
            print(f"               {'[SEED]' if is_seed else '[NBR] '} {node_id[:40]:40s}"
                  f"  sem={semantic:.3f}  str={structural:.3f}  rec={recency:.3f}  => final={final_score:.3f}"
                  f"{'  ❌ SKIP' if final_score < cfg.min_score_threshold else ''}")

            # Skip if below threshold
            if final_score < cfg.min_score_threshold:
                skipped_threshold += 1
                continue

            # Get relations for context
            relations = self._get_node_relations(node_id)

            result = FusionResult(
                node_id=node_id,
                final_score=final_score,
                semantic_score=semantic,
                structural_score=structural,
                recency_score=recency,
                is_seed=is_seed,
                hop_distance=node_info.get("hop_distance", 0),
                content=content_map.get(node_id),
                node_type=node_info.get("type"),
                label=node_info.get("label"),
                relations=relations,
                metadata={"importance": node_info.get("importance", 0.5)}
            )

            results.append(result)

        # Sort by final score (descending)
        results.sort(key=lambda x: x.final_score, reverse=True)

        print(f"\n[FUSION STEP 6] 🏆 Ranking complete")
        print(f"               Total scored  : {len(expanded)}")
        print(f"               Skipped (<{cfg.min_score_threshold}) : {skipped_threshold}")
        print(f"               Kept          : {len(results)}")
        print(f"               Returned (top_k={cfg.top_k}): {min(len(results), cfg.top_k)}")
        print(f"               Top results:")
        for rank, r in enumerate(results[:cfg.top_k], 1):
            print(f"               #{rank:02d}  [{r.node_id[:35]:35s}]  score={r.final_score:.4f}"
                  f"  type={r.node_type or '?'}  seed={r.is_seed}  hop={r.hop_distance}")
        print("=" * 60 + "\n")

        return results[:cfg.top_k]
    
    def build_llm_context(
        self,
        results: List[FusionResult],
        include_relations: bool = True,
        include_scores: bool = False
    ) -> Dict:
        """
        Build the context dictionary for LLM consumption.
        
        Format:
        {
            "nodes": ["A"],
            "neighbors": ["H", "B", "F"],
            "relations": [
                {"A": "MENTIONS → B"},
                {"A": "RELATED_TO → H"}
            ],
            "context_texts": ["content of A", "content of B"],
            "reasoning_hints": ["A is highly connected", "B causes C"]
        }
        
        Args:
            results: List of FusionResult from query()
            include_relations: Whether to include relation info
            include_scores: Whether to include score breakdowns
            
        Returns:
            Context dict for LLM
        """
        seed_nodes = [r.node_id for r in results if r.is_seed]
        neighbor_nodes = [r.node_id for r in results if not r.is_seed]
        
        context = {
            "nodes": seed_nodes,
            "neighbors": neighbor_nodes,
            "relations": [],
            "context_texts": [],
            "reasoning_hints": []
        }
        
        for result in results:
            # Add content if available
            if result.content:
                context["context_texts"].append({
                    "node": result.node_id,
                    "text": result.content
                })
            
            # Add relations
            if include_relations:
                for rel in result.relations:
                    context["relations"].append(rel)
            
            # Add reasoning hints based on scores
            if result.structural_score > 0.7:
                context["reasoning_hints"].append(
                    f"{result.node_id} is a highly connected concept (structural={result.structural_score:.2f})"
                )
            
            if include_scores:
                context["reasoning_hints"].append(
                    f"{result.node_id}: semantic={result.semantic_score:.2f}, "
                    f"structural={result.structural_score:.2f}, recency={result.recency_score:.2f}"
                )
        
        return context
    
    def build_prompt_context(self, results: List[FusionResult]) -> str:
        """
        Build a text context string suitable for LLM prompts.
        
        Args:
            results: List of FusionResult from query()
            
        Returns:
            Formatted context string
        """
        lines = []
        lines.append("=== Retrieved Knowledge ===\n")
        
        # Group by seed vs neighbor
        seeds = [r for r in results if r.is_seed]
        neighbors = [r for r in results if not r.is_seed]
        
        if seeds:
            lines.append("**Direct Matches:**")
            for r in seeds:
                lines.append(f"• [{r.node_type or 'Node'}] {r.label or r.node_id}")
                if r.content:
                    lines.append(f"  Content: {r.content[:200]}...")
        
        if neighbors:
            lines.append("\n**Related Concepts:**")
            for r in neighbors[:5]:  # Limit neighbors
                lines.append(f"• [{r.node_type or 'Node'}] {r.label or r.node_id} (via {r.hop_distance}-hop)")
        
        # Add key relations
        all_relations = []
        for r in results:
            all_relations.extend(r.relations)
        
        if all_relations:
            lines.append("\n**Relationships:**")
            for rel in all_relations[:10]:  # Limit relations
                for source, desc in rel.items():
                    lines.append(f"• {source}: {desc}")
        
        return "\n".join(lines)
    
    def _normalize_semantic_score(self, l2_distance: float) -> float:
        """
        Convert L2 distance to similarity score (0-1, higher is better).
        
        Chroma uses L2 distance where lower = more similar.
        Typical range is 0-2.
        """
        # Clamp and invert
        similarity = max(0, 1 - (l2_distance / 2))
        return min(1.0, similarity)
    
    def _compute_recency_score(
        self, 
        timestamp: Optional[str], 
        cfg: FusionConfig
    ) -> float:
        """
        Compute recency score with decay.
        
        Recent = 1.0, decays to min_score over decay_hours.
        """
        if not timestamp:
            return 0.5  # Default for no timestamp
        
        try:
            ts = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            now = datetime.now(ts.tzinfo) if ts.tzinfo else datetime.now()
            age_hours = (now - ts).total_seconds() / 3600
            
            # Linear decay
            decay = age_hours / cfg.recency_decay_hours
            score = max(cfg.recency_min_score, 1.0 - decay)
            
            return score
        except Exception:
            return 0.5
    
    def _get_node_relations(self, node_id: str) -> List[Dict]:
        """Get formatted relations for a node."""
        relations = []
        
        edges = self.graph.get_edges(node_id, direction="both")
        for edge in edges:
            source = edge.get("source", node_id)
            target = edge.get("target", "?")
            relation = edge.get("relation", "RELATED_TO")
            
            if source == node_id:
                relations.append({node_id: f"{relation} → {target}"})
            else:
                relations.append({source: f"{relation} → {node_id}"})
        
        return relations
    
    def _convert_semantic_only(
        self, 
        semantic_results: List[Dict],
        cfg: FusionConfig
    ) -> List[FusionResult]:
        """Convert semantic-only results to FusionResult when no graph links exist."""
        results = []
        
        for result in semantic_results[:cfg.top_k]:
            semantic = self._normalize_semantic_score(result.get("similarity_score", 1.0))
            
            ts = result.get("metadata", {}).get("timestamp")
            recency = self._compute_recency_score(ts, cfg)
            
            # No structural info, so use semantic + recency only
            final_score = (cfg.alpha + cfg.beta) * semantic + cfg.gamma * recency
            
            results.append(FusionResult(
                node_id=result.get("metadata", {}).get("message_id", "unknown"),
                final_score=final_score,
                semantic_score=semantic,
                structural_score=0.0,
                recency_score=recency,
                is_seed=True,
                hop_distance=0,
                content=result.get("content"),
                node_type="Message",
                label=None,
                relations=[],
                metadata=result.get("metadata", {})
            ))
        
        return results
    
    def adjust_weights(
        self,
        query_type: str = "general"
    ) -> FusionConfig:
        """
        Get recommended weights based on query type.
        
        Args:
            query_type: "general", "factual", "recent", "exploratory"
            
        Returns:
            FusionConfig with adjusted weights
        """
        configs = {
            "general": FusionConfig(alpha=0.4, beta=0.4, gamma=0.2),
            "factual": FusionConfig(alpha=0.5, beta=0.4, gamma=0.1),  # Favor semantic
            "recent": FusionConfig(alpha=0.3, beta=0.2, gamma=0.5),   # Favor recency
            "exploratory": FusionConfig(alpha=0.3, beta=0.5, gamma=0.2),  # Favor structure
        }
        
        return configs.get(query_type, self.config)


# ==================== TESTING ====================

if __name__ == "__main__":
    print("FusionLayer module loaded successfully!")
    print(f"Default config: α={FusionConfig().alpha}, β={FusionConfig().beta}, γ={FusionConfig().gamma}")
