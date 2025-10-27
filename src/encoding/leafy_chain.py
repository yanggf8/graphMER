"""
Leafy Chain Graph Encoding - Core GraphMER algorithm for linearizing knowledge graphs.
Converts graph triples into linearized sequences while preserving graph topology.
"""
from typing import List, Tuple, Dict, Set
from collections import defaultdict, deque


class LeafyChainEncoder:
    """Implements the Leafy Chain Graph Encoding algorithm from GraphMER paper."""
    
    def __init__(self):
        self.chain_separator = "[CHAIN]"
        self.relation_prefix = "[REL]"
        self.entity_prefix = "[ENT]"
    
    def linearize_graph(self, triples: List[Tuple[str, str, str]]) -> List[str]:
        """
        Convert KG triples to linearized chain sequences.
        
        Args:
            triples: List of (head, relation, tail) triples
            
        Returns:
            Linearized token sequence preserving graph structure
        """
        if not triples:
            return []
        
        # Build adjacency graph
        graph = self._build_graph(triples)
        
        # Find chain sequences using DFS
        chains = self._extract_chains(graph, triples)
        
        # Linearize chains into token sequence
        return self._chains_to_tokens(chains)
    
    def _build_graph(self, triples: List[Tuple[str, str, str]]) -> Dict[str, List[Tuple[str, str]]]:
        """Build adjacency list representation."""
        graph = defaultdict(list)
        for head, relation, tail in triples:
            graph[head].append((relation, tail))
        return graph
    
    def _extract_chains(self, graph: Dict[str, List[Tuple[str, str]]], 
                       triples: List[Tuple[str, str, str]]) -> List[List[Tuple[str, str, str]]]:
        """Extract chain sequences using modified DFS."""
        visited = set()
        chains = []
        
        # Find root nodes (nodes with no incoming edges)
        all_tails = {tail for _, _, tail in triples}
        all_heads = {head for head, _, _ in triples}
        roots = all_heads - all_tails
        
        # If no clear roots, start from most connected nodes
        if not roots:
            node_counts = defaultdict(int)
            for head, _, tail in triples:
                node_counts[head] += 1
                node_counts[tail] += 1
            roots = [max(node_counts.items(), key=lambda x: x[1])[0]]
        
        # Extract chains from each root
        for root in roots:
            if root not in visited:
                chain = self._dfs_chain(root, graph, visited)
                if chain:
                    chains.append(chain)
        
        # Handle remaining unvisited triples
        remaining_triples = [t for t in triples if not any(
            t[0] in visited and t[2] in visited for _ in [None]
        )]
        
        for head, relation, tail in remaining_triples:
            if head not in visited or tail not in visited:
                chains.append([(head, relation, tail)])
                visited.add(head)
                visited.add(tail)
        
        return chains
    
    def _dfs_chain(self, node: str, graph: Dict[str, List[Tuple[str, str]]], 
                   visited: Set[str]) -> List[Tuple[str, str, str]]:
        """Extract single chain using DFS."""
        if node in visited:
            return []
        
        visited.add(node)
        chain = []
        
        # Follow edges to build chain
        for relation, tail in graph.get(node, []):
            if tail not in visited:
                chain.append((node, relation, tail))
                # Recursively extend chain
                sub_chain = self._dfs_chain(tail, graph, visited)
                chain.extend(sub_chain)
            else:
                # Add edge even if tail is visited (for completeness)
                chain.append((node, relation, tail))
        
        return chain
    
    def _chains_to_tokens(self, chains: List[List[Tuple[str, str, str]]]) -> List[str]:
        """Convert chains to linearized token sequence."""
        tokens = []
        
        for i, chain in enumerate(chains):
            if i > 0:
                tokens.append(self.chain_separator)
            
            for j, (head, relation, tail) in enumerate(chain):
                if j == 0:
                    # Start of chain - add head entity
                    tokens.append(f"{self.entity_prefix}{head}")
                
                # Add relation and tail
                tokens.append(f"{self.relation_prefix}{relation}")
                tokens.append(f"{self.entity_prefix}{tail}")
        
        return tokens
    
    def get_relation_ids(self, tokens: List[str], relation_vocab: Dict[str, int]) -> List[int]:
        """Generate relation IDs for tokens (for attention mechanism)."""
        rel_ids = []
        current_chain_id = 1
        
        for token in tokens:
            if token == self.chain_separator:
                current_chain_id += 1
                rel_ids.append(0)  # Separator gets ID 0
            elif token.startswith(self.relation_prefix):
                rel_name = token[len(self.relation_prefix):]
                rel_id = relation_vocab.get(rel_name, 0)
                rel_ids.append(rel_id)
            elif token.startswith(self.entity_prefix):
                rel_ids.append(current_chain_id)  # Entities get chain ID
            else:
                rel_ids.append(0)  # Default
        
        return rel_ids


def test_leafy_chain():
    """Test the implementation with sample triples."""
    encoder = LeafyChainEncoder()
    
    # Sample software engineering triples
    triples = [
        ("MyClass", "inherits_from", "BaseClass"),
        ("BaseClass", "contains", "method1"),
        ("MyClass", "contains", "method2"),
        ("method1", "calls", "helper_func"),
        ("method2", "uses", "variable_x")
    ]
    
    tokens = encoder.linearize_graph(triples)
    print("Linearized tokens:", tokens)
    
    # Test relation IDs
    relation_vocab = {"inherits_from": 1, "contains": 2, "calls": 3, "uses": 4}
    rel_ids = encoder.get_relation_ids(tokens, relation_vocab)
    print("Relation IDs:", rel_ids)


if __name__ == "__main__":
    test_leafy_chain()
