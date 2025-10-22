from typing import List, Tuple, Optional
import heapq
from collections import defaultdict, deque

class Graph:
    def __init__(self):
        self.edges = defaultdict(list)
        self.weights = {}
    
    def add_edge(self, u: int, v: int, weight: float = 1.0):
        self.edges[u].append(v)
        self.weights[(u, v)] = weight
    
    def dijkstra(self, start: int) -> Dict[int, float]:
        distances = defaultdict(lambda: float('inf'))
        distances[start] = 0
        pq = [(0, start)]
        
        while pq:
            current_dist, u = heapq.heappop(pq)
            
            if current_dist > distances[u]:
                continue
            
            for v in self.edges[u]:
                weight = self.weights.get((u, v), 1.0)
                distance = current_dist + weight
                
                if distance < distances[v]:
                    distances[v] = distance
                    heapq.heappush(pq, (distance, v))
        
        return dict(distances)
    
    def bfs(self, start: int) -> List[int]:
        visited = set()
        queue = deque([start])
        result = []
        
        while queue:
            node = queue.popleft()
            if node not in visited:
                visited.add(node)
                result.append(node)
                
                for neighbor in self.edges[node]:
                    if neighbor not in visited:
                        queue.append(neighbor)
        
        return result

def quicksort(arr: List[int]) -> List[int]:
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quicksort(left) + middle + quicksort(right)

def binary_search(arr: List[int], target: int) -> Optional[int]:
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return None

class TreeNode:
    def __init__(self, val: int):
        self.val = val
        self.left = None
        self.right = None

def inorder_traversal(root: Optional[TreeNode]) -> List[int]:
    if not root:
        return []
    
    result = []
    result.extend(inorder_traversal(root.left))
    result.append(root.val)
    result.extend(inorder_traversal(root.right))
    
    return result
