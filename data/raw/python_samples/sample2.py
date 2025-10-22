import json
from typing import Dict, List

class DataProcessor:
    def __init__(self, config: Dict):
        self.config = config
        self.cache = {}
    
    def process_data(self, items: List[str]) -> Dict:
        results = {}
        for item in items:
            try:
                parsed = json.loads(item)
                results[parsed['id']] = self.transform(parsed)
            except ValueError as e:
                self.handle_error(e)
        return results
    
    def transform(self, data: Dict) -> Dict:
        return {
            'processed': True,
            'value': data.get('value', 0) * 2
        }
    
    def handle_error(self, error: Exception):
        print(f"Error: {error}")
