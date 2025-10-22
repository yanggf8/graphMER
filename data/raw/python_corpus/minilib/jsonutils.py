import json
from typing import Any, Dict, List, Union
from pathlib import Path

JsonValue = Union[str, int, float, bool, None, Dict[str, Any], List[Any]]

class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)

def load_json(path: Union[str, Path]) -> JsonValue:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data: JsonValue, path: Union[str, Path], indent: int = 2):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, cls=JSONEncoder, ensure_ascii=False)

def merge_json(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    result = base.copy()
    
    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_json(result[key], value)
        else:
            result[key] = value
    
    return result

def flatten_json(data: Dict[str, Any], separator: str = '.') -> Dict[str, Any]:
    def _flatten(obj, parent_key=''):
        items = []
        
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_key = f"{parent_key}{separator}{k}" if parent_key else k
                items.extend(_flatten(v, new_key).items())
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                new_key = f"{parent_key}{separator}{i}" if parent_key else str(i)
                items.extend(_flatten(v, new_key).items())
        else:
            return {parent_key: obj}
        
        return dict(items)
    
    return _flatten(data)

class JSONPath:
    def __init__(self, path: str):
        self.path = path.split('.')
    
    def get(self, data: JsonValue) -> Any:
        current = data
        
        for key in self.path:
            if isinstance(current, dict) and key in current:
                current = current[key]
            elif isinstance(current, list) and key.isdigit():
                idx = int(key)
                if 0 <= idx < len(current):
                    current = current[idx]
                else:
                    return None
            else:
                return None
        
        return current
    
    def set(self, data: Dict[str, Any], value: Any):
        current = data
        
        for key in self.path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[self.path[-1]] = value
