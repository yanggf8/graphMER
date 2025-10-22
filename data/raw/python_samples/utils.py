import os
import json
from typing import Optional, Dict, Any
from datetime import datetime

class Logger:
    def __init__(self, name: str):
        self.name = name
        self.handlers = []
    
    def info(self, msg: str):
        self._log("INFO", msg)
    
    def error(self, msg: str):
        self._log("ERROR", msg)
    
    def _log(self, level: str, msg: str):
        timestamp = datetime.now()
        for handler in self.handlers:
            handler.write(f"{timestamp} [{level}] {msg}")

def read_config(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        Logger("config").error(f"Failed to read {path}: {e}")
        return None

def validate_data(data: Dict[str, Any]) -> bool:
    required_fields = ['id', 'type', 'value']
    return all(field in data for field in required_fields)
