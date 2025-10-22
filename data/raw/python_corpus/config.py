import os
import json
import yaml
from typing import Any, Dict, Optional, Union
from pathlib import Path

class ConfigError(Exception):
    pass

class Config:
    def __init__(self, data: Dict[str, Any] = None):
        self._data = data or {}
        self._env_prefix = ""
    
    def set_env_prefix(self, prefix: str):
        self._env_prefix = prefix
    
    def get(self, key: str, default: Any = None) -> Any:
        # Try nested key access (e.g., "database.host")
        keys = key.split('.')
        value = self._data
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                # Try environment variable
                env_key = f"{self._env_prefix}{key.upper().replace('.', '_')}"
                env_value = os.getenv(env_key)
                if env_value is not None:
                    return self._convert_env_value(env_value)
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        keys = key.split('.')
        current = self._data
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    def _convert_env_value(self, value: str) -> Any:
        # Try to convert string environment values to appropriate types
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            return value
    
    def update(self, other: Union[Dict[str, Any], 'Config']):
        if isinstance(other, Config):
            other_data = other._data
        else:
            other_data = other
        
        self._merge_dict(self._data, other_data)
    
    def _merge_dict(self, base: Dict[str, Any], update: Dict[str, Any]):
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_dict(base[key], value)
            else:
                base[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        return self._data.copy()
    
    def __getitem__(self, key: str) -> Any:
        value = self.get(key)
        if value is None:
            raise KeyError(key)
        return value
    
    def __setitem__(self, key: str, value: Any):
        self.set(key, value)
    
    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None

class ConfigLoader:
    @staticmethod
    def from_file(filepath: Union[str, Path]) -> Config:
        path = Path(filepath)
        
        if not path.exists():
            raise ConfigError(f"Config file not found: {filepath}")
        
        with open(path, 'r', encoding='utf-8') as f:
            if path.suffix.lower() == '.json':
                data = json.load(f)
            elif path.suffix.lower() in ('.yml', '.yaml'):
                data = yaml.safe_load(f)
            else:
                raise ConfigError(f"Unsupported config file format: {path.suffix}")
        
        return Config(data)
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> Config:
        return Config(data)
    
    @staticmethod
    def from_env(prefix: str = "") -> Config:
        config = Config()
        config.set_env_prefix(prefix)
        
        # Load all environment variables with the prefix
        env_data = {}
        for key, value in os.environ.items():
            if key.startswith(prefix):
                clean_key = key[len(prefix):].lower().replace('_', '.')
                env_data[clean_key] = config._convert_env_value(value)
        
        config._data = env_data
        return config

class Settings:
    def __init__(self, config_files: list = None, env_prefix: str = ""):
        self.config = Config()
        self.config.set_env_prefix(env_prefix)
        
        # Load from files
        if config_files:
            for file_path in config_files:
                try:
                    file_config = ConfigLoader.from_file(file_path)
                    self.config.update(file_config)
                except ConfigError:
                    continue  # Skip missing files
        
        # Override with environment variables
        env_config = ConfigLoader.from_env(env_prefix)
        self.config.update(env_config)
    
    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)
    
    def __getattr__(self, name: str) -> Any:
        value = self.config.get(name)
        if value is None:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        return value

class DatabaseConfig:
    def __init__(self, config: Config):
        self.host = config.get('database.host', 'localhost')
        self.port = config.get('database.port', 5432)
        self.name = config.get('database.name', 'myapp')
        self.user = config.get('database.user', 'user')
        self.password = config.get('database.password', '')
    
    @property
    def url(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"

class AppConfig:
    def __init__(self, config_files: list = None):
        self.settings = Settings(config_files or ['config.yml', 'config.json'], 'MYAPP_')
        self.database = DatabaseConfig(self.settings.config)
    
    @property
    def debug(self) -> bool:
        return self.settings.get('debug', False)
    
    @property
    def secret_key(self) -> str:
        key = self.settings.get('secret_key')
        if not key:
            raise ConfigError("SECRET_KEY is required")
        return key
    
    @property
    def log_level(self) -> str:
        return self.settings.get('log_level', 'INFO')

# Example usage
def load_app_config() -> AppConfig:
    config_files = [
        'config/default.yml',
        'config/production.yml',
        'config/local.yml'
    ]
    
    return AppConfig(config_files)

# Configuration validation
def validate_config(config: Config, required_keys: list):
    missing_keys = []
    for key in required_keys:
        if key not in config:
            missing_keys.append(key)
    
    if missing_keys:
        raise ConfigError(f"Missing required configuration keys: {missing_keys}")

# Example configuration schema
DEFAULT_CONFIG = {
    'app': {
        'name': 'MyApp',
        'version': '1.0.0',
        'debug': False
    },
    'database': {
        'host': 'localhost',
        'port': 5432,
        'name': 'myapp',
        'pool_size': 10
    },
    'cache': {
        'type': 'redis',
        'host': 'localhost',
        'port': 6379,
        'ttl': 3600
    }
}
