import requests
from typing import Optional, Dict, Any
from utils import Logger, read_config
from database import UserRepository, User

class APIClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({'Authorization': f'Bearer {api_key}'})
    
    def get(self, endpoint: str) -> Optional[Dict[str, Any]]:
        try:
            response = self.session.get(f"{self.base_url}/{endpoint}")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            Logger("api").error(f"GET {endpoint} failed: {e}")
            return None
    
    def post(self, endpoint: str, data: Dict[str, Any]) -> bool:
        try:
            response = self.session.post(f"{self.base_url}/{endpoint}", json=data)
            response.raise_for_status()
            return True
        except requests.RequestException as e:
            Logger("api").error(f"POST {endpoint} failed: {e}")
            return False

class UserService:
    def __init__(self, config_path: str):
        self.config = read_config(config_path)
        if not self.config:
            raise ValueError("Invalid config")
        
        self.repo = UserRepository(self.config['db_path'])
        self.api = APIClient(self.config['api_url'], self.config['api_key'])
        self.logger = Logger("UserService")
    
    def create_user(self, name: str, email: str) -> Optional[int]:
        user = User(0, name, email)
        
        if not user.validate_email():
            self.logger.error(f"Invalid email: {email}")
            return None
        
        try:
            user_id = self.repo.save(user)
            
            # Sync with external API
            sync_data = {'id': user_id, 'name': name, 'email': email}
            if self.api.post('users', sync_data):
                self.logger.info(f"Created user {user_id}")
                return user_id
            else:
                self.logger.error("Failed to sync user")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to create user: {e}")
            return None
    
    def get_user(self, user_id: int) -> Optional[User]:
        try:
            return self.repo.find_by_id(user_id)
        except ValueError:
            return None
