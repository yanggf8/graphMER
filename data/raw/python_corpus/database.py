from abc import ABC, abstractmethod
from typing import List, Dict, Union, Generic, TypeVar
import sqlite3

T = TypeVar('T')

class Repository(ABC, Generic[T]):
    @abstractmethod
    def save(self, entity: T) -> int:
        pass
    
    @abstractmethod
    def find_by_id(self, id: int) -> T:
        pass
    
    @abstractmethod
    def find_all(self) -> List[T]:
        pass

class User:
    def __init__(self, id: int, name: str, email: str):
        self.id = id
        self.name = name
        self.email = email
    
    def validate_email(self) -> bool:
        return '@' in self.email and '.' in self.email

class UserRepository(Repository[User]):
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self._create_table()
    
    def _create_table(self):
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL
            )
        ''')
    
    def save(self, user: User) -> int:
        if not user.validate_email():
            raise ValueError("Invalid email")
        
        cursor = self.conn.execute(
            'INSERT INTO users (name, email) VALUES (?, ?)',
            (user.name, user.email)
        )
        self.conn.commit()
        return cursor.lastrowid
    
    def find_by_id(self, id: int) -> User:
        cursor = self.conn.execute('SELECT * FROM users WHERE id = ?', (id,))
        row = cursor.fetchone()
        if row:
            return User(row[0], row[1], row[2])
        raise ValueError(f"User {id} not found")
    
    def find_all(self) -> List[User]:
        cursor = self.conn.execute('SELECT * FROM users')
        return [User(row[0], row[1], row[2]) for row in cursor.fetchall()]
