import sqlite3
from typing import Any, Dict, List, Optional, Type, Union
from abc import ABC, abstractmethod

class Column:
    def __init__(self, type_: str, primary_key: bool = False, nullable: bool = True, 
                 unique: bool = False, default: Any = None):
        self.type = type_
        self.primary_key = primary_key
        self.nullable = nullable
        self.unique = unique
        self.default = default
        self.name = None
    
    def __set_name__(self, owner, name):
        self.name = name

class Integer(Column):
    def __init__(self, **kwargs):
        super().__init__('INTEGER', **kwargs)

class String(Column):
    def __init__(self, length: int = None, **kwargs):
        super().__init__(f'VARCHAR({length})' if length else 'TEXT', **kwargs)
        self.length = length

class Boolean(Column):
    def __init__(self, **kwargs):
        super().__init__('BOOLEAN', **kwargs)

class DateTime(Column):
    def __init__(self, **kwargs):
        super().__init__('DATETIME', **kwargs)

class ModelMeta(type):
    def __new__(cls, name, bases, attrs):
        columns = {}
        for key, value in attrs.items():
            if isinstance(value, Column):
                columns[key] = value
                value.name = key
        
        attrs['__columns__'] = columns
        attrs['__table_name__'] = attrs.get('__tablename__', name.lower())
        return super().__new__(cls, name, bases, attrs)

class Model(metaclass=ModelMeta):
    def __init__(self, **kwargs):
        for name, column in self.__columns__.items():
            value = kwargs.get(name, column.default)
            setattr(self, name, value)
    
    @classmethod
    def create_table(cls, engine: 'Engine'):
        columns_sql = []
        for name, column in cls.__columns__.items():
            col_def = f"{name} {column.type}"
            if column.primary_key:
                col_def += " PRIMARY KEY"
            if not column.nullable:
                col_def += " NOT NULL"
            if column.unique:
                col_def += " UNIQUE"
            columns_sql.append(col_def)
        
        sql = f"CREATE TABLE IF NOT EXISTS {cls.__table_name__} ({', '.join(columns_sql)})"
        engine.execute(sql)
    
    def save(self, session: 'Session'):
        session.add(self)
        session.commit()
    
    def to_dict(self) -> Dict[str, Any]:
        return {name: getattr(self, name) for name in self.__columns__.keys()}

class Query:
    def __init__(self, model_class: Type[Model], session: 'Session'):
        self.model_class = model_class
        self.session = session
        self._filters = []
        self._order_by = []
        self._limit_value = None
    
    def filter(self, **kwargs):
        new_query = Query(self.model_class, self.session)
        new_query._filters = self._filters + list(kwargs.items())
        new_query._order_by = self._order_by[:]
        new_query._limit_value = self._limit_value
        return new_query
    
    def order_by(self, column: str, desc: bool = False):
        new_query = Query(self.model_class, self.session)
        new_query._filters = self._filters[:]
        new_query._order_by = self._order_by + [(column, desc)]
        new_query._limit_value = self._limit_value
        return new_query
    
    def limit(self, count: int):
        new_query = Query(self.model_class, self.session)
        new_query._filters = self._filters[:]
        new_query._order_by = self._order_by[:]
        new_query._limit_value = count
        return new_query
    
    def all(self) -> List[Model]:
        sql = f"SELECT * FROM {self.model_class.__table_name__}"
        params = []
        
        if self._filters:
            where_clauses = []
            for key, value in self._filters:
                where_clauses.append(f"{key} = ?")
                params.append(value)
            sql += f" WHERE {' AND '.join(where_clauses)}"
        
        if self._order_by:
            order_clauses = []
            for column, desc in self._order_by:
                order_clauses.append(f"{column} {'DESC' if desc else 'ASC'}")
            sql += f" ORDER BY {', '.join(order_clauses)}"
        
        if self._limit_value:
            sql += f" LIMIT {self._limit_value}"
        
        cursor = self.session.engine.execute(sql, params)
        results = []
        
        for row in cursor.fetchall():
            kwargs = {}
            for i, (name, _) in enumerate(self.model_class.__columns__.items()):
                kwargs[name] = row[i]
            results.append(self.model_class(**kwargs))
        
        return results
    
    def first(self) -> Optional[Model]:
        results = self.limit(1).all()
        return results[0] if results else None
    
    def count(self) -> int:
        sql = f"SELECT COUNT(*) FROM {self.model_class.__table_name__}"
        params = []
        
        if self._filters:
            where_clauses = []
            for key, value in self._filters:
                where_clauses.append(f"{key} = ?")
                params.append(value)
            sql += f" WHERE {' AND '.join(where_clauses)}"
        
        cursor = self.session.engine.execute(sql, params)
        return cursor.fetchone()[0]

class Engine:
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.connection = sqlite3.connect(database_url)
    
    def execute(self, sql: str, params: List[Any] = None):
        return self.connection.execute(sql, params or [])
    
    def commit(self):
        self.connection.commit()
    
    def close(self):
        self.connection.close()

class Session:
    def __init__(self, engine: Engine):
        self.engine = engine
        self._new_objects = []
        self._dirty_objects = []
    
    def add(self, obj: Model):
        self._new_objects.append(obj)
    
    def query(self, model_class: Type[Model]) -> Query:
        return Query(model_class, self)
    
    def commit(self):
        # Insert new objects
        for obj in self._new_objects:
            columns = list(obj.__columns__.keys())
            values = [getattr(obj, col) for col in columns]
            placeholders = ', '.join(['?' for _ in columns])
            
            sql = f"INSERT INTO {obj.__table_name__} ({', '.join(columns)}) VALUES ({placeholders})"
            self.engine.execute(sql, values)
        
        self.engine.commit()
        self._new_objects.clear()
        self._dirty_objects.clear()
    
    def rollback(self):
        self._new_objects.clear()
        self._dirty_objects.clear()
    
    def close(self):
        self.rollback()

def create_engine(database_url: str) -> Engine:
    return Engine(database_url)

def sessionmaker(engine: Engine):
    def make_session():
        return Session(engine)
    return make_session

# Example models
class User(Model):
    __tablename__ = 'users'
    
    id = Integer(primary_key=True)
    username = String(50, nullable=False, unique=True)
    email = String(100, nullable=False)
    active = Boolean(default=True)

class Post(Model):
    __tablename__ = 'posts'
    
    id = Integer(primary_key=True)
    title = String(200, nullable=False)
    content = String()
    user_id = Integer(nullable=False)
    published = Boolean(default=False)

# Usage example
def setup_database():
    engine = create_engine('example.db')
    
    User.create_table(engine)
    Post.create_table(engine)
    
    SessionLocal = sessionmaker(engine)
    session = SessionLocal()
    
    # Create a user
    user = User(username='john_doe', email='john@example.com')
    user.save(session)
    
    # Query users
    users = session.query(User).filter(active=True).all()
    first_user = session.query(User).filter(username='john_doe').first()
    
    session.close()
    engine.close()
