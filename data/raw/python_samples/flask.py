import json
import re
from typing import Dict, Any, Callable, List, Optional, Tuple
from urllib.parse import parse_qs
from functools import wraps

class Request:
    def __init__(self, environ: Dict[str, Any]):
        self.environ = environ
        self.method = environ.get('REQUEST_METHOD', 'GET')
        self.path = environ.get('PATH_INFO', '/')
        self.query_string = environ.get('QUERY_STRING', '')
        self._json = None
        self._form = None
    
    @property
    def args(self) -> Dict[str, str]:
        return parse_qs(self.query_string)
    
    @property
    def json(self) -> Dict[str, Any]:
        if self._json is None:
            content_length = int(self.environ.get('CONTENT_LENGTH', 0))
            if content_length > 0:
                body = self.environ['wsgi.input'].read(content_length)
                self._json = json.loads(body.decode('utf-8'))
        return self._json
    
    @property
    def form(self) -> Dict[str, str]:
        if self._form is None:
            content_type = self.environ.get('CONTENT_TYPE', '')
            if 'application/x-www-form-urlencoded' in content_type:
                content_length = int(self.environ.get('CONTENT_LENGTH', 0))
                if content_length > 0:
                    body = self.environ['wsgi.input'].read(content_length)
                    self._form = parse_qs(body.decode('utf-8'))
        return self._form or {}

class Response:
    def __init__(self, response: str, status: int = 200, headers: Dict[str, str] = None):
        self.response = response
        self.status = status
        self.headers = headers or {}
    
    def __iter__(self):
        yield self.response.encode('utf-8')

class Flask:
    def __init__(self, name: str):
        self.name = name
        self.routes = {}
        self.before_request_funcs = []
        self.after_request_funcs = []
        self.error_handlers = {}
    
    def route(self, rule: str, methods: List[str] = None):
        if methods is None:
            methods = ['GET']
        
        def decorator(f: Callable):
            self.add_url_rule(rule, f.__name__, f, methods)
            return f
        return decorator
    
    def add_url_rule(self, rule: str, endpoint: str, view_func: Callable, methods: List[str]):
        pattern = self._rule_to_regex(rule)
        for method in methods:
            key = (method, pattern)
            self.routes[key] = (view_func, rule)
    
    def _rule_to_regex(self, rule: str) -> str:
        # Convert Flask-style routes to regex
        pattern = rule
        pattern = re.sub(r'<(\w+)>', r'(?P<\1>[^/]+)', pattern)
        pattern = f'^{pattern}$'
        return pattern
    
    def before_request(self, f: Callable):
        self.before_request_funcs.append(f)
        return f
    
    def after_request(self, f: Callable):
        self.after_request_funcs.append(f)
        return f
    
    def errorhandler(self, code: int):
        def decorator(f: Callable):
            self.error_handlers[code] = f
            return f
        return decorator
    
    def dispatch_request(self, request: Request) -> Response:
        # Run before_request functions
        for func in self.before_request_funcs:
            result = func()
            if result is not None:
                return result
        
        # Find matching route
        for (method, pattern), (view_func, rule) in self.routes.items():
            if method == request.method:
                match = re.match(pattern, request.path)
                if match:
                    kwargs = match.groupdict()
                    try:
                        result = view_func(**kwargs)
                        if isinstance(result, str):
                            response = Response(result)
                        elif isinstance(result, tuple):
                            response = Response(*result)
                        else:
                            response = result
                        
                        # Run after_request functions
                        for func in self.after_request_funcs:
                            response = func(response) or response
                        
                        return response
                    except Exception as e:
                        return self.handle_exception(e)
        
        return self.handle_404()
    
    def handle_404(self) -> Response:
        if 404 in self.error_handlers:
            return Response(self.error_handlers[404](), 404)
        return Response("Not Found", 404)
    
    def handle_exception(self, e: Exception) -> Response:
        if 500 in self.error_handlers:
            return Response(self.error_handlers[500](e), 500)
        return Response("Internal Server Error", 500)
    
    def wsgi_app(self, environ: Dict[str, Any], start_response: Callable):
        request = Request(environ)
        response = self.dispatch_request(request)
        
        status = f"{response.status} OK"
        headers = list(response.headers.items())
        headers.append(('Content-Type', 'text/html'))
        
        start_response(status, headers)
        return response
    
    def run(self, host: str = '127.0.0.1', port: int = 5000, debug: bool = False):
        print(f"Running on http://{host}:{port}")
        # In a real implementation, this would start a WSGI server

# Utility functions
def jsonify(data: Dict[str, Any]) -> Response:
    json_str = json.dumps(data)
    response = Response(json_str)
    response.headers['Content-Type'] = 'application/json'
    return response

def redirect(location: str, code: int = 302) -> Response:
    response = Response("", code)
    response.headers['Location'] = location
    return response

def abort(code: int):
    raise HTTPException(code)

class HTTPException(Exception):
    def __init__(self, code: int):
        self.code = code
        super().__init__(f"HTTP {code}")

# Example app
app = Flask(__name__)

@app.route('/')
def index():
    return "Hello World!"

@app.route('/user/<name>')
def user_profile(name: str):
    return f"User: {name}"

@app.route('/api/data', methods=['GET', 'POST'])
def api_data():
    if request.method == 'POST':
        return jsonify({"status": "created"})
    return jsonify({"data": "example"})
