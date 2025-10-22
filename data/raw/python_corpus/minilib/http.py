import urllib.request
import urllib.parse
import json
from typing import Dict, Any, Optional

class Response:
    def __init__(self, status_code: int, text: str, headers: Dict[str, str]):
        self.status_code = status_code
        self.text = text
        self.headers = headers
    
    def json(self) -> Dict[str, Any]:
        return json.loads(self.text)
    
    def raise_for_status(self):
        if self.status_code >= 400:
            raise HTTPError(f"HTTP {self.status_code}")

class HTTPError(Exception):
    pass

class Session:
    def __init__(self):
        self.headers = {}
        self.cookies = {}
    
    def get(self, url: str, params: Optional[Dict[str, str]] = None) -> Response:
        if params:
            query = urllib.parse.urlencode(params)
            url = f"{url}?{query}"
        
        req = urllib.request.Request(url, headers=self.headers)
        
        try:
            with urllib.request.urlopen(req) as response:
                return Response(
                    response.getcode(),
                    response.read().decode('utf-8'),
                    dict(response.headers)
                )
        except urllib.error.HTTPError as e:
            return Response(e.code, e.read().decode('utf-8'), {})
    
    def post(self, url: str, data: Optional[Dict[str, Any]] = None) -> Response:
        json_data = json.dumps(data).encode('utf-8') if data else b''
        
        req = urllib.request.Request(
            url, 
            data=json_data,
            headers={**self.headers, 'Content-Type': 'application/json'},
            method='POST'
        )
        
        try:
            with urllib.request.urlopen(req) as response:
                return Response(
                    response.getcode(),
                    response.read().decode('utf-8'),
                    dict(response.headers)
                )
        except urllib.error.HTTPError as e:
            return Response(e.code, e.read().decode('utf-8'), {})

def get(url: str, **kwargs) -> Response:
    session = Session()
    return session.get(url, **kwargs)

def post(url: str, **kwargs) -> Response:
    session = Session()
    return session.post(url, **kwargs)
