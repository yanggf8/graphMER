import urllib.request
import urllib.parse
import urllib.error
import json
import ssl
from typing import Dict, Any, Optional, Union
from http.cookiejar import CookieJar

class Response:
    def __init__(self, status_code: int, content: bytes, headers: Dict[str, str], url: str):
        self.status_code = status_code
        self.content = content
        self.headers = headers
        self.url = url
        self._text = None
        self._json = None
    
    @property
    def text(self) -> str:
        if self._text is None:
            self._text = self.content.decode('utf-8')
        return self._text
    
    def json(self) -> Dict[str, Any]:
        if self._json is None:
            self._json = json.loads(self.text)
        return self._json
    
    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 400
    
    def raise_for_status(self):
        if not self.ok:
            raise HTTPError(f"HTTP {self.status_code} Error", response=self)

class HTTPError(Exception):
    def __init__(self, message: str, response: Response = None):
        super().__init__(message)
        self.response = response

class Session:
    def __init__(self):
        self.headers = {}
        self.cookies = CookieJar()
        self.verify = True
        self.timeout = 30
    
    def prepare_request(self, method: str, url: str, **kwargs) -> urllib.request.Request:
        headers = {**self.headers, **kwargs.get('headers', {})}
        
        data = kwargs.get('data')
        json_data = kwargs.get('json')
        
        if json_data is not None:
            data = json.dumps(json_data).encode('utf-8')
            headers['Content-Type'] = 'application/json'
        elif data and isinstance(data, dict):
            data = urllib.parse.urlencode(data).encode('utf-8')
            headers['Content-Type'] = 'application/x-www-form-urlencoded'
        
        params = kwargs.get('params')
        if params:
            query = urllib.parse.urlencode(params)
            url = f"{url}?{query}"
        
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        return req
    
    def send_request(self, req: urllib.request.Request) -> Response:
        try:
            if not self.verify:
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context))
            else:
                opener = urllib.request.build_opener()
            
            opener.add_handler(urllib.request.HTTPCookieProcessor(self.cookies))
            
            with opener.open(req, timeout=self.timeout) as response:
                content = response.read()
                headers = dict(response.headers)
                return Response(response.getcode(), content, headers, response.url)
                
        except urllib.error.HTTPError as e:
            content = e.read()
            headers = dict(e.headers) if hasattr(e, 'headers') else {}
            return Response(e.code, content, headers, req.full_url)
        except urllib.error.URLError as e:
            raise HTTPError(f"Connection error: {e}")
    
    def request(self, method: str, url: str, **kwargs) -> Response:
        req = self.prepare_request(method, url, **kwargs)
        return self.send_request(req)
    
    def get(self, url: str, **kwargs) -> Response:
        return self.request('GET', url, **kwargs)
    
    def post(self, url: str, **kwargs) -> Response:
        return self.request('POST', url, **kwargs)
    
    def put(self, url: str, **kwargs) -> Response:
        return self.request('PUT', url, **kwargs)
    
    def delete(self, url: str, **kwargs) -> Response:
        return self.request('DELETE', url, **kwargs)
    
    def patch(self, url: str, **kwargs) -> Response:
        return self.request('PATCH', url, **kwargs)

# Module-level convenience functions
_session = Session()

def get(url: str, **kwargs) -> Response:
    return _session.get(url, **kwargs)

def post(url: str, **kwargs) -> Response:
    return _session.post(url, **kwargs)

def put(url: str, **kwargs) -> Response:
    return _session.put(url, **kwargs)

def delete(url: str, **kwargs) -> Response:
    return _session.delete(url, **kwargs)

def patch(url: str, **kwargs) -> Response:
    return _session.patch(url, **kwargs)
